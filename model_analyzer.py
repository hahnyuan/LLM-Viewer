import os
import importlib
from hardwares.hardware_params import hardware_params
from roofline_model import roofline_analyze, evaluate_op
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from utils import str_number, str_number_time
from op_handlers import OpContext, OP_HANDLERS

ALL_DATA_NAMES = [
    "OPs",
    "memory_access",
    "load_weight",
    "load_act",
    "store_act",
    "load_kv_cache",
    "store_kv_cache",
    "inference_time",
]


class ModelAnalyzer:
    def __init__(self, model_id, hardware, config_file=None, source="huggingface"):
        """
        source: 'huggingface' or 'DiT'
        """
        self.model_id = model_id
        self.hardware = hardware
        if config_file is None:
            # get the current file directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # auto search the config
            for file in os.listdir(current_dir + "/configs"):
                if file.endswith(".py") and file.replace(".py", "") in model_id:
                    config_file = "configs/" + file
                # print(f"auto search config file {config_file} {file} {model_id}")
        assert config_file is not None, "config file is not found, please specify it manually."
        print(f"use config file {config_file} for {model_id}")
        if source == "huggingface":
            self.model_params = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        else:
            if not os.path.exists(f"model_params/{source}.py"):
                raise Exception(f"model_params/{source}.py is not found")
            # from model_params.DiT import model_params
            module = importlib.import_module(f"model_params.{source}")
            self.model_params = module.model_params[model_id]
        self.config = importlib.import_module(config_file.replace("/", ".").replace(".py", ""))

        # temporary variables
        self.results = None
        self.w_bit = None
        self.a_bit = None
        self.kv_bit = None
        self.batchsize = None
        self.seqlen = None

    def _analyze_to_results(
        self,
        stage,
        name,
        OPs=0,
        load_weight=0,
        load_act=0,
        store_act=0,
        load_kv_cache=0,
        store_kv_cache=0,
    ):

        bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
        self.results[stage][name] = evaluate_op(
            OPs, load_weight, load_act, store_act, load_kv_cache, store_kv_cache, bandwidth, max_OPS
        )

    def save_csv(self, save_path=None):
        if save_path is None:
            save_path = f"output/{self.model_id[:self.model_id.rfind('/')]}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path += f"{self.model_id[self.model_id.rfind('/'):]}"

        decode_file_name = f"{save_path}_decode.csv"
        prefill_file_name = f"{save_path}_prefill.csv"
        print(f"save to {decode_file_name} and {prefill_file_name}")

        for file_name, stage in [
            (decode_file_name, "decode"),
            (prefill_file_name, "prefill"),
        ]:
            with open(file_name, "a+") as f:

                f.write(
                    f"\n\n=== {self.model_id} {self.hardware} w_bit={self.w_bit} a_bit={self.a_bit} kv_bit={self.kv_bit} batchsize={self.batchsize} seqlen={self.seqlen} tp_size={self.tp_size} ===\n"
                )
                # legend
                f.write(
                    f"layer_name,OPs,Access,arithmetic_intensity,performance,bound,load_weight,load_act,store_act,load_kv_cache,store_kv_cache,inference_time\n"
                )
            with open(file_name, "a+") as f:
                for layer_name, result in self.results[stage].items():
                    f.write(
                        f"{layer_name},{str_number(result['OPs'])},{str_number(result['memory_access'])}B,{str_number(result['arithmetic_intensity'])},{str_number(result['performance'])},"
                        f"{result['bound']},{str_number(result['load_weight'])}B,{str_number(result['load_act'])}B,{str_number(result['store_act'])}B,{str_number(result['load_kv_cache'])}B,"
                        f"{str_number(result['store_kv_cache'])}B,{str_number_time(result['inference_time'])}s\n"
                    )

    def analyze(
        self,
        seqlen,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention=False,
        kv_token_ratio=1,
        tp_size: int = 1
    ):
        """
        seqlen: sequence length
        batchsize: batch size
        w_bit: weight bit
        a_bit: activation bit
        kv_bit: key and value bit. if it is None, it will be the same as a_bit
        use_flashattention: use flash attention/flash decoding
        kv_token_ratio: use this for KV compression
        tp_size: the number of devices for tensor parallelism to use

        return is a dict with the following format:
        {
            "decode": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "prefill": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "total_results": {
                "decode": {},
                "prefill": {}
            }
        }
        """
        assert seqlen > 0
        assert batchsize > 0
        self.results = {"decode": {}, "prefill": {}}
        if kv_bit is None:
            kv_bit = a_bit
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.batchsize = batchsize
        self.seqlen = seqlen
        self.tp_size = tp_size

        w_byte = self.w_bit / 8
        a_byte = self.a_bit / 8
        kv_byte = self.kv_bit / 8

        config = self.config
        model_params = self.model_params
        num_attention_heads = config.get_num_attention_heads(model_params)
        hidden_size = config.get_hidden_size(model_params)
        num_key_value_heads = config.get_num_key_value_heads(model_params)
        num_hidden_layers = config.get_num_hidden_layers(model_params)

        head_size = hidden_size // num_attention_heads
        _, _, onchip_buffer = self.get_hardware_info()
        ctx = OpContext(
            batchsize=batchsize,
            a_byte=a_byte,
            w_byte=w_byte,
            kv_byte=kv_byte,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_size=head_size,
            onchip_buffer=onchip_buffer,
        )

        # query tokens processed per step; key/value tokens attended over.
        # (decode processes 1 query token against the full context.)
        stage_seqlens = {
            "decode": (1, seqlen),
            "prefill": (seqlen, seqlen),
        }

        def dispatch(stage, name, op_type, **kw):
            q_seqlen, kv_seqlen = stage_seqlens[stage]
            res = OP_HANDLERS[op_type](ctx, q_seqlen, kv_seqlen, **kw)
            self._analyze_to_results(stage, name, **res)

        # Linear projections. Both stages receive the same set of layers, in
        # get_linear_layers() order (q, k, v, out, gate, up, down).
        for name, (ic, oc) in config.get_linear_layers(model_params, tp_size).items():
            is_kv_proj = name in ["k_proj", "v_proj"]
            for stage in ["decode", "prefill"]:
                dispatch(stage, name, "linear", ic=ic, oc=oc, is_kv_proj=is_kv_proj)

        # Remaining per-layer transformer ops. The decode block is fully emitted
        # before the prefill block, matching the original insertion order so the
        # aggregated totals stay bit-identical.
        for stage in ["decode", "prefill"]:
            if use_flashattention:
                dispatch(stage, "fused_attention", "fused_attention")
            else:
                # Preserve the legacy Q-activation head count (see qk_matmul):
                # attention-heads in decode, key-value-heads in prefill.
                legacy_q_heads = num_attention_heads if stage == "decode" else num_key_value_heads
                dispatch(stage, "qk_matmul", "qk_matmul", query_act_heads=legacy_q_heads)
                dispatch(stage, "sv_matmul", "sv_matmul")
                dispatch(stage, "softmax", "softmax")
            for name in config.get_norm_layers(model_params):
                dispatch(stage, name, "norm")
            for name in ["attn_add", "mlp_add"]:
                dispatch(stage, name, "add")
            for name in ["mlp_act"]:
                dispatch(stage, name, "act")

        # compute total
        total_results = {"decode": {}, "prefill": {}}
        for data_name in ALL_DATA_NAMES:
            total_results["decode"][data_name] = 0
            total_results["prefill"][data_name] = 0
        for stage in ["decode", "prefill"]:
            for layer_name, result in self.results[stage].items():
                for data_name in ALL_DATA_NAMES:
                    total_results[stage][data_name] += result[data_name] * num_hidden_layers

        # memory footprint
        weight_kv_footprint = total_results["prefill"]["load_weight"] + total_results["prefill"]["store_kv_cache"]
        decode_tmp_act = 0
        for layer_name, result in self.results["decode"].items():
            decode_tmp_act += result["store_act"]
        total_results["decode"]["memory_consumption"] = decode_tmp_act + weight_kv_footprint
        total_results["decode"]["memory_consumption_tmp_act"] = decode_tmp_act
        total_results["decode"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["decode"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]
        prefill_tmp_act = 0
        for layer_name, result in self.results["prefill"].items():
            prefill_tmp_act += result["store_act"]
        total_results["prefill"]["memory_consumption"] = prefill_tmp_act + weight_kv_footprint
        total_results["prefill"]["memory_consumption_tmp_act"] = prefill_tmp_act
        total_results["prefill"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["prefill"]["memory_consumption_kv_cache"] = total_results["prefill"]["store_kv_cache"]

        # lm_head
        name = "lm_head"
        args = {"batchsize": batchsize, "a_byte": a_byte, "w_byte": w_byte}
        for layer_info in self.config.post_process(self.model_params, args):
            self._analyze_to_results(**layer_info)
            for data_name in ALL_DATA_NAMES:
                total_results[layer_info["stage"]][data_name] += self.results[layer_info["stage"]][layer_info["name"]][
                    data_name
                ]
        # for stage in ["prefill", "decode"]:
        #     self._analyze_to_results(
        #         stage,
        #         name,
        #         OPs=batchsize * hidden_size * vocab_size * 1,
        #         load_weight=hidden_size * vocab_size,
        #         load_act=hidden_size * a_byte,
        #         store_act=vocab_size * a_byte,
        #         load_kv_cache=0,
        #         store_kv_cache=0,
        #     )
        #     for data_name in ALL_DATA_NAMES:
        #         total_results[stage][data_name] += self.results[stage][name][data_name]

        self.results["total_results"] = total_results
        return self.results

    def analyze_generate_task(
        self,
        prompt_len,
        gen_len,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention = False,
        tp_size: int = 1
    ):
        prefill_result = self.analyze(
            prompt_len,
            batchsize,
            w_bit,
            a_bit,
            kv_bit,
            use_flashattention=use_flashattention,
            tp_size=tp_size
        )
        prefill_time = inference_time = prefill_result["total_results"]["prefill"]["inference_time"]

        for i in range(prompt_len, prompt_len + gen_len):
            result = self.analyze(i, batchsize, w_bit, a_bit, kv_bit, use_flashattention=use_flashattention, tp_size=tp_size)
            inference_time += result["total_results"]["decode"]["inference_time"]
        return {"inference_time": inference_time, "prefill_time": prefill_time}

    def get_hardware_info(self):
        params = hardware_params[self.hardware]
        bandwidth = params["bandwidth"]
        # Pick the peak-throughput tier for the lowest common bitwidth the
        # hardware actually supports, falling back up the precision ladder when
        # a tier is absent (e.g. Hopper has no INT4; intel has only FP16).
        max_bit = max(self.w_bit, self.a_bit, self.kv_bit)
        if max_bit <= 4 and "INT4" in params:
            max_OPS = params["INT4"]
        elif max_bit <= 8 and "INT8" in params:
            max_OPS = params["INT8"]
        else:
            max_OPS = params["FP16"]
        onchip_buffer = params["onchip_buffer"]
        return bandwidth, max_OPS, onchip_buffer

    def get_model_info(self):
        if self.config.get_num_attention_heads(self.model_params) != self.config.get_num_key_value_heads(
            self.model_params
        ):
            GQA = True
        else:
            GQA = False

        info = {"GQA": GQA}  # group query attention
        return info
