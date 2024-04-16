from .llm_parser import LLMParser


class Chatglm3Parser(LLMParser):
    def __init__(self, model_id, args: dict):
        super().__init__(model_id, args)
        
        if getattr(self.p,"multi_query_attention"):
            self.p.num_key_value_heads= getattr(self.p, "multi_query_group_num")
        else:
            self.p.num_key_value_heads= getattr(self.p, "num_attention_heads")
        self.p.num_hidden_layers=getattr(self.p, "num_layers")
        self.p.intermediate_size=getattr(self.p, "ffn_hidden_size")
        self.p.vocab_size=getattr(self.p, "padded_vocab_size")
