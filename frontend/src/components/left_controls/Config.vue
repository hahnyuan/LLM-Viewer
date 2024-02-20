<template>
    <h2>Inference Config</h2>
    <!-- radio，可选 decode prefill 默认decode-->
    <div>
        Stage:
        <input type="radio" v-model="inference_stage" id="decode" value="decode" checked>
        <label for="decode">Decode</label>
        <input type="radio" v-model="inference_stage" id="prefill" value="prefill">
        <label for="prefill">Prefill</label>
    </div>
    <div class="slider">
        Batchsize:
        <input type="range" min="1" max="256" value="1" v-model.lazy="batch_size" oninput="batch_size.innerText = this.value">
        <p id="batch_size">1</p>
    </div>
    <div class="slider">
        Seqence Length:
        <input type="range" min="1" max="4096" value="1024" v-model.lazy="seq_length" oninput="seq_length.innerText = this.value">
        <p id="seq_length">1024</p>
    </div>
    <!-- <div class="slider">
        Generation Length:
        <input type="range" min="1" max="4096" value="1024" oninput="gen_length.innerText = this.value">
        <p id="gen_length">1</p>
    </div> -->
    <h2>Optimization Config</h2>
    <div class="slider">
        Weight Quantization:
        <select v-model="w_quant">
            <option value="FP16">FP16</option>
            <option value="INT8">INT8</option>
            <option value="INT4">INT4</option>
        </select>
    </div>
    <div class="slider">
        Activation Quantization
        <select v-model="a_quant">
            <option value="FP16">FP16</option>
            <option value="INT8">INT8</option>
            <option value="INT4">INT4</option>
        </select>
    </div>
    <div class="slider">
        KV Cache Quantization
        <select v-model="kv_quant">
            <option value="FP16">FP16</option>
            <option value="INT8">INT8</option>
            <option value="INT4">INT4</option>
        </select>
    </div>
    <!-- <div class="slider">
        Use Flash Attention
        <input type="checkbox">
    </div>
    <div class="slider">
        Decoding Method
        <select>
            <option value="Greedy">Greedy</option>
        </select>
    </div> -->
</template>

<script setup>
import { inject, ref, watch, computed } from 'vue';

const ip_port = inject('settingsData').value.ip_port
const serverStatus = inject('serverStatus');
const selectedNodeInfo = inject('selectedNodeInfo');
const graphUpdateTrigger = inject('graphUpdateTrigger');


const InferenceConfig = inject('InferenceConfig');

const inference_stage = ref('');
const batch_size = ref(1);
const seq_length = ref(1);
const w_quant = ref('FP16');
const a_quant = ref('FP16');
const kv_quant = ref('FP16');

// 当inference_stage改变时，更新InferenceConfig.step
watch(inference_stage, (new_stage) => {
    console.log("inference_stage", new_stage)
    InferenceConfig.value.stage = new_stage
    graphUpdateTrigger.value += 1
})

watch(batch_size, (n) => {
    console.log("inference_stage", n)
    InferenceConfig.value.batch_size = n
    graphUpdateTrigger.value += 1
})

watch(seq_length, (n) => {
    console.log("seq_length", n)
    InferenceConfig.value.seq_length = n
    graphUpdateTrigger.value += 1
})

watch(w_quant, (n) => {
    console.log("w_quant", n)
    InferenceConfig.value.w_quant = n
    graphUpdateTrigger.value += 1
})

watch(a_quant, (n) => {
    console.log("a_quant", n)
    InferenceConfig.value.a_quant = n
    graphUpdateTrigger.value += 1
})

watch(kv_quant, (n) => {
    console.log("kv_quant", n)
    InferenceConfig.value.kv_quant = n
    graphUpdateTrigger.value += 1
})

</script>

<style>
.input_config {
    /* 外边有一圈框框 */
    border: 1px solid #ccc;
    margin: 3px;
}

.hover_color {
    /* 当鼠标悬停时，改变文字颜色 */
    color: #0000ff;
    cursor: pointer;

}
</style>