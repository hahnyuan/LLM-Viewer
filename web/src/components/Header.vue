<template>
    <!-- 这里放7个按钮 -->
    <!-- 加粗 -->
    <div class="title">LLMViewer </div>
    <div class="header_button">
        <button @click="change_step('settings')" :class="{ active: step == 'settings' }">Settings</button>
        | Model:
        <button @click="change_step('parse_model')" :class="{ active: step == 'parse_model' }">LLaMA</button>
        <!-- <button @click="change_step('exec_opt')">执行图优化</button> -->
        <button @click="change_step('quant_config')" :class="{ active: step == 'quant_config' }">ChatGLM</button>
        <!-- <button @click="change_step('quant_dataset')" :class="{ active: step == 'quant_dataset' }">数据准备</button>
        <button @click="change_step('exec_quant')" :class="{ active: step == 'exec_quant' }">执行量化</button>
        <button @click="change_step('analyze')" :class="{ active: step == 'analyze' }">结果分析</button>
        <button @click="exit_quant()">结束退出</button> -->
        | Stage:
        <button @click="change_step('onnx_modifier')" :class="{ active: step == 'onnx_modifier' }">Prefill</button>
        <button @click="change_step('hm_onnx_viewer')" :class="{ active: step == 'hm_onnx_viewer' }">Decode</button>
    </div>
</template>

<script setup>
import { inject, ref, watch, computed } from 'vue';
const step = inject('step');
const settingsData = inject('settingsData');

function change_step(new_step) {
    if (new_step == "analyze") {
        alert("功能优化中，暂未开放")
        return
    }
    console.log("change_step", new_step)
    step.value = new_step
}


</script>

<style scoped>
.header_button button {
    font-size: 1.0rem;
    margin: 5px;
    padding: 5px;
    border-radius: 5px;
    border: 1px solid #000000;
    /* background-color: #fff; */
    /* color: #000; */
    cursor: pointer;
}

.header_button button:hover {
    color: #fff;
    background-color: #000;
}

.header_button button:active {
    color: #fff;
    background-color: #000;
}

.active {
    color: #fff;
    background-color: #5b5b5b;
}



.title {
    font-size: 18px;
    /* 左对齐 */
    text-align: left;
}
</style>