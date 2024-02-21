<template>
    <div class="title">LLMViewer v0.1</div>
    <div class="header_button">
        | 
        <span>Model: </span>
        <select v-model="select_model_id">
            <option value="meta-llama/Llama-2-7b-hf">meta-llama/Llama-2-7b-hf</option>
            <option value="meta-llama/Llama-2-13b-hf">meta-llama/Llama-2-13b-hf</option>
            <option value="meta-llama/Llama-2-70b-hf">meta-llama/Llama-2-70b-hf</option>
            <!-- <option value="ChatGLM">ChatGLM</option> -->
        </select>
        <span> | </span>
        <span>Hardware: </span>
        <select v-model="select_hardware">
            <option value="nvidia_V100">nvidia_V100</option>
            <option value="nvidia_A100">nvidia_A100</option>
            <option value="nvidia_H100">nvidia_H100</option>
            <!-- <option value="ChatGLM">ChatGLM</option> -->
        </select>
    </div>
</template>

<script setup>
import { inject, ref, watch, computed } from 'vue';
const model_id = inject('model_id');
const hardware = inject('hardware');
const graphUpdateTrigger = inject('graphUpdateTrigger');

var select_model_id = ref('meta-llama/Llama-2-7b-hf');
watch(select_model_id, (n) => {
    console.log("select_model_id", n)
    model_id.value = n
    graphUpdateTrigger.value += 1
})

var select_hardware = ref('nvidia_V100');
watch(select_hardware, (n) => {
    console.log("select_hardware", n)
    hardware.value = n
    graphUpdateTrigger.value += 1
})


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