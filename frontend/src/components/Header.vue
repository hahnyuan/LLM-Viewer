<template>
    <div class="title">LLMViewer v0.1</div>
    <div class="header_button">
        <!-- <button @click="change_step('settings')" :class="{ active: step == 'settings' }">Settings</button> -->
        | Model:
        <select v-model="selectedOption" @change="handleSelection">
            <option value="LLaMA">LLaMA</option>
            <option value="ChatGLM">ChatGLM</option>
        </select>

        <!-- | Stage:
        <button @click="change_step('onnx_modifier')" :class="{ active: step == 'onnx_modifier' }">Prefill</button>
        <button @click="change_step('hm_onnx_viewer')" :class="{ active: step == 'hm_onnx_viewer' }">Decode</button> -->
    </div>
</template>

<script setup>
import { inject, ref, watch, computed } from 'vue';
const step = inject('step');
const settingsData = inject('settingsData');

// function change_step(new_step) {
//     console.log("change_step", new_step)
//     step.value = new_step
// }
var selectedOption = ref('LLaMA');

function handleSelection() {
    console.log("handleSelection", selectedOption.value)
    step.value = selectedOption.value

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