<template>
    <div class="title">LLMViewer v0.1</div>
    <div class="header_button">
        | Model:
        <select v-model="selectedOption" @change="handleSelection" >
            <option value="meta-llama/Llama-2-7b-hf">meta-llama/Llama-2-7b-hf</option>
            <option value="meta-llama/Llama-2-13b-hf">meta-llama/Llama-2-13b-hf</option>
            <option value="meta-llama/Llama-2-70b-hf">meta-llama/Llama-2-70b-hf</option>
            <!-- <option value="ChatGLM">ChatGLM</option> -->
        </select>
    </div>
</template>

<script setup>
import { inject, ref, watch, computed } from 'vue';
const model_id = inject('model_id');
const graphUpdateTrigger = inject('graphUpdateTrigger');

var selectedOption = ref('meta-llama/Llama-2-7b-hf');

function handleSelection() {
    console.log("handleSelection", selectedOption.value)
    model_id.value = selectedOption.value
    graphUpdateTrigger.value += 1

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