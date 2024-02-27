<template>
    <div class="title">
        <!-- <img :src="publicPath + '/favicon.ico'" alt="Header Image"> -->
        LLM-Viewer v{{ version }}</div>
    <div class="header_button">
        | 
        <span>Model: </span>
        <select v-model="select_model_id">
            <!-- <option value="meta-llama/Llama-2-7b-hf">meta-llama/Llama-2-7b-hf</option>
            <option value="meta-llama/Llama-2-13b-hf">meta-llama/Llama-2-13b-hf</option>
            <option value="meta-llama/Llama-2-70b-hf">meta-llama/Llama-2-70b-hf</option> -->
            <!-- <option value="ChatGLM">ChatGLM</option> -->
            <option v-for="model_id in avaliable_model_ids" :value="model_id">{{model_id}}</option>
        </select>
        <span> | </span>
        <span>Hardware: </span>
        <select v-model="select_hardware">
            <!-- <option value="nvidia_V100">nvidia_V100</option>
            <option value="nvidia_A100">nvidia_A100</option>
            <option value="nvidia_H100">nvidia_H100</option> -->
            <!-- <option value="ChatGLM">ChatGLM</option> -->
            <option v-for="hardware in avaliable_hardwares" :value="hardware">{{hardware}}</option>
        </select>
    </div>
    <div>
        <span> | </span>
        <span>Server: </span>
        <select v-model="ip_port"  >
            <option value="api.llm-viewer.com:5000">api.llm-viewer.com</option>
            <option value="localhost:5000">localhost</option>
        </select>
    </div>
    <div>
        <span> | </span>
        <a href="https://github.com/hahnyuan/LLM-Viewer" target="_blank" class="hover-bold">Document</a>
        <!-- <a href="https://github.com/hahnyuan/LLM-Viewer" target="_blank"> Document </a> -->
        <!-- Document -->
    </div>
</template>

<script setup>
import { inject, ref, watch, computed, onMounted } from 'vue';
import axios from 'axios'
const model_id = inject('model_id');
const hardware = inject('hardware');
const graphUpdateTrigger = inject('graphUpdateTrigger');
const ip_port = inject('ip_port');

const avaliable_hardwares = ref([]);
const avaliable_model_ids=ref([]);

const version=ref(llm_viewer_frontend_version)

function update_avaliable(){
    const url = 'http://' + ip_port.value + '/get_avaliable'
    axios.get(url).then(function (response) {
        console.log(response);
        avaliable_hardwares.value = response.data.avaliable_hardwares
        avaliable_model_ids.value = response.data.avaliable_model_ids
    })
        .catch(function (error) {
            console.log("error in get_avaliable");
            console.log(error);
        });
}

onMounted(() => {
    console.log("Header mounted")
    update_avaliable()
})

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

watch(ip_port, (n) => {
    console.log("ip_port", n)
    update_avaliable()
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

.hover-bold:hover {
  font-weight: bold;
}

</style>