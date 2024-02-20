<script setup>
import Graph from "./components/Graph.vue"
import LeftPannel from "./components/LeftPannel.vue"
import Header from "./components/Header.vue"
import { ref, computed, provide } from 'vue';

const model_id = ref("meta-llama/Llama-2-7b-hf");
const settingsData = ref({ ip_port: '127.0.0.1:9600'});
const selectedNodeInfo = ref({});
const selectedNodeId = ref("");
const serverStatus = ref("idel");
const graphUpdateTrigger = ref(1);
const selectedNodeListRef = ref(new Array())

provide("settingsData", settingsData);
provide("selectedNodeInfo", selectedNodeInfo);
provide("selectedNodeId", selectedNodeId);
provide("serverStatus", serverStatus);
provide("model_id", model_id);
provide("graphUpdateTrigger", graphUpdateTrigger);
provide("selectedNodeListRef", selectedNodeListRef);


const InferenceConfig = ref({"stage": "decode", batch_size:1 ,seq_length:2, w_quant:"FP16", a_quant:"FP16", kv_quant:"FP16"});
provide("InferenceConfig", InferenceConfig);

</script>

<template>
  <div class="app_container">
    <div class="upper_header">
      <Header></Header>
    </div>
    <div class="bottom-block">
      <LeftPannel></LeftPannel>
      <Graph></Graph>
    </div>

  </div>
</template>

<style>
.app_container {
  /* display: flex;
  flex-direction: column;
  width: 98vw; */
  width: 100%;
  height: 100vh;

}

.upper_header {
  flex: 1;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 50px;
  background-color: #f0f0f0;
  /* border-right: 1px solid #e2e2e2; */
  border-bottom: 3px solid #e2e2e2;
}

.bottom-block {
  display: flex;
  flex-direction: row;
  height: calc(100% - 60px);
}
</style>
