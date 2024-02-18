<script setup>
import Graph from "./components/Graph.vue"
import LeftPannel from "./components/LeftPannel.vue"
import Header from "./components/Header.vue"
import { ref, computed, provide } from 'vue';

const step = ref("LLaMA");
const settingsData = ref({ ip_port: '127.0.0.1:9600' });
const selectedNodeInfo = ref({});
const selectedNodeId = ref("");
const serverStatus = ref("idel");
const graphUpdateTrigger = ref(1);
const selectedNodeListRef = ref(new Array())
provide("settingsData", settingsData);
provide("selectedNodeInfo", selectedNodeInfo);
provide("selectedNodeId", selectedNodeId);
provide("serverStatus", serverStatus);
provide("step", step);
provide("graphUpdateTrigger", graphUpdateTrigger);
provide("selectedNodeListRef", selectedNodeListRef);

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
  /* 设置容器的宽度为100% */
  height: 100vh;
  /* 设置容器的高度为100% */

}

.upper_header {
  flex: 1;
  /* 设置顶部区块占据剩余的可用空间 */
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
  /* 使用flex布局 */
  /* flex-wrap: wrap; */
  height: calc(100% - 60px);
  /* 允许子元素换行 */
  /* display: flex;
  flex: 1;
  height: 90vh;
  设置底部区块占据剩余的可用空间 */
}
</style>
