<template>
    <div class="left_control">
        <Settings v-if="step == 'settings'" class="contrl_content"></Settings>
        <Parse v-if="step == 'LLaMA'" class="contrl_content"></Parse>
    </div>
</template>

<script setup>
import Settings from "./left_controls/Settings.vue"
import Parse from "./left_controls/Parse.vue"
// import QuantConfig from "./left_controls/QuantConfig.vue"
// import ExecQuant from "./left_controls/ExecQuant.vue"
// import OnnxModifier from "./left_controls/OnnxModifier.vue"
// import QuantDataset from "./left_controls/QuantDataset.vue"
import { inject, ref, watch, computed, onMounted } from 'vue';

const step = inject('step');

onMounted(() => {
    console.log("LeftControl onMounted", step.value)
})

</script>

<style>
.input_config {
    /* 外边有一圈框框 */
    border: 1px solid #ccc;
    margin: 3px;
}

.left_control {
    /* flex-direction: column; */
    align-items: left;
    /* justify-content: center; */
    /* width 最大14% 最小100px */
    width: 20%;
    height: 100%;
    /* 设置左侧面板和图形的高度为100% */

    /* float: left; */

    /* height: 100%; */
    /* 内容顶端对齐 */

    /* height: 100%; */
    background-color: #f0f0f0;
    /* border-right: 1px solid #e2e2e2; */
    border: 3px solid #e2e2e2;
    overflow-y: auto;
    /* 底部margin */
    margin-bottom: 10px;
}
</style>