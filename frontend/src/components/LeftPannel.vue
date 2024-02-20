<template>
    <div class="left_control">
        <Config></Config>
    </div>
</template>

<script setup>
import Config from "./left_controls/Config.vue"
import { inject, ref, watch, computed, onMounted } from 'vue';

const model_id = inject('model_id');

onMounted(() => {
    console.log("LeftControl onMounted", model_id.value)
})

</script>

<style>
.input_config {
    border: 1px solid #ccc;
    margin: 3px;
}

.left_control {
    align-items: left;
    width: 20%;
    height: 100%;
    background-color: #f0f0f0;
    border: 3px solid #e2e2e2;
    overflow-y: auto;
    margin-bottom: 10px;
}
</style>