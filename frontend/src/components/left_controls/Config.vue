<template>
    <h2>Network Config</h2>
    <div v-for="(param_info) in frontend_params_info">
        {{ param_info.name }}:
        <span class="config_div" v-if="param_info.type === 'select'">
            <select v-model="param_info.value">
                <option v-for="option in param_info.choices" :value="option">{{ option }}</option>
            </select>
        </span>
        <span class="config_div" v-else-if="param_info.type === 'int'">
            <input type="range" :min="param_info.min" :max="param_info.max" v-model.lazy="param_info.value">
            <input type="number" v-model.lazy="param_info.value" style="width: 50px;" :min="param_info.min"
                :max="param_info.max">
        </span>
        <span class="config_div" v-else-if="param_info.type === 'bool'">
            <input type="checkbox" v-model="param_info.value">
        </span>
        <span class="config_div" v-else-if="param_info.type === 'float'">
            <input type="range" :min="param_info.min" :max="param_info.max" v-model.lazy="param_info.value" step="0.01">
            <input type="number" v-model.lazy="param_info.value" style="width: 50px;" :min="param_info.min"
                :max="param_info.max" step="0.01">
        </span>
        <span class="config_div" v-else-if="param_info.type === 'str'">
            <input type="text" v-model.lazy="param_info.value">
        </span>

    </div>
    <h2>Network-wise Analysis</h2>
    <div>
        <div v-for="(value, key) in network_results" :key="key" class="network-wise-info-item">
            <span v-if="['bound'].includes(key)">{{ key }}: {{ value }}</span>
            <span v-else-if="['inference_time'].includes(key)">{{ key }}: {{ strNumberTime(value) }}</span>
            <span
                v-else-if="['memory_access', 'load_act', 'load_kv_cache', 'load_weight', 'store_act', 'store_kv_cache'].includes(key)">{{
        key }}: {{ strNumber_1024(value) }}B</span>
            <span v-else>{{ key }}: {{ strNumber(value) }}</span>
        </div>
        <p>NOTE: The time estimated by the roofline model represents the theoretical performance that the hardware can
            achieve.
            The purpose of creating this tool is to help readers gain a clearer understanding of the key factors that
            influence LLM inference.
            Only the relative relationships can be referenced. </p>
    </div>
</template>

<script setup>
import { inject, ref, watch, computed } from 'vue';
import { strNumber, strNumberTime, strNumber_1024 } from '@/utils.js';

const global_update_trigger = inject('global_update_trigger');
const frontend_params_info = inject('frontend_params_info');
const network_results = inject('network_results');

watch(frontend_params_info, () => {
    console.log("frontend_params_info change")
    global_update_trigger.value += 1
}, { deep: true });

</script>

<style>
.config_div {
    border-top: 1px solid #e2e2e2;
}

.hover_color {
    color: #0000ff;
    cursor: pointer;
}

.network-wise-info-item {
    padding: 3px;
    border-top: 1px solid #e2e2e2;
}
</style>