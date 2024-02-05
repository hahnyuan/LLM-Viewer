<template>
    <h2>Adjust Pannel</h2>
    <!-- 两个滑块，一个是batchsize，一个是sequence length -->
    <div class="slider">
        Batchsize: 
        <input type="range" min="1" max="256" value="1" oninput="batch_size.innerText = this.value">
        <p id="batch_size">1</p>
    </div>
    <div class="slider">
        Sequence Length:
        <input type="range" min="1" max="4096" value="1024" oninput="seq_length.innerText = this.value">
        <p id="seq_length">1024</p>
    </div>
    
</template>

<script setup>
import { inject, ref, watch, computed } from 'vue';

const ip_port = inject('settingsData').value.ip_port
const serverStatus = inject('serverStatus');
const selectedNodeInfo = inject('selectedNodeInfo');

</script>

<style>
.input_config {
    /* 外边有一圈框框 */
    border: 1px solid #ccc;
    margin: 3px;
}

.hover_color {
    /* 当鼠标悬停时，改变文字颜色 */
    color: #0000ff;
    cursor: pointer;

}
</style>