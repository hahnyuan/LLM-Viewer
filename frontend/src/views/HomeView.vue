<script setup>
// Landing page: choose which viewer to open.
import { RouterLink } from 'vue-router'
</script>

<template>
  <div class="home">
    <h1>LLM / VLA Viewer</h1>
    <p class="sub">Roofline analysis of transformer inference.</p>
    <div class="choices">
      <RouterLink to="/llm" class="choice">
        <div class="choice_title">LLM Viewer</div>
        <div class="choice_desc">Language models on datacenter / cloud GPUs — throughput, batching, tensor parallelism.</div>
      </RouterLink>
      <RouterLink to="/vla" class="choice">
        <div class="choice_title">VLA Viewer</div>
        <div class="choice_desc">Vision-Language-Action models on robotics edge devices — per-phase latency, control-budget &amp; memory-fit.</div>
      </RouterLink>
    </div>
  </div>
</template>

<style scoped>
.home { min-height: 100vh; display: flex; flex-direction: column; align-items: center;
  justify-content: center; font-family: sans-serif; background: #f4f6f8; padding: 20px; }
h1 { margin: 0 0 4px; color: #1f2d3d; }
.sub { color: #7a8a99; margin: 0 0 32px; }
.choices { display: flex; gap: 24px; flex-wrap: wrap; justify-content: center; }
.choice { width: 300px; background: #fff; border: 1px solid #e2e2e2; border-top: 4px solid #1f2d3d;
  border-radius: 6px; padding: 22px; text-decoration: none; color: inherit;
  transition: box-shadow .15s, transform .15s; }
.choice:hover { box-shadow: 0 4px 16px rgba(0,0,0,.12); transform: translateY(-2px); }
.choice_title { font-size: 20px; font-weight: 700; color: #1f2d3d; margin-bottom: 8px; }
.choice_desc { font-size: 13px; color: #5a6b7b; line-height: 1.5; }
</style>
