<script setup>
// VLA viewer (edge robotics inference): a purpose-built phase dashboard.
// Talks to the backend's /get_vla_avaliable and /get_vla_graph endpoints.
import { ref, reactive, onMounted, watch } from 'vue'
import { RouterLink } from 'vue-router'
import axios from 'axios'
import { strNumber, strNumberTime } from '@/utils.js'
import { Chart, registerables } from 'chart.js'
import annotationPlugin from 'chartjs-plugin-annotation'
Chart.register(...registerables, annotationPlugin)

const ipPort = ref('127.0.0.1:5000')
const vlaModels = ref([])      // [{id, action_head}]
const hardwares = ref([])
const result = ref(null)
const errorMsg = ref('')
const loading = ref(false)
const selectedPhase = ref(null)
let rooflineChart = null

const cfg = reactive({
  vla_model: 'tinyvla_demo',
  hardware: 'jetson_orin_nx_16gb',
  w_quant: '8-bit', a_quant: '8-bit', kv_quant: '8-bit',
  num_text_tokens: 256, num_images: 1, batch_size: 1, use_flashattention: false,
  control_hz: 10, exec_horizon: 10,   // exec_horizon 0 = execute the full chunk open-loop
  // architecture choices (override the preset's defaults)
  action_head: 'ar',                 // 'ar' | 'flow' | 'parallel'
  tokens_per_action: 7, action_horizon: 1,
  num_flow_steps: 10, action_chunk: 50, expert_attention: 'quadratic',
})

const QUANTS = ['FP16', '8-bit', '4-bit', '2-bit']
const PHASE_COLORS = {
  patch_embed: '#9b59b6', vision_encoder: '#3498db', projector: '#1abc9c',
  llm_prefill: '#e67e22', action_generation: '#e74c3c',
}

// A model's preset action head is just the default; the user can override it.
function presetHead(id) {
  const m = vlaModels.value.find((x) => x.id === id)
  return m ? m.action_head : 'ar'
}
// When the backbone changes, reset the action head to that model's default.
watch(() => cfg.vla_model, (id) => { cfg.action_head = presetHead(id) })

function gib(n) {
  return (n / (1024 ** 3)).toFixed(2) + ' GiB'
}

function fetchAvailable() {
  axios.get('http://' + ipPort.value + '/get_vla_avaliable')
    .then((r) => {
      vlaModels.value = r.data.vla_models
      hardwares.value = r.data.avaliable_hardwares
      cfg.action_head = presetHead(cfg.vla_model)  // initialize from preset default
      fetchGraph()
    })
    .catch((e) => { errorMsg.value = 'Cannot reach backend at ' + ipPort.value })
}

function fetchGraph() {
  loading.value = true
  errorMsg.value = ''
  const vla_config = {
    w_quant: cfg.w_quant, a_quant: cfg.a_quant, kv_quant: cfg.kv_quant,
    num_text_tokens: cfg.num_text_tokens, batch_size: cfg.batch_size,
    use_flashattention: cfg.use_flashattention, num_images: cfg.num_images,
    control_hz: cfg.control_hz, exec_horizon: cfg.exec_horizon,
    action_head: cfg.action_head,
    tokens_per_action: cfg.tokens_per_action, action_horizon: cfg.action_horizon,
    num_flow_steps: cfg.num_flow_steps, action_chunk: cfg.action_chunk,
    expert_attention: cfg.expert_attention,
  }
  axios.post('http://' + ipPort.value + '/get_vla_graph',
    { vla_model: cfg.vla_model, hardware: cfg.hardware, vla_config })
    .then((r) => {
      result.value = r.data
      loading.value = false
      // keep the drill-down pinned to the same phase across re-fetches
      if (selectedPhase.value) {
        const np = r.data.phases.find((p) => p.name === selectedPhase.value.name)
        selectedPhase.value = np || null
        if (np) setTimeout(drawRoofline, 0)
      }
    })
    .catch((e) => {
      loading.value = false
      const be = e.response && e.response.data && e.response.data.error
      errorMsg.value = be ? ('Backend: ' + be) : ('Request failed: ' + e.message)
    })
}

function selectPhase(p) {
  selectedPhase.value = p
  setTimeout(drawRoofline, 0)
}

function drawRoofline() {
  const el = document.getElementById('vlaRoofline')
  if (!el || !result.value || !selectedPhase.value) return
  if (rooflineChart) rooflineChart.destroy()
  const hw = result.value.hardware_info
  const { turning_point: tp, max_OPS: maxOPS } = hw
  const ai = selectedPhase.value.arithmetic_intensity
  const perf = selectedPhase.value.performance
  const xMax = Math.max(tp * 1.5, ai * 1.2)
  rooflineChart = new Chart(el, {
    type: 'line',
    data: {
      datasets: [
        { label: 'Roofline', data: [{ x: 0, y: 0 }, { x: tp, y: maxOPS }, { x: xMax, y: maxOPS }],
          borderColor: '#1f2d3d', borderWidth: 2, fill: false, pointRadius: 0 },
        { label: 'phase', data: [{ x: ai, y: perf }], borderColor: '#e74c3c',
          backgroundColor: '#e74c3c', pointRadius: 6, showLine: false },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { type: 'linear', min: 0, max: xMax, title: { display: true, text: 'Arithmetic intensity (OPs/byte)' } },
        y: { min: 0, max: maxOPS * 1.1, title: { display: true, text: 'Performance (OPS)' },
             ticks: { callback: (v) => v.toExponential(0) } },
      },
      plugins: {
        legend: { display: false },
        title: { display: true, text: 'Roofline — ' + selectedPhase.value.name + ' (' + selectedPhase.value.bound + '-bound)' },
        annotation: { annotations: { aiLine: { type: 'line', xMin: ai, xMax: ai, borderColor: '#e74c3c',
          borderWidth: 1, borderDash: [5, 5] } } },
      },
    },
  })
}

let debounce = null
watch(() => ({ ...cfg }), () => {
  clearTimeout(debounce)
  debounce = setTimeout(fetchGraph, 150)
}, { deep: true })

onMounted(fetchAvailable)
</script>

<template>
  <div class="vla_app">
    <!-- header -->
    <div class="vla_header">
      <span class="title">VLA Viewer <small>— edge robotics inference</small></span>
      <span class="spacer" />
      <label>Server <input v-model.lazy="ipPort" @change="fetchAvailable" class="server_in" /></label>
      <RouterLink to="/llm" class="nav_link">LLM viewer</RouterLink>
      <RouterLink to="/" class="nav_link">home</RouterLink>
    </div>

    <div class="vla_body">
      <!-- config -->
      <div class="vla_config">
        <h3>Model</h3>
        <label>VLA model
          <select v-model="cfg.vla_model">
            <option v-for="m in vlaModels" :key="m.id" :value="m.id">{{ m.id }}</option>
          </select>
        </label>
        <label>Hardware
          <select v-model="cfg.hardware">
            <option v-for="h in hardwares" :key="h" :value="h">{{ h }}</option>
          </select>
        </label>

        <h3>Architecture</h3>
        <label>Action decoding
          <select v-model="cfg.action_head">
            <option value="ar">Autoregressive</option>
            <option value="flow">Flow matching</option>
            <option value="parallel">Parallel (1-step)</option>
          </select>
        </label>
        <template v-if="cfg.action_head === 'ar'">
          <label>Action horizon <input type="number" min="1" v-model.number.lazy="cfg.action_horizon" /></label>
          <label>Tokens / action <input type="number" min="1" v-model.number.lazy="cfg.tokens_per_action" /></label>
        </template>
        <template v-else>
          <label v-if="cfg.action_head === 'flow'">Denoising steps <input type="number" min="1" v-model.number.lazy="cfg.num_flow_steps" /></label>
          <label>Action chunk <input type="number" min="1" v-model.number.lazy="cfg.action_chunk" /></label>
          <label>Expert attention
            <select v-model="cfg.expert_attention">
              <option value="quadratic">quadratic</option>
              <option value="linear">linear (SARA-RT)</option>
            </select>
          </label>
        </template>

        <h3>Inference</h3>
        <label>Camera images <input type="number" min="1" v-model.number.lazy="cfg.num_images"
          title="number of camera views; each adds tokens_per_image to the LLM prefix" /></label>
        <label>Text prompt tokens <input type="number" min="0" v-model.number.lazy="cfg.num_text_tokens" /></label>
        <label>Batch size <input type="number" min="1" v-model.number.lazy="cfg.batch_size" /></label>
        <label>Control freq (Hz) <input type="number" min="0" step="1" v-model.number.lazy="cfg.control_hz" /></label>
        <label>Exec horizon <input type="number" min="0" step="1" v-model.number.lazy="cfg.exec_horizon"
          title="actions executed open-loop before replanning; 0 = full chunk" /></label>
        <label class="cb">Flash attention <input type="checkbox" v-model="cfg.use_flashattention" /></label>

        <h3>Quantization</h3>
        <label>Weights <select v-model="cfg.w_quant"><option v-for="q in QUANTS" :key="q">{{ q }}</option></select></label>
        <label>Activations <select v-model="cfg.a_quant"><option v-for="q in QUANTS" :key="q">{{ q }}</option></select></label>
        <label>KV cache <select v-model="cfg.kv_quant"><option v-for="q in QUANTS" :key="q">{{ q }}</option></select></label>
      </div>

      <!-- dashboard -->
      <div class="vla_dash">
        <div v-if="errorMsg" class="err">{{ errorMsg }}</div>
        <template v-if="result">
          <div class="meta">{{ result.config }} on {{ result.hardware }} ·
            {{ result.num_images }}×{{ result.tokens_per_image }} = {{ result.num_image_tokens }} image tokens ·
            {{ result.prefix_len }}-token prefix · {{ result.action_head }} head</div>
          <div v-if="cfg.hardware.includes('thor')" class="caveat">⚠ Jetson Thor specs are preliminary (derived from the FP4 headline).</div>

          <!-- verdict cards -->
          <div class="cards">
            <div class="card">
              <div class="card_label">Step latency</div>
              <div class="card_big">{{ strNumberTime(result.total_time) }}s</div>
              <div class="card_sub">{{ result.steps_per_sec.toFixed(1) }} steps/s</div>
            </div>
            <div class="card" :class="result.memory.fits ? 'ok' : 'bad'">
              <div class="card_label">Memory fit</div>
              <div class="card_big">{{ result.memory.fits ? 'FITS' : 'OVER' }}</div>
              <div class="card_sub">{{ gib(result.memory.peak) }} / {{ gib(result.memory.capacity) }}</div>
            </div>
            <div v-if="result.control" class="card" :class="result.control.ok ? 'ok' : 'bad'">
              <div class="card_label">Control @ {{ result.control.hz }} Hz · exec {{ result.control.exec_horizon }}/{{ result.control.chunk_horizon }}</div>
              <div class="card_big">{{ result.control.ok ? 'WITHIN' : 'OVER' }}</div>
              <div class="card_sub">budget {{ strNumberTime(result.control.budget) }}s ({{ result.control.exec_horizon }}× period) ·
                sustains ~{{ result.control.achievable_hz.toFixed(1) }} Hz open-loop</div>
            </div>
          </div>

          <!-- latency share bar -->
          <h4>Latency by phase</h4>
          <div class="share_bar">
            <div v-for="p in result.phases" :key="p.name" class="seg"
                 :style="{ width: (p.share * 100) + '%', background: PHASE_COLORS[p.name] || '#888' }"
                 :title="p.name + ' ' + (p.share * 100).toFixed(1) + '%'"></div>
          </div>

          <!-- phase table (click a row to drill into its roofline) -->
          <table class="phases">
            <thead><tr><th>phase</th><th>latency</th><th>share</th><th>bound</th><th>OPs</th><th>weights</th></tr></thead>
            <tbody>
              <tr v-for="p in result.phases" :key="p.name" class="clickable"
                  :class="{ selrow: selectedPhase && selectedPhase.name === p.name }" @click="selectPhase(p)">
                <td><span class="dot" :style="{ background: PHASE_COLORS[p.name] || '#888' }"></span>{{ p.name }}<span v-if="p.mode" class="tag">{{ p.mode }}</span></td>
                <td>{{ strNumberTime(p.time) }}s</td>
                <td>{{ (p.share * 100).toFixed(1) }}%</td>
                <td><span :class="'bd_' + p.bound">{{ p.bound }}</span></td>
                <td>{{ strNumber(p.OPs) }}</td>
                <td>{{ p.weight ? gib(p.weight) : '—' }}</td>
              </tr>
            </tbody>
          </table>
          <p class="hint">Click a phase for its roofline. Vision is a single-encoder approximation —
            dual-encoder VLAs (e.g. OpenVLA's SigLIP+DINOv2) roughly double the vision phase.</p>

          <div v-if="selectedPhase" class="roofline_panel">
            <div class="roofline_canvas"><canvas id="vlaRoofline"></canvas></div>
            <div class="roofline_meta">
              <div><strong>{{ selectedPhase.name }}</strong></div>
              <div>arithmetic intensity: <b>{{ selectedPhase.arithmetic_intensity.toFixed(1) }}</b> OPs/byte</div>
              <div>bound: <span :class="'bd_' + selectedPhase.bound">{{ selectedPhase.bound }}</span>
                (turning point {{ result.hardware_info.turning_point.toFixed(0) }})</div>
              <div>effective perf: {{ selectedPhase.performance.toExponential(2) }} OPS</div>
              <div>memory access: {{ strNumber(selectedPhase.memory_access) }}B</div>
            </div>
          </div>

          <!-- memory breakdown -->
          <h4>Memory footprint</h4>
          <table class="phases">
            <tbody>
              <tr><td>weights</td><td>{{ gib(result.memory.weights) }}</td></tr>
              <tr><td>kv cache</td><td>{{ gib(result.memory.kv_cache) }}</td></tr>
              <tr><td>activations (peak)</td><td>{{ gib(result.memory.act_peak) }}</td></tr>
              <tr class="total"><td>peak total</td><td>{{ gib(result.memory.peak) }} / {{ gib(result.memory.capacity) }}</td></tr>
            </tbody>
          </table>
          <p class="note">Roofline estimate: theoretical hardware ceiling; use for relative comparison, not absolute timings.</p>
        </template>
        <div v-else-if="!errorMsg" class="loading">Loading…</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.vla_app { font-family: sans-serif; height: 100vh; display: flex; flex-direction: column; }
.vla_header { height: 46px; display: flex; align-items: center; gap: 14px; padding: 0 16px;
  background: #1f2d3d; color: #fff; }
.vla_header .title { font-size: 18px; font-weight: 600; }
.vla_header small { opacity: .7; font-weight: 400; }
.spacer { flex: 1; }
.server_in { width: 130px; margin-left: 4px; }
.nav_link { color: #8ec6ff; text-decoration: none; }
.vla_body { display: flex; flex: 1; min-height: 0; }
.vla_config { width: 280px; background: #f4f6f8; border-right: 2px solid #e2e2e2;
  padding: 10px 14px; overflow-y: auto; }
.vla_config h3 { margin: 14px 0 6px; font-size: 13px; text-transform: uppercase; color: #5a6b7b; }
.vla_config label { display: flex; justify-content: space-between; align-items: center;
  gap: 8px; margin: 5px 0; font-size: 13px; }
.vla_config input[type=number], .vla_config select, .vla_config input.server_in { width: 120px; }
.vla_config label.cb { justify-content: flex-start; }
.vla_dash { flex: 1; padding: 16px 22px; overflow-y: auto; }
.meta { color: #5a6b7b; font-size: 13px; margin-bottom: 12px; }
.cards { display: flex; gap: 14px; flex-wrap: wrap; }
.card { flex: 1; min-width: 150px; background: #fff; border: 1px solid #e2e2e2;
  border-left: 4px solid #1f2d3d; border-radius: 4px; padding: 10px 14px; }
.card.ok { border-left-color: #27ae60; }
.card.bad { border-left-color: #c0392b; }
.card_label { font-size: 12px; color: #7a8a99; }
.card_big { font-size: 26px; font-weight: 700; margin: 2px 0; }
.card.ok .card_big { color: #27ae60; }
.card.bad .card_big { color: #c0392b; }
.card_sub { font-size: 12px; color: #7a8a99; }
h4 { margin: 20px 0 8px; }
.share_bar { display: flex; height: 26px; border-radius: 4px; overflow: hidden; border: 1px solid #e2e2e2; }
.seg { height: 100%; }
table.phases { width: 100%; border-collapse: collapse; font-size: 13px; }
table.phases th, table.phases td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #eee; }
table.phases th { color: #7a8a99; font-weight: 600; }
table.phases tr.total td { font-weight: 700; border-top: 2px solid #ddd; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }
.tag { margin-left: 6px; font-size: 11px; color: #fff; background: #888; padding: 1px 5px; border-radius: 3px; }
.err { color: #c0392b; padding: 10px; background: #fdecea; border-radius: 4px; }
.loading { color: #7a8a99; }
.note { color: #9aa7b3; font-size: 12px; margin-top: 16px; }
.caveat { color: #b9770e; font-size: 12px; margin-bottom: 8px; }
.hint { color: #9aa7b3; font-size: 12px; margin: 8px 0; }
table.phases tr.clickable { cursor: pointer; }
table.phases tr.clickable:hover { background: #f5f9ff; }
table.phases tr.selrow { background: #eef6ff; }
.bd_memory { color: #c0392b; font-weight: 600; }
.bd_compute { color: #27ae60; font-weight: 600; }
.roofline_panel { display: flex; gap: 18px; margin-top: 14px; align-items: center; }
.roofline_canvas { width: 380px; height: 260px; }
.roofline_meta { font-size: 13px; color: #444; display: flex; flex-direction: column; gap: 5px; }
</style>
