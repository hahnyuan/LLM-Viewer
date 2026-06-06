import { createRouter, createWebHashHistory } from 'vue-router'
import HomeView from './views/HomeView.vue'
import LlmView from './views/LlmView.vue'
import VlaView from './views/VlaView.vue'

// One app shell, a landing page that links to the two separate products:
//   /      -> landing page (choose a viewer)
//   /llm   -> LLM viewer (cloud LLM serving) -- behavior unchanged
//   /vla   -> VLA viewer (edge robotics inference)
// Hash history keeps deep links working on plain static hosting.
const routes = [
  { path: '/', name: 'home', component: HomeView },
  { path: '/llm', name: 'llm', component: LlmView },
  { path: '/vla', name: 'vla', component: VlaView },
]

export const router = createRouter({
  history: createWebHashHistory(),
  routes,
})
