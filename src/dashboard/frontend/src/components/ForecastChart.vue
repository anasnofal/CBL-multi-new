<template>
  <div class="chart-panel" :class="{ expanded: lsoaCode }">
    <div class="chart-title" v-if="lsoaCode">
      <span class="model-badge" :class="currentModel === 'XGBoost' ? 'xgb' : 'sar'" v-if="currentModel">
        {{ currentModel }}
      </span>
      {{ crimeType }} &mdash; {{ lsoaCode }}
    </div>

    <div class="panel-tabs" v-if="lsoaCode">
      <button
        class="panel-tab"
        :class="{ active: activeTab === 'forecast' }"
        @click="activeTab = 'forecast'"
      >
        Forecast
      </button>
      <button
        class="panel-tab"
        :class="{ active: activeTab === 'history' }"
        @click="activeTab = 'history'"
      >
        History
      </button>
    </div>

    <div class="chart-placeholder" v-if="activeTab === 'forecast' && !lsoaCode">
      <span>Select an LSOA on the map to see the forecast</span>
    </div>

    <div class="chart-placeholder" v-else-if="activeTab === 'forecast' && isLoading">
      <span>Loading forecast...</span>
    </div>

    <div class="chart-placeholder error" v-else-if="activeTab === 'forecast' && error">
      <span>{{ error }}</span>
    </div>

    <div class="all-alert-panel" v-else-if="activeTab === 'forecast' && isAllCrimeTypes && lsoaCode">
      <div class="all-alert-title">Crime types with an active alert</div>
      <div class="all-alert-subtitle">
        Alerts for {{ startMonth }} to {{ endMonth }}
      </div>

      <div v-if="selectedAlertRows.length === 0" class="all-alert-empty">
        No alerts for this LSOA in the selected period
      </div>

      <div v-else class="all-alert-list">
        <div
          v-for="alert in selectedAlertRows"
          :key="`${alert.crime_type}-${alert.month}`"
          class="all-alert-item"
        >
          <div>
            <div class="all-alert-crime">{{ alert.crime_type }}</div>
            <div class="all-alert-meta">{{ alert.month }} · predicted {{ alert.predicted_count }} crimes</div>
          </div>
        </div>
      </div>
    </div>

    <div class="chart-placeholder" v-else-if="activeTab === 'forecast' && predictions.length === 0">
      <span>No forecast data for this selection</span>
    </div>

    <canvas
      ref="chartCanvas"
      class="forecast-canvas"
      v-show="activeTab === 'forecast' && !isAllCrimeTypes && !isLoading && !error && predictions.length > 0"
    ></canvas>

    <p v-if="activeTab === 'forecast' && !isAllCrimeTypes && predictions.length > 0 && !explainData && !isExplainLoading" class="click-hint">
      Click a point to see why
    </p>
    <p v-if="activeTab === 'forecast' && isExplainLoading" class="click-hint">Loading explanation…</p>

    <!-- Waterfall explain panel -->
    <div v-if="activeTab === 'forecast' && !isAllCrimeTypes && explainData" class="explain-panel">
      <div class="explain-header">
        <span class="model-badge" :class="explainData.model === 'XGBoost' ? 'xgb' : 'sar'">
          {{ explainData.model }}
        </span>
        <span class="explain-month">{{ explainData.month }}</span>
        <button class="explain-close" @click="explainData = null">✕</button>
      </div>
      <p class="explain-summary">{{ explainData.summary }}</p>
      <canvas ref="waterfallCanvas" class="waterfall-canvas"></canvas>
    </div>

    <!-- Savings panel -->
    <div v-if="activeTab === 'forecast' && !isAllCrimeTypes && lsoaCode && predictions.length > 0" class="savings-panel">
      <div class="savings-title">Savings for alerted month</div>

      <div v-if="!alertMonth" class="savings-muted">
        No alert for this LSOA in the selected period
      </div>

      <div v-else-if="isSavingsLoading" class="savings-muted">Loading savings...</div>

      <div v-else-if="savingsError" class="savings-error">{{ savingsError }}</div>

      <template v-else-if="savings">
        <div class="savings-context">
          Alerted month: <strong>{{ alertMonth }}</strong>
          Predicted crimes: <strong>{{ alertPredictedCount }}</strong>
          Cost per crime incident: <strong>£{{savings.cost}}</strong>
        </div>
        <div class="recommendation-panel">
          <div class="recommendation-title">Recommended Intervention</div>
          <div class="recommendation-grid">
            <div>
              <div class="recommendation-name">
                {{ savings.recommendation_l1 }}<sup class="footnote-marker">1</sup>
              </div>
              <div class="recommendation-description">
                {{ savings.description }}
              </div>
            </div>
            <div class="recommendation-metric-box">
              <div class="metric-value">
                ~{{ (savings.effectiveness * 100).toFixed(0) }}%<sup class="footnote-marker">2</sup>
              </div>
              <div class="metric-label">Effectiveness</div>
            </div>
          </div>
          <hr class="recommendation-divider">
          <div>
            <div class="source-text">
              <sup>1</sup> Source: {{ savings.source }}
            </div>
            <div class="source-text">
              <sup>2</sup> Effectiveness is an approximation based on aggregated evidence and may vary by context.
            </div>
          </div>
        </div>
        <div class="savings-grid">
          <div>
            <span>Estimated savings</span>
            <strong>£{{ savings.estimated_savings.toLocaleString() }}</strong>
          </div>
          <div>
            <span>Lower bound</span>
            <strong>£{{ savings.lower_bound.toLocaleString() }}</strong>
          </div>
          <div>
            <span>Upper bound</span>
            <strong>£{{ savings.upper_bound.toLocaleString() }}</strong>
          </div>
        </div>
      </template>
    </div>

    <HistoricalCrimeChart
      v-if="activeTab === 'history'"
      :police-force="policeForce"
      :lsoa-code="lsoaCode"
    />
  </div>
</template>

<script setup>
import { ref, watch, nextTick, computed } from 'vue'
import { Chart, registerables } from 'chart.js'
import HistoricalCrimeChart from './HistoricalCrimeChart.vue'

Chart.register(...registerables)

const props = defineProps({
  policeForce: { type: String, required: true },
  lsoaCode:    { type: String, default: null },
  crimeType:   { type: String, required: true },
  startMonth:  { type: String, default: null },
  endMonth:    { type: String, default: null },
  alertLevels: { type: Object, default: () => ({}) },
})

const chartCanvas     = ref(null)
const waterfallCanvas = ref(null)
const isLoading       = ref(false)
const error           = ref(null)
const predictions     = ref([])
const savings         = ref(null)
const savingsError    = ref(null)
const isSavingsLoading = ref(false)
const explainData     = ref(null)
const isExplainLoading = ref(false)
const selectedIdx     = ref(null)
const activeTab       = ref('forecast')

const currentModel = computed(() => predictions.value[0]?.model ?? null)
const isAllCrimeTypes = computed(() => props.crimeType === 'All')

const alertInfo = computed(() => {
  if (!props.lsoaCode) return null
  const entry = props.alertLevels[props.lsoaCode]
  if (!entry) return null
  return typeof entry === 'number' ? { level: entry, dates: [] } : entry
})

const selectedAlertRows = computed(() => alertInfo.value?.alerts ?? [])

const alertMonth = computed(() => alertInfo.value?.dates?.[0] ?? null)

function getMonthKey(value) {
  return String(value).slice(0, 7)
}

const alertPrediction = computed(() =>
  predictions.value.find(p => getMonthKey(p.Month) === alertMonth.value) ?? null
)

const alertPredictedCount = computed(() =>
  alertPrediction.value ? Math.round(Math.max(0, Number(alertPrediction.value.predicted))) : null
)

let chartInstance    = null
let waterfallInstance = null

function buildUrl() {
  const base = `/api/v1/forecast/${props.policeForce}/${encodeURIComponent(props.lsoaCode)}/${encodeURIComponent(props.crimeType)}`
  const params = new URLSearchParams()
  if (props.startMonth) params.set('time_start', props.startMonth)
  if (props.endMonth)   params.set('time_end',   props.endMonth)
  return `${base}?${params}`
}

function buildExplainUrl(month) {
  return `/api/v1/explain/${props.policeForce}/${encodeURIComponent(props.lsoaCode)}/${encodeURIComponent(props.crimeType)}/${month}`
}

function drawChart() {
  if (activeTab.value !== 'forecast') return
  if (!chartCanvas.value) return

  const labels  = predictions.value.map(p => {
    const [year, month] = getMonthKey(p.Month).split('-')
    return new Date(Number(year), Number(month) - 1).toLocaleDateString('en-GB', { month: 'short', year: 'numeric' })
  })
  const values   = predictions.value.map(p => Math.max(0, p.predicted))
  const rawLower = predictions.value.map(p => p.ci_lower != null ? Math.max(0, p.ci_lower) : null)
  const rawUpper = predictions.value.map(p => p.ci_upper != null ? p.ci_upper : null)

  const hasCi = rawLower.every(v => v != null) && rawUpper.every(v => v != null)

  const ciLower = hasCi ? rawLower.map((lo, i) => Math.min(lo, rawUpper[i])) : []
  const ciUpper = hasCi ? rawUpper.map((hi, i) => Math.max(hi, rawLower[i])) : []

  if (chartInstance) { chartInstance.destroy(); chartInstance = null }

  const datasets = []

  if (hasCi) {
    datasets.push({
      label: 'CI Lower',
      data: ciLower,
      borderWidth: 0,
      pointRadius: 0,
      fill: '+1',
      backgroundColor: 'rgba(245,158,11,0.12)',
      borderColor: 'transparent',
      tension: 0.3,
    })
    datasets.push({
      label: 'CI Upper',
      data: ciUpper,
      borderWidth: 0,
      pointRadius: 0,
      fill: false,
      borderColor: 'transparent',
      tension: 0.3,
    })
  }

  datasets.push({
    label: 'Predicted',
    data: values,
    borderColor: '#f59e0b',
    backgroundColor: 'transparent',
    tension: 0.3,
    pointRadius: predictions.value.map((_, i) => i === selectedIdx.value ? 8 : 5),
    pointBackgroundColor: predictions.value.map((_, i) => i === selectedIdx.value ? '#fff' : '#f59e0b'),
    pointHoverRadius: 8,
    pointHitRadius: 20,
    fill: false,
  })

  chartInstance = new Chart(chartCanvas.value, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: {
        padding: {
          left: 10,
          right: 8,
        },
      },
      onHover: (evt) => {
        if (evt.native?.target) {
          const points = chartInstance?.getElementsAtEventForMode(evt, 'index', { intersect: false }, false) ?? []
          evt.native.target.style.cursor = points.length ? 'pointer' : 'default'
        }
      },
      onClick: (evt) => {
        if (!chartInstance) return
        const points = chartInstance.getElementsAtEventForMode(evt, 'index', { intersect: false }, false)
        const hit = points.find(e => chartInstance.data.datasets[e.datasetIndex].label === 'Predicted')
        if (!hit) return
        const idx = hit.index
        const month = getMonthKey(predictions.value[idx].Month)
        selectedIdx.value = idx
        fetchExplain(month, predictions.value[idx].predicted)
        const ds = chartInstance.data.datasets.find(d => d.label === 'Predicted')
        ds.pointRadius = predictions.value.map((_, i) => i === idx ? 8 : 5)
        ds.pointBackgroundColor = predictions.value.map((_, i) => i === idx ? '#fff' : '#f59e0b')
        chartInstance.update()
      },
      scales: {
        x: {
          ticks: { color: '#aaa', maxRotation: 45 },
          grid:  { color: '#2a2a4a' },
        },
        y: {
          beginAtZero: true,
          max: hasCi
            ? Math.ceil(Math.max(...ciUpper) * 1.1)
            : Math.ceil(Math.max(...values) * 1.5),
          ticks: { color: '#aaa', padding: 8 },
          grid:  { color: '#2a2a4a' },
          title: {
            display: true,
            text: 'Predicted crimes',
            color: '#cbd5e1',
            padding: 8,
            font: {
              size: 12,
            },
          },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          filter: item => item.dataset.label === 'Predicted',
          callbacks: {
            label: ctx => {
              const p = predictions.value[ctx.dataIndex]
              const ci = (p.ci_lower != null && p.ci_upper != null)
                ? `  CI [${Math.round(p.ci_lower)} – ${Math.round(p.ci_upper)}]`
                : ''
              return `${ctx.parsed.y.toFixed(1)} crimes${ci}`
            },
          },
        },
      },
    },
  })
}

function buildWaterfallData(explain) {
  const items = []
  const baseline = explain.baseline ?? explain.predicted

  items.push({ label: 'Baseline', start: 0, end: baseline, isBaseline: true })

  let running = baseline
  const contribs = explain.model === 'XGBoost'
    ? (explain.drivers ?? []).map(d => ({ name: d.name, value: d.contribution }))
    : (explain.components ?? []).map(c => ({ name: c.name, value: c.contribution }))

  for (const c of contribs) {
    items.push({ label: c.name, start: running, end: running + c.value, value: c.value })
    running += c.value
  }

  // Any remaining SHAP contributions not stored as top-3 drivers
  const remaining = explain.predicted - running
  if (Math.abs(remaining) > 0.005) {
    items.push({ label: 'Other factors', start: running, end: running + remaining, value: remaining })
    running += remaining
  }

  // Total bar
  items.push({ label: 'Predicted', start: 0, end: explain.predicted, isTotal: true })

  return items
}

function drawWaterfall(explain) {
  if (waterfallInstance) { waterfallInstance.destroy(); waterfallInstance = null }
  if (!waterfallCanvas.value) return

  const items = buildWaterfallData(explain)

  const colors = items.map(item => {
    if (item.isBaseline) return 'rgba(99,149,237,0.85)'
    if (item.isTotal)    return 'rgba(245,158,11,0.85)'
    return item.value >= 0 ? 'rgba(34,197,94,0.85)' : 'rgba(239,68,68,0.85)'
  })

  waterfallInstance = new Chart(waterfallCanvas.value, {
    type: 'bar',
    data: {
      labels: items.map(i => i.label),
      datasets: [{
        data: items.map(i => [Math.min(i.start, i.end), Math.max(i.start, i.end)]),
        backgroundColor: colors,
        borderWidth: 0,
        borderRadius: 3,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: ctx => items[ctx[0]?.dataIndex]?.label ?? '',
            label: ctx => {
              const item = items[ctx.dataIndex]
              if (!item) return ''
              if (item.isBaseline) return ` Baseline: ${item.end.toFixed(1)} crimes`
              if (item.isTotal) return ` Predicted total: ${item.end.toFixed(1)} crimes`
              const sign = item.value >= 0 ? '+' : ''
              return ` Contribution: ${sign}${item.value.toFixed(2)} crimes`
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#aaa', font: { size: 10 } },
          grid:  { color: '#2a2a4a' },
          title: { display: true, text: 'crimes / month', color: '#888', font: { size: 10 } },
        },
        y: {
          ticks: { color: '#ccc', font: { size: 10 } },
          grid:  { color: '#2a2a4a' },
        },
      },
    },
  })
}

async function fetchExplain(month, predicted) {
  isExplainLoading.value = true
  explainData.value = null
  try {
    const res = await fetch(buildExplainUrl(month))
    if (!res.ok) return
    explainData.value = { ...await res.json(), predicted }
  } finally {
    isExplainLoading.value = false
  }
}

// Redraw waterfall when explainData is set (canvas is v-if so must wait nextTick)
watch(explainData, async (val) => {
  if (!val) {
    if (waterfallInstance) { waterfallInstance.destroy(); waterfallInstance = null }
    return
  }
  await nextTick()
  drawWaterfall(val)
})

watch(activeTab, async (tab) => {
  if (tab !== 'forecast') return

  await nextTick()
  if (!isAllCrimeTypes.value && !isLoading.value && !error.value && predictions.value.length > 0) {
    drawChart()
  }
})

watch(() => props.lsoaCode, (code) => {
  if (!code) activeTab.value = 'forecast'
})

async function fetchForecast() {
  if (!props.lsoaCode) return

  isLoading.value = true
  error.value = null
  predictions.value = []
  explainData.value = null
  selectedIdx.value = null

  if (isAllCrimeTypes.value) {
    if (chartInstance) { chartInstance.destroy(); chartInstance = null }
    savings.value = null
    savingsError.value = null
    isLoading.value = false
    return
  }

  try {
    const res = await fetch(buildUrl())
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    predictions.value = data.predictions ?? []
    await fetchSavings()
    isLoading.value = false
    await nextTick()
    if (activeTab.value === 'forecast' && predictions.value.length > 0) drawChart()
  } catch (e) {
    error.value = 'Could not load forecast data'
  } finally {
    isLoading.value = false
  }
}

watch(
  () => [props.policeForce, props.lsoaCode, props.crimeType, props.startMonth, props.endMonth, props.alertLevels],
  fetchForecast,
  { immediate: true }
)

function buildSavingsUrl(predictedCount) {
  return `/api/v1/savings/${encodeURIComponent(props.crimeType)}/${predictedCount}`
}

async function fetchSavings() {
  savings.value = null
  savingsError.value = null

  const predictedCount = alertPredictedCount.value
  if (isAllCrimeTypes.value || !props.lsoaCode || !alertMonth.value || predictedCount === null) return

  isSavingsLoading.value = true
  try {
    const res = await fetch(buildSavingsUrl(predictedCount))
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    savings.value = await res.json()
  } catch (e) {
    savingsError.value = 'Could not load savings'
  } finally {
    isSavingsLoading.value = false
  }
}
</script>

<style scoped>
.chart-panel {
  width: 380px;
  flex-shrink: 0;
  background: #16213e;
  border-left: 2px solid #0f3460;
  display: flex;
  flex-direction: column;
  padding: 16px;
  gap: 12px;
  transition: width 0.2s ease;
  overflow-y: auto;
}

.chart-panel.expanded {
  width: 720px;
}

.chart-title {
  font-size: 0.8rem;
  color: #f59e0b;
  font-weight: bold;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.panel-tabs {
  display: flex;
  gap: 4px;
  border-bottom: 1px solid #0f3460;
}

.panel-tab {
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  color: #94a3b8;
  cursor: pointer;
  font-size: 0.78rem;
  font-weight: bold;
  padding: 6px 10px;
}

.panel-tab:hover {
  color: #e5e7eb;
}

.panel-tab.active {
  border-bottom-color: #f59e0b;
  color: #f59e0b;
}

.chart-placeholder {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #555;
  font-size: 0.85rem;
  text-align: center;
}

.chart-placeholder.error { color: #e94560; }

.all-alert-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.all-alert-title {
  color: #f59e0b;
  font-size: 0.9rem;
  font-weight: bold;
}

.all-alert-subtitle {
  color: #94a3b8;
  font-size: 0.78rem;
}

.all-alert-empty {
  align-items: center;
  color: #aaa;
  display: flex;
  font-size: 0.85rem;
  justify-content: center;
  min-height: 180px;
  text-align: center;
}

.all-alert-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.all-alert-item {
  align-items: center;
  background: #0f3460;
  border-left: 3px solid #e94560;
  border-radius: 5px;
  display: flex;
  gap: 12px;
  justify-content: space-between;
  padding: 10px 12px;
}

.all-alert-crime {
  color: #e5e7eb;
  font-size: 0.85rem;
  font-weight: bold;
}

.all-alert-meta {
  color: #94a3b8;
  font-size: 0.72rem;
  margin-top: 3px;
}

.forecast-canvas {
  height: clamp(420px, 48vh, 540px) !important;
  flex: none;
  cursor: pointer;
}

.click-hint {
  font-size: 0.7rem;
  color: #555;
  text-align: center;
  margin: 0;
}

/* ── Explain panel ───────────────────────────────────────────── */
.explain-panel {
  border: 1px solid #0f3460;
  border-radius: 6px;
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  background: #0d1b35;
}

.explain-header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.model-badge {
  font-size: 0.65rem;
  font-weight: bold;
  padding: 2px 7px;
  border-radius: 10px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.model-badge.xgb { background: #1e40af; color: #bfdbfe; }
.model-badge.sar { background: #7c2d12; color: #fed7aa; }

.explain-month {
  font-size: 0.75rem;
  color: #94a3b8;
}

.explain-close {
  margin-left: auto;
  background: none;
  border: none;
  color: #555;
  cursor: pointer;
  font-size: 0.85rem;
  padding: 0 4px;
}

.explain-close:hover { color: #e94560; }

.explain-summary {
  font-size: 0.72rem;
  color: #94a3b8;
  margin: 0;
  line-height: 1.4;
}

.waterfall-canvas {
  height: 180px !important;
  flex: none;
  cursor: default;
}

/* ── Savings panel ───────────────────────────────────────────── */
.savings-panel {
  border-top: 1px solid #0f3460;
  padding-top: 12px;
  color: #e5e7eb;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.savings-title {
  font-size: 0.8rem;
  color: #f59e0b;
  font-weight: bold;
}

.savings-context {
  font-size: 0.8rem;
  color: #cbd5e1;
}

.savings-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}

.savings-grid div {
  background: #0f3460;
  padding: 8px;
  border-radius: 4px;
}

.savings-grid span {
  display: block;
  font-size: 0.7rem;
  color: #aaa;
  margin-bottom: 4px;
}

.savings-grid strong {
  font-size: 0.85rem;
  color: #fff;
}

.savings-muted { color: #aaa; font-size: 0.85rem; }
.savings-error { color: #e94560; font-size: 0.85rem; }

.recommendation-panel {
  border-radius: 6px;
  padding: 10px 12px;
  background: #0f3460;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.recommendation-title {
  font-size: 0.7rem;
  color: #aaa;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.recommendation-name {
  font-size: 0.85rem;
  color: #f59e0b;
  font-weight: bold;
}

.effectiveness-badge {
  font-size: 0.7rem;
  background: #14532d;
  color: #86efac;
  padding: 2px 8px;
  border-radius: 10px;
}

.source-text {
  font-size: 0.7rem;
  color: #64748b;
}

.recommendation-description {
  font-size: 0.75rem;
  font-weight: bold;
  margin-top: 6px;
}

.recommendation-grid {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.recommendation-metric-box {
  background: transparent;
  border-radius: 4px;
  padding: 8px 12px;
  text-align: center;
}

.metric-value {
  font-size: 1rem;
  font-weight: bold;
  color: #86efac;
}

.metric-label {
  font-size: 0.7rem;
  color: #86efac;
}

.recommendation-divider {
  border: none;
  border-top: 1px solid #64748b;
  margin: 4px 0;
}

.footnote-marker {
  font-size: 0.6rem;
  color: #64748b;
  vertical-align: super;
}

</style>