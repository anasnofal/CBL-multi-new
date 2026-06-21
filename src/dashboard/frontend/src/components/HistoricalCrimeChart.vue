<template>
  <section class="historical-panel">
    <div class="historical-title">Historical crime</div>

    <div class="historical-placeholder" v-if="!lsoaCode">
      Select an LSOA to see historical crime counts
    </div>

    <div class="historical-placeholder" v-else-if="isLoading">
      Loading historical data...
    </div>

    <div class="historical-placeholder error" v-else-if="error">
      {{ error }}
    </div>

    <div class="historical-placeholder" v-else-if="rows.length === 0">
      No historical data for this LSOA
    </div>

    <canvas
      ref="chartCanvas"
      class="historical-canvas"
      v-show="lsoaCode && !isLoading && !error && rows.length > 0"
    ></canvas>
  </section>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'
import { Chart, registerables } from 'chart.js'

Chart.register(...registerables)

const props = defineProps({
  policeForce: { type: String, required: true },
  lsoaCode: { type: String, default: null },
})

const chartCanvas = ref(null)
const isLoading = ref(false)
const error = ref(null)
const rows = ref([])

let chartInstance = null

function buildUrl() {
  return `/api/v1/historical/${props.policeForce}/${encodeURIComponent(props.lsoaCode)}`
}

function asCount(value) {
  const count = Number(value)
  return Number.isFinite(count) ? count : 0
}

function chartRows() {
  return [...rows.value]
    .sort((a, b) => asCount(b.all_time_monthly_avg) - asCount(a.all_time_monthly_avg))
}

function drawChart() {
  if (!chartCanvas.value) return

  if (chartInstance) {
    chartInstance.destroy()
    chartInstance = null
  }

  const dataRows = chartRows()

  chartInstance = new Chart(chartCanvas.value, {
    type: 'bar',
    data: {
      labels: dataRows.map(row => row.crime_type),
      datasets: [
        {
          label: 'Last month',
          data: dataRows.map(row => asCount(row.last_month_count)),
          backgroundColor: 'rgba(233,69,96,0.8)',
          borderRadius: 3,
        },
        {
          label: 'Last-year monthly avg',
          data: dataRows.map(row => asCount(row.last_year_monthly_avg)),
          backgroundColor: 'rgba(245,158,11,0.8)',
          borderRadius: 3,
        },
        {
          label: 'All-time monthly avg',
          data: dataRows.map(row => asCount(row.all_time_monthly_avg)),
          backgroundColor: 'rgba(96,165,250,0.75)',
          borderRadius: 3,
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          beginAtZero: true,
          ticks: { color: '#aaa' },
          grid: { color: '#2a2a4a' },
          title: { display: true, text: 'crimes / month', color: '#888' },
        },
        y: {
          ticks: { color: '#ccc', font: { size: 10 } },
          grid: { display: false },
        },
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#ccc', boxWidth: 10, font: { size: 10 } },
        },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${ctx.parsed.x.toFixed(1)} crimes`,
          },
        },
      },
    },
  })
}

async function fetchHistorical() {
  if (!props.lsoaCode) {
    rows.value = []
    error.value = null
    if (chartInstance) {
      chartInstance.destroy()
      chartInstance = null
    }
    return
  }

  isLoading.value = true
  error.value = null
  rows.value = []

  try {
    const res = await fetch(buildUrl())
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    rows.value = data.data ?? []
    isLoading.value = false
    await nextTick()
    if (rows.value.length > 0) drawChart()
  } catch (e) {
    error.value = 'Could not load historical data'
  } finally {
    isLoading.value = false
  }
}

watch(
  () => [props.policeForce, props.lsoaCode],
  fetchHistorical,
  { immediate: true }
)
</script>

<style scoped>
.historical-panel {
  border-top: 1px solid #0f3460;
  padding-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.historical-title {
  font-size: 0.8rem;
  color: #f59e0b;
  font-weight: bold;
}

.historical-placeholder {
  min-height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #aaa;
  font-size: 0.85rem;
  text-align: center;
}

.historical-placeholder.error {
  color: #e94560;
}

.historical-canvas {
  height: 360px !important;
  flex: none;
}
</style>
