<template>
  <header class="header">
    <h1>Crime Dashboard</h1>

    <span v-if="isLoading" class="badge loading">Loading map data...</span>
    <span v-else class="badge ready">{{ lsoaCount.toLocaleString() }} LSOAs loaded</span>

    <div class="controls">

      <!-- 1. Police force first — drives which LSOAs are highlighted -->
      <div class="control-group">
        <label>Police Force</label>
        <select v-model="policeForce">
          <option v-for="force in policeForces" :key="force" :value="force">
            {{ formatForce(force) }}
          </option>
        </select>
      </div>

      <div class="control-group">
        <label>Crime Type</label>
        <select v-model="selectedCrimeType">
          <option>All</option>
          <option>Violence and sexual offences</option>
          <option>Anti-social behaviour</option>
          <option>Shoplifting</option>
          <option>Criminal damage and arson</option>
          <option>Other theft</option>
          <option>Public order</option>
          <option>Vehicle crime</option>
          <option>Burglary</option>
          <option>Drugs</option>
          <option>Theft from the person</option>
          <option>Robbery</option>
          <option>Bicycle theft</option>
          <option>Possession of weapons</option>
          <option>Other crime</option>
        </select>
      </div>

      <div class="control-group">
        <label>Start Month</label>
        <input type="month" v-model="startMonth">
      </div>

      <div class="control-group">
        <label>End Month</label>
        <input type="month" v-model="endMonth">
      </div>

      <div class="control-group">
        <label>Search LSOA</label>

        <div class="search-row">
          <input type="text" v-model="searchInput" placeholder="E01000001">

          <button @click="searchLSOA">Search</button>
        </div>
      </div>

    </div>

    <div class="selected-info">
      <template v-if="selectedLSOA">
        Selected: <strong>{{ selectedLSOA.name }}</strong>
        &nbsp;({{ selectedLSOA.code }})
      </template>
      <template v-else>
        Click a highlighted area to select
      </template>
    </div>
  </header>

  <div class="content">
    <LeafletMap
      :police-force="policeForce"
      :searched-lsoa="searchedLSOA"
      :alert-levels="alertLevels"
      @data-loaded="onDataLoaded"
      @lsoa-selected="onLSOASelected"
    />
    <ForecastChart
      :police-force="policeForce"
      :lsoa-code="selectedLSOA ? selectedLSOA.code : null"
      :crime-type="selectedCrimeType"
      :start-month="startMonth"
      :end-month="endMonth"
      :alert-levels="alertLevels"
    />
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import LeafletMap from './components/LeafletMap.vue'
import ForecastChart from './components/ForecastChart.vue'
import { fetchAlerts } from './services/alertService.js'

const policeForces = [
  'avon-and-somerset', 'bedfordshire', 'cambridgeshire', 'cheshire',
  'city-of-london', 'cleveland', 'cumbria', 'derbyshire', 'devon-and-cornwall',
  'dorset', 'durham', 'dyfed-powys', 'essex', 'gloucestershire', 'gwent',
  'hampshire', 'hertfordshire', 'humberside', 'kent', 'lancashire',
  'leicestershire', 'lincolnshire', 'merseyside', 'metropolitan', 'norfolk',
  'north-wales', 'north-yorkshire', 'northamptonshire', 'northumbria',
  'nottinghamshire', 'south-wales', 'south-yorkshire', 'staffordshire',
  'suffolk', 'surrey', 'sussex', 'thames-valley', 'warwickshire',
  'west-mercia', 'west-midlands', 'west-yorkshire', 'wiltshire',
]

// "avon-and-somerset" → "Avon And Somerset"
function formatForce(slug) {
  return slug.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
}

const isLoading = ref(true)
const lsoaCount = ref(0)
const selectedLSOA = ref(null)
const selectedCrimeType = ref('Burglary')
const startMonth = ref('2026-06')
const endMonth = ref('2026-11')

const policeForce = ref('avon-and-somerset')

watch (policeForce, () => {
  selectedLSOA.value = null
  isLoading.value = true
})

const searchInput = ref('')
const searchedLSOA = ref('')

const alertLevels = ref({})
const isAlertLoading = ref(false)

watch ([policeForce, selectedCrimeType, startMonth, endMonth], async () => {
  isAlertLoading.value = true

  try {
    alertLevels.value = await fetchAlerts({
      policeForce: policeForce.value,
      crimeType: selectedCrimeType.value,
      startMonth: startMonth.value,
      endMonth: endMonth.value
    })
  } finally {
    isAlertLoading.value = false
  }
}, { immediate: true })

function onDataLoaded(count) {
  lsoaCount.value = count
  isLoading.value = false
}

function onLSOASelected(lsoa) {
  selectedLSOA.value = lsoa
}

function searchLSOA() {
  searchedLSOA.value = searchInput.value.trim().toUpperCase()
}
</script>

<style scoped>
.header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 10px 20px;
  background: #16213e;
  border-bottom: 2px solid #0f3460;
  flex-shrink: 0;
}

.header h1 {
  font-size: 1.2rem;
  color: #e94560;
  white-space: nowrap;
}

.badge {
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: bold;
  white-space: nowrap;
}

.badge.loading {
  background: #f39c12;
  color: #000;
}

.badge.ready {
  background: #27ae60;
  color: #fff;
}

.selected-info {
  font-size: 0.85rem;
  color: #aaa;
  margin-left: auto;
}

.content {
  display: flex;
  flex: 1;
  min-height: 0;
}

.controls {
  display: flex;
  gap: 16px;
  align-items: center;
}

.control-group {
  display: flex;
  flex-direction: column;
  font-size: 0.75rem;
  color: #ccc;
}

.control-group label {
  margin-bottom: 2px;
}

.control-group select, .control-group input {
  padding: 4px 6px;
  border-radius: 4px;
  border: none;
  background: #f5f5f5;
}

.search-row {
  display: flex;
  gap: 6px;
}

.search-row button {
  padding: 4px 10px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  background: #e94560;
  color: white;
}
</style>