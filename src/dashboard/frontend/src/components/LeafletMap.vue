<template>
  <div ref="mapEl" class="map"></div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

const props = defineProps({
  policeForce: { type: String, required: true },
  searchedLsoa: { type: String, default: ''},
  alertLevels: { type: Object, default: () => ({})}
})

function getAlertInfo(code) {
  const entry = props.alertLevels[code]
  if (!entry) return null
  return typeof entry === 'number' ? { level: entry, dates: [] } : entry
}

const emit = defineEmits(['data-loaded', 'lsoa-selected'])
const mapEl = ref(null)

let map = null
let geojsonLayer = null

const lsoaLayers = {}

function featureStyle(feature) {
  const p = feature.properties
  const code = p.LSOA21CD

  const alertLevel = getAlertInfo(code)?.level
  if (alertLevel === 2) {
    return { color: '#991b1b', weight: 2, fillColor: '#991b1b', fillOpacity: 0.75 }
  }
  if (alertLevel === 1) {
    return {
      color: '#dc2626',
      weight: 1.5,
      fillColor: '#dc2626',
      fillOpacity: 0.65
    }
  }

  if (p.is_hotspot) {
    return {
      color: '#f59e0b',
      weight: 1,
      fillColor: '#f59e0b',
      fillOpacity: 0.65
    }
  }

  return {
    color: '#94a3b8',
    weight: 0.5,
    fillColor: '#f8fafc',
    fillOpacity: 0.55
  }
}

function getPopupContent(properties) {
  const lsoaCode = properties.LSOA21CD
  const alertInfo = getAlertInfo(lsoaCode)
  const alertText = alertInfo?.level === 2 ? 'High' : alertInfo?.level === 1 ? 'Medium' : 'None'
  const datesLine = alertInfo?.dates?.length ? `<br><b>Alert dates:</b> ${alertInfo.dates.join(', ')}` : ''
  return `
  <b>${properties.LSOA21CD}</b><br>
  ${properties.LSOA21NM}<br>
  <b>Status:</b> ${properties.is_hotspot ? 'Hotspot' : 'Non-hotspot'}<br>
  <b>Alert:</b> ${alertText} ${datesLine}
  `
}

function drawForce(geojson) {
  if (geojsonLayer) {
    map.removeLayer(geojsonLayer)
    geojsonLayer = null
  }

  Object.keys(lsoaLayers).forEach(key => delete lsoaLayers[key])

  geojsonLayer = L.geoJson(geojson, {
    style: featureStyle,

    onEachFeature(feature, layer) {
      const p = feature.properties

      lsoaLayers[p.LSOA21CD] = layer

      layer.on({
        mouseover(e) {
          e.target.setStyle({ weight: 2, fillOpacity: p.is_hotspot ? 0.65 : 0.45 })
        },
        mouseout() {
          geojsonLayer.resetStyle(layer)
        },
        click() {
          layer.setPopupContent(getPopupContent(p))
          emit('lsoa-selected', { code: p.LSOA21CD, name: p.LSOA21NM })
        }
      })

      layer.bindPopup(getPopupContent(p))
    }
  }).addTo(map)

  if (geojsonLayer.getBounds().isValid()) {
    map.fitBounds(geojsonLayer.getBounds(), {
      padding: [20, 20]
    })
  }

  emit('data-loaded', geojson.features.length)
}

// When policeForce changes, fetch only that force's GeoJSON from the backend
watch(() => props.policeForce, async (force) => {
  if (!map) return
  const res = await fetch(`/api/v1/geojson/${force}`)
  const data = await res.json()
  drawForce(data)
})

watch(() => props.searchedLsoa, (newCode) => {
  if (!newCode) return

  const layer = lsoaLayers[newCode]

  if (!layer) {
    alert('LSOA not found')
    return
  }

  const p = layer.feature.properties

  layer.openPopup()

  layer.setStyle({weight: 3, fillOpacity: 0.8})

  layer.bringToFront()

  map.fitBounds(layer.getBounds())

  emit('lsoa-selected', {code: p.LSOA21CD, name: p.LSOA21NM})
})

watch(() => props.alertLevels, () => {
  if (!geojsonLayer) return

  geojsonLayer.setStyle(featureStyle)
}, { deep: true })

onMounted(async () => {
  map = L.map(mapEl.value).setView([52.67, -1.98], 6.5)

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    opacity: 0.5
  }).addTo(map)

  const legend = L.control({ position: 'bottomright' })
  legend.onAdd = function () {
    const div = L.DomUtil.create('div', 'map-legend')

    div.innerHTML = `
    <div><span class="legend-box alert"></span> Alert</div>
    <div><span class="legend-box hotspot"></span> Hotspot</div>
    <div><span class="legend-box non-hotspot"></span> Non-hotspot</div>
    `

    return div
  }
  legend.addTo(map)

  // Initial load
  const res = await fetch(`/api/v1/geojson/${props.policeForce}`)
  const data = await res.json()
  drawForce(data)
})
</script>

<style scoped>
.map {
  flex: 1;
  width: 100%;
  min-height: 0;
}

:global(.map-legend) {
  background: white;
  padding: 8px 10px;
  border-radius: 6px;
  font-size: 12px;
  color: #111;
  box-shadow: 0 1px 5px rgba(0, 0, 0, 0.3);
}

:global(.legend-box) {
  display: inline-block;
  width: 14px;
  height: 14px;
  margin-right: 6px;
  border: 1px solid #555;
  vertical-align: middle;
}

:global(.legend-box.alert) {
  background: #dc2626;
}

:global(.legend-box.hotspot) {
  background: #f59e0b;
}

:global(.legend-box.non-hotspot) {
  background: #f8fafc;
}
</style>