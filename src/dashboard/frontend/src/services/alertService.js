export async function fetchAlerts({
    policeForce,
    crimeType,
    startMonth,
    endMonth
}) {
    const params = new URLSearchParams({
        crime_type: crimeType,
        time_start: startMonth,
        time_end: endMonth
    })

    const response = await fetch(`/api/v1/alerts/${policeForce}?${params.toString()}`)

    if (!response.ok) {
        throw new Error(`Failed to fetch alerts: ${response.status}`)
    }

    const data = await response.json()

    return data.alert_levels ?? {}
}