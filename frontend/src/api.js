// Central API client — all requests go through here
const BASE = '/api'

async function get(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error || res.statusText)
  }
  return res.json()
}

export const api = {
  health:                  ()       => get('/health'),
  metrics:                 ()       => get('/metrics'),
  kpis:                    ()       => get('/kpis'),
  filterOptions:           ()       => get('/filter-options'),
  predictions:             (params) => get(`/predictions?${new URLSearchParams(params)}`),
  riskDistribution:        ()       => get('/charts/risk-distribution'),
  attackCategories:        ()       => get('/charts/attack-categories'),
  attackCategoriesSuspicious: ()    => get('/charts/attack-categories-suspicious'),
  protocolTraffic:         ()       => get('/charts/protocol-traffic'),
  protocolStatus:          ()       => get('/charts/protocol-status'),
  predictionSplit:         ()       => get('/charts/prediction-split'),
  actualVsPredicted:       ()       => get('/charts/actual-vs-predicted'),
  probDistribution:        ()       => get('/charts/prob-distribution'),
}
