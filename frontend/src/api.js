// Central API client — all requests go through here
const BASE = '/api'

async function get(path) {
  let res
  try {
    res = await fetch(`${BASE}${path}`)
  } catch (networkErr) {
    throw new Error('Cannot reach backend. Start the API server: python api.py')
  }
  if (!res.ok) {
    let msg = res.statusText
    try {
      const body = await res.json()
      msg = body.detail || body.error || msg
    } catch { /* ignore parse error */ }
    throw new Error(msg)
  }
  return res.json()
}

async function post(path, body) {
  let res
  try {
    res = await fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch (networkErr) {
    throw new Error('Cannot reach backend. Start the API server: python api.py')
  }
  if (!res.ok) {
    let msg = res.statusText
    try {
      const body = await res.json()
      msg = body.detail || body.error || msg
    } catch { /* ignore parse error */ }
    throw new Error(msg)
  }
  return res.json()
}

export const api = {
  health:                    ()       => get('/health'),
  metrics:                   ()       => get('/metrics'),
  kpis:                      ()       => get('/kpis'),
  filterOptions:             ()       => get('/filter-options'),
  predictions:               (params) => get(`/predictions?${new URLSearchParams(params)}`),
  riskDistribution:          ()       => get('/charts/risk-distribution'),
  attackCategories:          ()       => get('/charts/attack-categories'),
  attackCategoriesSuspicious:()       => get('/charts/attack-categories-suspicious'),
  protocolTraffic:           ()       => get('/charts/protocol-traffic'),
  protocolStatus:            ()       => get('/charts/protocol-status'),
  predictionSplit:           ()       => get('/charts/prediction-split'),
  actualVsPredicted:         ()       => get('/charts/actual-vs-predicted'),
  probDistribution:          ()       => get('/charts/prob-distribution'),
  predict:                   (data)   => post('/predict', data),
}
