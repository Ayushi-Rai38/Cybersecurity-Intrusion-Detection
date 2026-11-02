import { useState } from 'react'
import { api } from '../api.js'
import s from './shared.module.css'
import p from './Predict.module.css'

const PROTO_OPTIONS  = ['tcp', 'udp', 'unas', 'arp', 'ospf', 'icmp', 'igmp', 'rtp', 'sctp', 'other']
const STATE_OPTIONS  = ['FIN', 'INT', 'CON', 'ECO', 'REQ', 'RST', 'PAR', 'URN', 'no', '-']

// All numeric fields with defaults
const NUMERIC_FIELDS = [
  { key: 'dur',               label: 'Duration (s)',         defaultVal: 0.0,   step: 0.0001, min: 0 },
  { key: 'sbytes',            label: 'Src Bytes',            defaultVal: 0,     step: 1,      min: 0 },
  { key: 'dbytes',            label: 'Dst Bytes',            defaultVal: 0,     step: 1,      min: 0 },
  { key: 'spkts',             label: 'Src Packets',          defaultVal: 0,     step: 1,      min: 0 },
  { key: 'dpkts',             label: 'Dst Packets',          defaultVal: 0,     step: 1,      min: 0 },
  { key: 'rate',              label: 'Rate (pkt/s)',         defaultVal: 0.0,   step: 0.001,  min: 0 },
  { key: 'sttl',              label: 'Src TTL',              defaultVal: 64,    step: 1,      min: 0, max: 255 },
  { key: 'dttl',              label: 'Dst TTL',              defaultVal: 0,     step: 1,      min: 0, max: 255 },
  { key: 'sload',             label: 'Src Load (bps)',       defaultVal: 0.0,   step: 0.001,  min: 0 },
  { key: 'dload',             label: 'Dst Load (bps)',       defaultVal: 0.0,   step: 0.001,  min: 0 },
  { key: 'sloss',             label: 'Src Packet Loss',      defaultVal: 0,     step: 1,      min: 0 },
  { key: 'dloss',             label: 'Dst Packet Loss',      defaultVal: 0,     step: 1,      min: 0 },
  { key: 'sinpkt',            label: 'Src Inter-pkt (ms)',   defaultVal: 0.0,   step: 0.001,  min: 0 },
  { key: 'dinpkt',            label: 'Dst Inter-pkt (ms)',   defaultVal: 0.0,   step: 0.001,  min: 0 },
  { key: 'sjit',              label: 'Src Jitter (ms)',      defaultVal: 0.0,   step: 0.001,  min: 0 },
  { key: 'djit',              label: 'Dst Jitter (ms)',      defaultVal: 0.0,   step: 0.001,  min: 0 },
  { key: 'swin',              label: 'Src TCP Window',       defaultVal: 0,     step: 1,      min: 0 },
  { key: 'dwin',              label: 'Dst TCP Window',       defaultVal: 0,     step: 1,      min: 0 },
  { key: 'stcpb',             label: 'Src TCP Base Seq',     defaultVal: 0,     step: 1,      min: 0 },
  { key: 'dtcpb',             label: 'Dst TCP Base Seq',     defaultVal: 0,     step: 1,      min: 0 },
  { key: 'tcprtt',            label: 'TCP RTT (s)',          defaultVal: 0.0,   step: 0.0001, min: 0 },
  { key: 'synack',            label: 'SYN→ACK time (s)',     defaultVal: 0.0,   step: 0.0001, min: 0 },
  { key: 'ackdat',            label: 'ACK→DAT time (s)',     defaultVal: 0.0,   step: 0.0001, min: 0 },
  { key: 'smean',             label: 'Src Mean Pkt Size',    defaultVal: 0,     step: 1,      min: 0 },
  { key: 'dmean',             label: 'Dst Mean Pkt Size',    defaultVal: 0,     step: 1,      min: 0 },
  { key: 'trans_depth',       label: 'Trans Depth',          defaultVal: 0,     step: 1,      min: 0 },
  { key: 'response_body_len', label: 'Response Body Len',    defaultVal: 0,     step: 1,      min: 0 },
  { key: 'ct_srv_src',        label: 'CT Srv→Src',           defaultVal: 0,     step: 1,      min: 0 },
  { key: 'ct_state_ttl',      label: 'CT State TTL',         defaultVal: 0,     step: 1,      min: 0 },
  { key: 'ct_dst_ltm',        label: 'CT Dst LTM',           defaultVal: 0,     step: 1,      min: 0 },
  { key: 'ct_src_dport_ltm',  label: 'CT Src Dport LTM',     defaultVal: 0,     step: 1,      min: 0 },
  { key: 'ct_dst_sport_ltm',  label: 'CT Dst Sport LTM',     defaultVal: 0,     step: 1,      min: 0 },
  { key: 'ct_dst_src_ltm',    label: 'CT Dst Src LTM',       defaultVal: 0,     step: 1,      min: 0 },
  { key: 'is_ftp_login',      label: 'Is FTP Login',         defaultVal: 0,     step: 1,      min: 0, max: 1 },
  { key: 'ct_ftp_cmd',        label: 'CT FTP Cmd',           defaultVal: 0,     step: 1,      min: 0 },
  { key: 'ct_flw_http_mthd',  label: 'CT Flow HTTP Methods', defaultVal: 0,     step: 1,      min: 0 },
  { key: 'ct_src_ltm',        label: 'CT Src LTM',           defaultVal: 0,     step: 1,      min: 0 },
  { key: 'ct_srv_dst',        label: 'CT Srv→Dst',           defaultVal: 0,     step: 1,      min: 0 },
  { key: 'is_sm_ips_ports',   label: 'Is SM IPs/Ports',      defaultVal: 0,     step: 1,      min: 0, max: 1 },
]

// Attack-like preset
const ATTACK_PRESET = {
  proto: 'tcp', service: 'http', state: 'FIN',
  dur: 0.0001, sbytes: 52000, dbytes: 0, spkts: 500, dpkts: 0,
  rate: 5000.0, sttl: 254, dttl: 0, sload: 500000, dload: 0,
  sloss: 40, dloss: 0, sinpkt: 0.0002, dinpkt: 0, sjit: 0, djit: 0,
  swin: 0, dwin: 0, stcpb: 0, dtcpb: 0, tcprtt: 0, synack: 0, ackdat: 0,
  smean: 104, dmean: 0, trans_depth: 0, response_body_len: 0,
  ct_srv_src: 50, ct_state_ttl: 2, ct_dst_ltm: 0, ct_src_dport_ltm: 0,
  ct_dst_sport_ltm: 0, ct_dst_src_ltm: 0, is_ftp_login: 0, ct_ftp_cmd: 0,
  ct_flw_http_mthd: 0, ct_src_ltm: 50, ct_srv_dst: 1, is_sm_ips_ports: 0,
}

// Normal-like preset
const NORMAL_PRESET = {
  proto: 'tcp', service: 'http', state: 'FIN',
  dur: 0.212, sbytes: 432, dbytes: 10240, spkts: 6, dpkts: 8,
  rate: 65.0, sttl: 64, dttl: 64, sload: 2100, dload: 49000,
  sloss: 0, dloss: 0, sinpkt: 35.0, dinpkt: 26.5, sjit: 0.5, djit: 0.3,
  swin: 255, dwin: 255, stcpb: 0, dtcpb: 0, tcprtt: 0.045, synack: 0.022, ackdat: 0.023,
  smean: 72, dmean: 1280, trans_depth: 1, response_body_len: 9800,
  ct_srv_src: 3, ct_state_ttl: 3, ct_dst_ltm: 2, ct_src_dport_ltm: 1,
  ct_dst_sport_ltm: 1, ct_dst_src_ltm: 2, is_ftp_login: 0, ct_ftp_cmd: 0,
  ct_flw_http_mthd: 1, ct_src_ltm: 3, ct_srv_dst: 4, is_sm_ips_ports: 0,
}

function buildDefault() {
  const d = { proto: 'tcp', service: '-', state: 'FIN' }
  NUMERIC_FIELDS.forEach(f => { d[f.key] = f.defaultVal })
  return d
}

export default function Predict() {
  const [form, setForm]       = useState(buildDefault)
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [section, setSection] = useState('basic') // 'basic' | 'advanced'

  const set = (key, val) => setForm(f => ({ ...f, [key]: val }))

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      // Cast numeric strings to numbers
      const payload = { ...form }
      NUMERIC_FIELDS.forEach(f => { payload[f.key] = Number(payload[f.key]) })
      const res = await api.predict(payload)
      setResult(res)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function applyPreset(preset) {
    setForm({ ...buildDefault(), ...preset })
    setResult(null)
    setError(null)
  }

  const BASIC_KEYS  = ['dur','sbytes','dbytes','spkts','dpkts','rate','sttl','dttl','sload','dload']
  const basicFields = NUMERIC_FIELDS.filter(f => BASIC_KEYS.includes(f.key))
  const advFields   = NUMERIC_FIELDS.filter(f => !BASIC_KEYS.includes(f.key))

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'1.4rem' }}>

      {/* Header */}
      <div className={s.sectionTitle}>
        <span className={s.sectionDot}/>
        Real-Time Intrusion Prediction
        <span style={{ fontSize:'0.72rem', color:'var(--muted)', marginLeft:'0.6rem' }}>
          Uses the saved Random Forest model · First call may take ~30s to load Spark
        </span>
      </div>

      {/* Preset buttons */}
      <div style={{ display:'flex', gap:'0.75rem', flexWrap:'wrap' }}>
        <button className={p.presetBtn} data-variant="normal" onClick={() => applyPreset(NORMAL_PRESET)}>
          ✅ Load Normal Traffic Preset
        </button>
        <button className={p.presetBtn} data-variant="attack" onClick={() => applyPreset(ATTACK_PRESET)}>
          🚨 Load Attack Traffic Preset
        </button>
        <button className={p.presetBtn} data-variant="reset" onClick={() => { setForm(buildDefault()); setResult(null); setError(null) }}>
          ↺ Reset
        </button>
      </div>

      <div style={{ display:'grid', gridTemplateColumns:'1fr 340px', gap:'1.4rem', alignItems:'start' }}>

        {/* Form */}
        <form onSubmit={handleSubmit} className={p.formCard}>

          {/* Categorical row */}
          <div className={p.catRow}>
            <div className={p.field}>
              <label className={p.label}>Protocol</label>
              <select className={p.select} value={form.proto} onChange={e => set('proto', e.target.value)}>
                {PROTO_OPTIONS.map(v => <option key={v}>{v}</option>)}
              </select>
            </div>
            <div className={p.field}>
              <label className={p.label}>Service</label>
              <input className={p.input} value={form.service} onChange={e => set('service', e.target.value)} placeholder='e.g. http, dns, ftp, -' />
            </div>
            <div className={p.field}>
              <label className={p.label}>State</label>
              <select className={p.select} value={form.state} onChange={e => set('state', e.target.value)}>
                {STATE_OPTIONS.map(v => <option key={v}>{v}</option>)}
              </select>
            </div>
          </div>

          {/* Section tabs */}
          <div className={p.sectionTabs}>
            <button type="button" className={p.sectionTab} data-active={section==='basic'} onClick={() => setSection('basic')}>
              Basic Features
            </button>
            <button type="button" className={p.sectionTab} data-active={section==='advanced'} onClick={() => setSection('advanced')}>
              Advanced Features
            </button>
          </div>

          {/* Numeric fields */}
          <div className={p.numGrid}>
            {(section === 'basic' ? basicFields : advFields).map(f => (
              <div key={f.key} className={p.field}>
                <label className={p.label}>{f.label}</label>
                <input
                  type="number"
                  className={p.input}
                  value={form[f.key]}
                  min={f.min ?? undefined}
                  max={f.max ?? undefined}
                  step={f.step}
                  onChange={e => set(f.key, e.target.value)}
                />
              </div>
            ))}
          </div>

          <button type="submit" className={p.submitBtn} disabled={loading}>
            {loading ? '⏳ Running Model…' : '⚡ Predict Intrusion'}
          </button>

          {loading && (
            <p className={p.hint}>
              ℹ️ Spark loads on first call — this may take up to 30 seconds. Subsequent calls are fast.
            </p>
          )}
        </form>

        {/* Result panel */}
        <div style={{ display:'flex', flexDirection:'column', gap:'1rem' }}>

          {error && (
            <div className={p.resultCard} data-verdict="error">
              <div className={p.verdictIcon}>⚠️</div>
              <div className={p.verdictLabel}>Error</div>
              <div className={p.verdictDetail} style={{ fontSize:'0.75rem', color:'var(--muted)', marginTop:'0.5rem', wordBreak:'break-word' }}>
                {error}
              </div>
            </div>
          )}

          {result && (
            <div className={p.resultCard} data-verdict={result.label === 'Attack' ? 'attack' : 'normal'}>
              <div className={p.verdictIcon}>{result.label === 'Attack' ? '🚨' : '✅'}</div>
              <div className={p.verdictLabel}>{result.label}</div>

              <div className={p.riskBadge} data-risk={result.risk_level}>
                {result.risk_level} Risk
              </div>

              <div className={p.probBar}>
                <div className={p.probBarInner}
                  style={{
                    width: `${(result.attack_prob * 100).toFixed(1)}%`,
                    background: result.label === 'Attack' ? '#f85149' : '#3fb950',
                  }}
                />
              </div>
              <div className={p.probLabel}>
                Attack Probability: <strong>{(result.attack_prob * 100).toFixed(2)}%</strong>
              </div>
              <div className={p.probLabel}>
                Confidence: <strong>{(result.confidence * 100).toFixed(2)}%</strong>
              </div>

              <div className={p.explanationBox}>
                {result.label === 'Attack'
                  ? '⚠️ The model classifies this traffic as a potential intrusion. High-rate, low-byte, asymmetric connections are typical attack signatures.'
                  : '✅ The model classifies this traffic as normal. Balanced byte ratios and typical TTL values indicate legitimate activity.'
                }
              </div>
            </div>
          )}

          {!result && !error && !loading && (
            <div className={p.resultCard} data-verdict="idle">
              <div className={p.verdictIcon} style={{ fontSize:'2.5rem' }}>🔍</div>
              <div className={p.verdictLabel} style={{ color:'var(--muted)' }}>Awaiting Input</div>
              <div style={{ fontSize:'0.73rem', color:'var(--muted)', marginTop:'0.5rem', textAlign:'center', lineHeight:1.5 }}>
                Fill in the network traffic parameters and click <em>Predict Intrusion</em> to classify the connection using the trained Random Forest model.
              </div>
            </div>
          )}

          {/* Feature reference */}
          <div className={s.card} style={{ fontSize:'0.72rem', color:'var(--muted)', lineHeight:1.7 }}>
            <div style={{ color:'var(--text)', fontWeight:600, marginBottom:'0.4rem' }}>📖 Key Feature Reference</div>
            <div><strong>proto</strong> — Network protocol (tcp/udp/icmp…)</div>
            <div><strong>state</strong> — Connection state (FIN=completed, INT=partial…)</div>
            <div><strong>sbytes/dbytes</strong> — Payload bytes src→dst / dst→src</div>
            <div><strong>sttl/dttl</strong> — Time-to-live (Linux=64, Windows=128, Router=254)</div>
            <div><strong>rate</strong> — Packets per second</div>
            <div><strong>ct_srv_src</strong> — Connections with same service & source in last 100</div>
          </div>
        </div>
      </div>
    </div>
  )
}
