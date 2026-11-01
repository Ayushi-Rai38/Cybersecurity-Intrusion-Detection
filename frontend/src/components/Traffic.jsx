import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, Legend
} from 'recharts'
import { Activity, Network, Database, AlertTriangle } from 'lucide-react'
import { useFetch } from '../hooks/useFetch'
import { api } from '../api'
import s from './shared.module.css'

/* ── Tooltip ──────────────────────────────────────────────────── */
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background:'#161b22', border:'1px solid #30363d',
      borderRadius:8, padding:'8px 12px', fontSize:'0.8rem'
    }}>
      <div style={{color:'#c9d1d9', fontWeight:700, marginBottom:4}}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{color: p.color}}>
          {p.name}: {Number(p.value).toLocaleString()}
        </div>
      ))}
    </div>
  )
}

function BytesTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const fmt = v => v >= 1e6 ? `${(v/1e6).toFixed(2)} MB` : v >= 1000 ? `${(v/1000).toFixed(1)} KB` : `${v} B`
  return (
    <div style={{
      background:'#161b22', border:'1px solid #30363d',
      borderRadius:8, padding:'8px 12px', fontSize:'0.8rem'
    }}>
      <div style={{color:'#c9d1d9', fontWeight:700, marginBottom:4}}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{color: p.color}}>
          {p.name}: {fmt(p.value)}
        </div>
      ))}
    </div>
  )
}

/* ── Stat card ────────────────────────────────────────────────── */
function StatCard({ icon: Icon, label, value, sub, color = '#58a6ff' }) {
  return (
    <div style={{
      background:'#161b22', border:'1px solid #30363d',
      borderRadius:12, padding:'1.1rem 1.2rem',
      borderLeft:`3px solid ${color}`
    }}>
      <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
        <div style={{fontSize:'0.68rem', fontWeight:700, color:'#8b949e',
          textTransform:'uppercase', letterSpacing:'0.06em'}}>{label}</div>
        <Icon size={15} color={color} />
      </div>
      <div style={{fontSize:'1.55rem', fontWeight:800, color:'#c9d1d9',
        letterSpacing:'-0.02em', marginTop:6}}>{value}</div>
      {sub && <div style={{fontSize:'0.68rem', color:'#8b949e', marginTop:3}}>{sub}</div>}
    </div>
  )
}

/* ── Main component ───────────────────────────────────────────── */
export default function Traffic() {
  const { data: protoTraffic, loading: l1, error: e1 } = useFetch(api.protocolTraffic)
  const { data: protoStatus,  loading: l2, error: e2 } = useFetch(api.protocolStatus)
  const { data: kpis,         loading: l3, error: e3 } = useFetch(api.kpis)

  const loading = l1 || l2 || l3
  const error   = e1 || e2 || e3

  if (loading) return <div className={s.loading}>⏳ Loading traffic data…</div>
  if (error)   return <div className={s.error}>⚠ {error}</div>

  /* ── Protocol volume data (top 15 by total bytes) */
  const volData = (protoTraffic?.protocols || [])
    .slice(0, 15)
    .map(p => ({
      name:      p.proto,
      src_bytes: p.total_src_bytes || 0,
      dst_bytes: p.total_dst_bytes || 0,
    }))

  /* ── Protocol status data (top 10, grouped bar: Benign vs Suspicious) */
  const statusData = (protoStatus?.protocols || [])
    .slice(0, 10)
    .map(p => ({
      name:       p.proto,
      Benign:     p.benign_count     || 0,
      Suspicious: p.suspicious_count || 0,
    }))

  /* ── KPI summary values */
  const totalConns  = (kpis?.total_records  || 0)
  const suspConns   = (kpis?.suspicious     || 0)
  const benignConns = (kpis?.benign         || 0)
  const suspPct     = totalConns > 0 ? ((suspConns / totalConns) * 100).toFixed(1) : '0.0'

  /* ── Total bytes across all protocols */
  const totalSrcBytes = (protoTraffic?.protocols || []).reduce((a, p) => a + (p.total_src_bytes || 0), 0)
  const totalDstBytes = (protoTraffic?.protocols || []).reduce((a, p) => a + (p.total_dst_bytes || 0), 0)
  const fmtBytes = v => v >= 1e9 ? `${(v/1e9).toFixed(2)} GB` : v >= 1e6 ? `${(v/1e6).toFixed(1)} MB` : `${(v/1000).toFixed(0)} KB`

  return (
    <div style={{display:'flex', flexDirection:'column', gap:'1.2rem'}}>

      {/* ── Header */}
      <div style={{display:'flex', alignItems:'center', gap:10}}>
        <Activity size={18} color="#58a6ff" />
        <div>
          <div style={{fontSize:'1.05rem', fontWeight:700, color:'#c9d1d9'}}>
            Traffic Analysis
          </div>
          <div style={{fontSize:'0.75rem', color:'#8b949e', marginTop:2}}>
            Protocol-level breakdown of network volume and intrusion predictions
          </div>
        </div>
      </div>

      {/* ── Dataset notice */}
      <div className={s.info}>
        <strong>ℹ Dataset Note:</strong> The UNSW-NB15 dataset does not contain timestamp columns,
        so chronological traffic trends are not available. All charts below show protocol and
        byte-level breakdowns across the full test set of <strong>{totalConns.toLocaleString()}</strong> connections.
      </div>

      {/* ── Summary stat cards */}
      <div className={s.grid4}>
        <StatCard icon={Network}       label="Total Connections"   value={totalConns.toLocaleString()}  color="#58a6ff" />
        <StatCard icon={AlertTriangle} label="Suspicious"          value={suspConns.toLocaleString()}
          sub={`${suspPct}% of traffic`} color="#f85149" />
        <StatCard icon={Activity}      label="Benign"              value={benignConns.toLocaleString()}  color="#3fb950" />
        <StatCard icon={Database}      label="Total Src Bytes"     value={fmtBytes(totalSrcBytes)}       color="#d29922" />
      </div>

      {/* ── Source vs Destination bytes by protocol */}
      <div className={s.card}>
        <div className={s.sectionTitle}>
          <Database size={13} />
          Source vs Destination Bytes by Protocol (Top 15)
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={volData} margin={{top:4, right:20, left:8, bottom:50}}>
            <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
            <XAxis dataKey="name" tick={{fill:'#8b949e', fontSize:11}}
              angle={-35} textAnchor="end" interval={0} />
            <YAxis tick={{fill:'#8b949e', fontSize:11}}
              tickFormatter={v => v >= 1e6 ? `${(v/1e6).toFixed(0)}M` : v >= 1000 ? `${(v/1000).toFixed(0)}K` : v} />
            <Tooltip content={<BytesTooltip />} />
            <Legend wrapperStyle={{paddingTop:12}} formatter={v => (
              <span style={{fontSize:'0.72rem', color:'#8b949e'}}>{v}</span>
            )} />
            <Bar dataKey="src_bytes" name="Src Bytes" fill="#58a6ff" radius={[3,3,0,0]} />
            <Bar dataKey="dst_bytes" name="Dst Bytes" fill="#bc8cff" radius={[3,3,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ── Benign vs Suspicious per protocol */}
      <div className={s.card}>
        <div className={s.sectionTitle}>
          <Network size={13} />
          Benign vs Suspicious Connections by Protocol (Top 10)
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={statusData} margin={{top:4, right:20, left:8, bottom:50}}>
            <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
            <XAxis dataKey="name" tick={{fill:'#8b949e', fontSize:11}}
              angle={-35} textAnchor="end" interval={0} />
            <YAxis tick={{fill:'#8b949e', fontSize:11}}
              tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{paddingTop:12}} formatter={v => (
              <span style={{fontSize:'0.72rem', color:'#8b949e'}}>{v}</span>
            )} />
            <Bar dataKey="Benign"     fill="#3fb950" radius={[3,3,0,0]} />
            <Bar dataKey="Suspicious" fill="#f85149" radius={[3,3,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ── Protocol table */}
      <div className={s.card}>
        <div className={s.sectionTitle}>
          <Network size={13} />
          Protocol Summary Table
        </div>
        <div className={s.tableWrap}>
          <table>
            <thead>
              <tr>
                <th>Protocol</th>
                <th>Total Connections</th>
                <th>Benign</th>
                <th>Suspicious</th>
                <th>Suspicious %</th>
                <th>Src Bytes</th>
                <th>Dst Bytes</th>
              </tr>
            </thead>
            <tbody>
              {(protoStatus?.protocols || []).slice(0, 15).map((p, i) => {
                const tot  = (p.benign_count || 0) + (p.suspicious_count || 0)
                const pct  = tot > 0 ? ((p.suspicious_count / tot) * 100).toFixed(1) : '0.0'
                const srcB = (protoTraffic?.protocols || []).find(x => x.proto === p.proto)?.total_src_bytes || 0
                const dstB = (protoTraffic?.protocols || []).find(x => x.proto === p.proto)?.total_dst_bytes || 0
                const isHigh = parseFloat(pct) > 50
                return (
                  <tr key={i}>
                    <td style={{fontWeight:600, color:'#58a6ff', fontFamily:'JetBrains Mono, monospace'}}>{p.proto}</td>
                    <td>{tot.toLocaleString()}</td>
                    <td style={{color:'#3fb950'}}>{(p.benign_count||0).toLocaleString()}</td>
                    <td style={{color:'#f85149'}}>{(p.suspicious_count||0).toLocaleString()}</td>
                    <td>
                      <span style={{
                        background: isHigh ? 'rgba(248,81,73,0.12)' : 'rgba(63,185,80,0.1)',
                        color: isHigh ? '#f85149' : '#3fb950',
                        borderRadius:6, padding:'2px 8px',
                        fontSize:'0.72rem', fontWeight:700
                      }}>{pct}%</span>
                    </td>
                    <td style={{color:'#8b949e', fontFamily:'JetBrains Mono, monospace', fontSize:'0.75rem'}}>
                      {fmtBytes(srcB)}
                    </td>
                    <td style={{color:'#8b949e', fontFamily:'JetBrains Mono, monospace', fontSize:'0.75rem'}}>
                      {fmtBytes(dstB)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  )
}
