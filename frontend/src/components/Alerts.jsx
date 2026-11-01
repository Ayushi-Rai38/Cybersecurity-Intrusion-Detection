import { useState, useEffect } from 'react'
import { api } from '../api.js'
import { useFetch } from '../hooks/useFetch.js'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer } from 'recharts'
import s from './shared.module.css'

const RISK_COLOR = { High:'#f85149', Medium:'#d29922', Low:'#3fb950' }
const STATUS_COLOR = { Suspicious:'#f85149', Benign:'#3fb950' }

function RiskBadge({ risk }) {
  const color = RISK_COLOR[risk] || '#8b949e'
  return (
    <span className={s.badge} style={{ color, borderColor: color, background: `${color}18` }}>
      {risk}
    </span>
  )
}

function StatusBadge({ status }) {
  const color = STATUS_COLOR[status] || '#8b949e'
  return (
    <span className={s.badge} style={{ color, borderColor: color, background: `${color}18` }}>
      {status}
    </span>
  )
}

export default function Alerts({ filters }) {
  const [page, setPage] = useState(1)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const { data: probDist } = useFetch(api.probDistribution)

  // Reset to page 1 whenever filters change
  useEffect(() => { setPage(1) }, [filters.status, filters.risk, filters.proto])

  useEffect(() => {
    setLoading(true)
    setError(null)
    api.predictions({
      status:   filters.status,
      risk:     filters.risk,
      proto:    filters.proto,
      page,
      per_page: 50,
    })
      .then(setResult)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [filters, page])

  // Risk summary from full suspicious set
  const { data: riskData } = useFetch(api.riskDistribution)

  const rows = result?.data ?? []
  const cols = ['risk_level','attack_prob','attack_cat','proto','state','sbytes','dbytes','dur','actual_status','predicted_status']
    .filter(c => rows.length > 0 && c in rows[0])

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'1.4rem' }}>

      {/* Risk summary pills */}
      <div className={s.sectionTitle}><span className={s.sectionDot}/>Risk Summary</div>
      <div className={s.grid4}>
        {['High','Medium','Low'].map(r => {
          const cnt = riskData?.find(x => x.risk === r)?.count ?? 0
          return (
            <div key={r} className={s.card} style={{ borderLeft:`3px solid ${RISK_COLOR[r]}`, textAlign:'center' }}>
              <div style={{ fontSize:'1.6rem', fontWeight:800, color:RISK_COLOR[r] }}>{cnt.toLocaleString()}</div>
              <div style={{ fontSize:'0.7rem', color:'var(--muted)', textTransform:'uppercase', letterSpacing:'0.05em', marginTop:'0.2rem' }}>{r} Risk Alerts</div>
            </div>
          )
        })}
        <div className={s.card} style={{ borderLeft:`3px solid #bc8cff`, textAlign:'center' }}>
          <div style={{ fontSize:'1.6rem', fontWeight:800, color:'#bc8cff' }}>{result?.total?.toLocaleString() ?? '—'}</div>
          <div style={{ fontSize:'0.7rem', color:'var(--muted)', textTransform:'uppercase', letterSpacing:'0.05em', marginTop:'0.2rem' }}>Filtered Records</div>
        </div>
      </div>

      {/* Probability histogram */}
      {probDist && (
        <>
          <div className={s.sectionTitle}><span className={s.sectionDot}/>Attack Probability Distribution (Suspicious)</div>
          <div className={s.card}>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={probDist} margin={{ left:0, right:10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#21262d" vertical={false} />
                <XAxis dataKey="lo" tick={{ fill:'#8b949e', fontSize:9 }} tickFormatter={v => v.toFixed(2)} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill:'#8b949e', fontSize:9 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background:'#1c2230', border:'1px solid #30363d', borderRadius:8, fontSize:11 }}
                  formatter={(v,n,p) => [v, `Range: ${p.payload.range}`]} />
                <Bar dataKey="count" radius={[3,3,0,0]}>
                  {probDist.map(e => (
                    <Cell key={e.range} fill={e.lo >= 0.8 ? '#f85149' : e.lo >= 0.5 ? '#d29922' : '#58a6ff'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ display:'flex', gap:'1.2rem', fontSize:'0.72rem', color:'var(--muted)', marginTop:'0.4rem' }}>
              <span style={{color:'#f85149'}}>■ High ≥ 0.80</span>
              <span style={{color:'#d29922'}}>■ Medium 0.50–0.80</span>
              <span style={{color:'#58a6ff'}}>■ Low &lt; 0.50</span>
            </div>
          </div>
        </>
      )}

      {/* Predictions table */}
      <div className={s.sectionTitle}><span className={s.sectionDot}/>Filtered Predictions (page {page} of {result?.pages ?? 1})</div>

      {loading && <div className={s.loading}>Loading predictions…</div>}
      {error   && <div className={s.error}>{error}</div>}

      {!loading && !error && (
        <>
          <div className={s.tableWrap}>
            <table>
              <thead>
                <tr>{cols.map(c => <th key={c}>{c.replace(/_/g,' ')}</th>)}</tr>
              </thead>
              <tbody>
                {rows.length === 0
                  ? <tr><td colSpan={cols.length} style={{ textAlign:'center', color:'var(--muted)', padding:'2rem' }}>No records match the current filters.</td></tr>
                  : rows.map((row, i) => (
                    <tr key={i}>
                      {cols.map(c => (
                        <td key={c}>
                          {c === 'risk_level'        ? <RiskBadge   risk={row[c]}   /> :
                           c === 'predicted_status'  ? <StatusBadge status={row[c]} /> :
                           c === 'actual_status'     ? <StatusBadge status={row[c]} /> :
                           c === 'attack_prob'       ? (typeof row[c] === 'number' ? row[c].toFixed(4) : row[c]) :
                           row[c] ?? '—'}
                        </td>
                      ))}
                    </tr>
                  ))
                }
              </tbody>
            </table>
          </div>

          <div className={s.pagination}>
            <span>Showing {rows.length} of {result?.total?.toLocaleString()} records</span>
            <div className={s.pgBtns}>
              <button className={s.pgBtn} disabled={page <= 1} onClick={() => setPage(1)}>«</button>
              <button className={s.pgBtn} disabled={page <= 1} onClick={() => setPage(p => p - 1)}>‹ Prev</button>
              <span style={{ padding:'4px 10px', fontSize:'0.78rem', color:'var(--muted)' }}>
                {page} / {result?.pages ?? 1}
              </span>
              <button className={s.pgBtn} disabled={page >= (result?.pages ?? 1)} onClick={() => setPage(p => p + 1)}>Next ›</button>
              <button className={s.pgBtn} disabled={page >= (result?.pages ?? 1)} onClick={() => setPage(result?.pages ?? 1)}>»</button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
