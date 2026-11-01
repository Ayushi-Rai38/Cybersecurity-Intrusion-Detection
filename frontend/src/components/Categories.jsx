import { useState } from 'react'
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend,
  BarChart, Bar, XAxis, YAxis, CartesianGrid
} from 'recharts'
import { ShieldAlert, PieChart as PieIcon, BarChart2, Table2 } from 'lucide-react'
import { useFetch } from '../hooks/useFetch'
import { api } from '../api'
import s from './shared.module.css'

/* ── colour palette for categories ───────────────────────────── */
const CAT_COLORS = [
  '#58a6ff','#f85149','#3fb950','#d29922','#bc8cff',
  '#ff7c7c','#56d364','#e3b341','#79c0ff','#ffa657'
]

/* ── Custom tooltip for pie ───────────────────────────────────── */
function PieTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const { name, value } = payload[0]
  return (
    <div style={{
      background:'#161b22', border:'1px solid #30363d',
      borderRadius:8, padding:'8px 12px', fontSize:'0.8rem'
    }}>
      <div style={{color:'#c9d1d9', fontWeight:700}}>{name}</div>
      <div style={{color:'#58a6ff'}}>{value.toLocaleString()} connections</div>
    </div>
  )
}

/* ── Custom tooltip for bar ───────────────────────────────────── */
function BarTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background:'#161b22', border:'1px solid #30363d',
      borderRadius:8, padding:'8px 12px', fontSize:'0.8rem', maxWidth:220
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

/* ── Category summary card ────────────────────────────────────── */
function CatCard({ name, count, pct, color, rank }) {
  return (
    <div style={{
      background:'#161b22', border:`1px solid ${color}40`,
      borderRadius:12, padding:'1rem 1.1rem',
      borderLeft:`3px solid ${color}`
    }}>
      <div style={{display:'flex', justifyContent:'space-between', alignItems:'flex-start'}}>
        <div>
          <div style={{
            fontSize:'0.68rem', fontWeight:700, color:'#8b949e',
            textTransform:'uppercase', letterSpacing:'0.06em', marginBottom:4
          }}>#{rank} Category</div>
          <div style={{fontSize:'0.95rem', fontWeight:700, color:'#c9d1d9'}}>{name}</div>
        </div>
        <div style={{
          background:`${color}20`, color, borderRadius:8,
          padding:'2px 9px', fontSize:'0.72rem', fontWeight:700
        }}>{pct}%</div>
      </div>
      <div style={{marginTop:'0.7rem', background:'#30363d', borderRadius:4, height:4, overflow:'hidden'}}>
        <div style={{width:`${pct}%`, height:4, background:color, borderRadius:4}} />
      </div>
      <div style={{marginTop:6, fontSize:'0.72rem', color:'#8b949e'}}>
        {count.toLocaleString()} connections
      </div>
    </div>
  )
}

/* ── Detection rate table ─────────────────────────────────────── */
function DetectionTable({ data }) {
  if (!data?.length) return null
  return (
    <div className={s.tableWrap}>
      <table>
        <thead>
          <tr>
            <th>Attack Category</th>
            <th>Total</th>
            <th>Predicted Suspicious</th>
            <th>Predicted Benign</th>
            <th>Detection Rate</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => {
            const detRate = row.total > 0
              ? ((row.suspicious / row.total) * 100).toFixed(1)
              : '0.0'
            const isGood = parseFloat(detRate) >= 70
            return (
              <tr key={i}>
                <td style={{fontWeight:600, color:'#c9d1d9'}}>{row.category}</td>
                <td>{row.total.toLocaleString()}</td>
                <td style={{color:'#f85149'}}>{row.suspicious.toLocaleString()}</td>
                <td style={{color:'#3fb950'}}>{row.benign.toLocaleString()}</td>
                <td>
                  <span style={{
                    background: isGood ? 'rgba(63,185,80,0.15)' : 'rgba(248,81,73,0.12)',
                    color: isGood ? '#3fb950' : '#f85149',
                    borderRadius:6, padding:'2px 8px',
                    fontSize:'0.72rem', fontWeight:700
                  }}>{detRate}%</span>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/* ── Main component ───────────────────────────────────────────── */
export default function Categories() {
  const { data: allCats,  loading: l1, error: e1 } = useFetch(api.attackCategories)
  const { data: suspCats, loading: l2, error: e2 } = useFetch(api.attackCategoriesSuspicious)

  const loading = l1 || l2
  const error   = e1 || e2

  if (loading) return <div className={s.loading}>⏳ Loading category data…</div>
  if (error)   return <div className={s.error}>⚠ {error}</div>

  /* ── Prepare pie data (all categories) */
  const pieData = (allCats?.categories || []).map((c, i) => ({
    name: c.attack_cat,
    value: c.count,
    color: CAT_COLORS[i % CAT_COLORS.length]
  }))

  /* ── Prepare bar data (suspicious only) */
  const barSusp = (suspCats?.categories || []).map((c, i) => ({
    name: c.attack_cat,
    count: c.count,
    color: CAT_COLORS[i % CAT_COLORS.length]
  }))

  /* ── Top 5 cards */
  const total   = pieData.reduce((a, b) => a + b.value, 0)
  const top5    = [...pieData].sort((a, b) => b.value - a.value).slice(0, 5)

  /* ── Detection table: join allCats and suspCats by category */
  const suspMap = {}
  ;(suspCats?.categories || []).forEach(c => { suspMap[c.attack_cat] = c.count })
  const detData = (allCats?.categories || []).map(c => ({
    category:   c.attack_cat,
    total:      c.count,
    suspicious: suspMap[c.attack_cat] || 0,
    benign:     c.count - (suspMap[c.attack_cat] || 0)
  })).sort((a, b) => b.total - a.total)

  return (
    <div style={{display:'flex', flexDirection:'column', gap:'1.2rem'}}>

      {/* ── Header */}
      <div style={{display:'flex', alignItems:'center', gap:10}}>
        <ShieldAlert size={18} color="#d29922" />
        <div>
          <div style={{fontSize:'1.05rem', fontWeight:700, color:'#c9d1d9'}}>
            Attack Category Analysis
          </div>
          <div style={{fontSize:'0.75rem', color:'#8b949e', marginTop:2}}>
            Distribution and detection rates across all UNSW-NB15 attack types
          </div>
        </div>
      </div>

      {/* ── Top 5 category cards */}
      <div>
        <div className={s.sectionTitle}>
          <span className={s.sectionDot} />
          Top 5 Attack Categories
        </div>
        <div className={s.grid5}>
          {top5.map((c, i) => (
            <CatCard
              key={c.name}
              rank={i + 1}
              name={c.name}
              count={c.value}
              pct={total > 0 ? ((c.value / total) * 100).toFixed(1) : 0}
              color={c.color}
            />
          ))}
        </div>
      </div>

      {/* ── Pie + Suspicious bar */}
      <div className={s.grid2}>

        {/* Pie chart */}
        <div className={s.card}>
          <div className={s.sectionTitle}>
            <PieIcon size={13} />
            Category Distribution (All Connections)
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={pieData}
                dataKey="value"
                nameKey="name"
                cx="50%" cy="50%"
                outerRadius={95}
                innerRadius={45}
                paddingAngle={2}
              >
                {pieData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} stroke="none" />
                ))}
              </Pie>
              <Tooltip content={<PieTooltip />} />
              <Legend
                formatter={v => (
                  <span style={{fontSize:'0.72rem', color:'#8b949e'}}>{v}</span>
                )}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Suspicious-only bar */}
        <div className={s.card}>
          <div className={s.sectionTitle}>
            <BarChart2 size={13} />
            Predicted Suspicious by Category
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              data={barSusp}
              layout="vertical"
              margin={{top:4, right:20, left:8, bottom:4}}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#21262d" horizontal={false} />
              <XAxis type="number" tick={{fill:'#8b949e', fontSize:11}}
                tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v} />
              <YAxis type="category" dataKey="name"
                tick={{fill:'#8b949e', fontSize:11}} width={90} />
              <Tooltip content={<BarTooltip />} />
              <Bar dataKey="count" name="Suspicious" radius={[0,4,4,0]}>
                {barSusp.map((entry, i) => (
                  <Cell key={i} fill={CAT_COLORS[i % CAT_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Detection rate table */}
      <div className={s.card}>
        <div className={s.sectionTitle}>
          <Table2 size={13} />
          Detection Rate by Attack Category
        </div>
        <div className={s.info} style={{marginBottom:'1rem'}}>
          Shows how many connections in each category were correctly flagged as suspicious by the model.
          Normal/Benign connections are included as the baseline class.
        </div>
        <DetectionTable data={detData} />
      </div>

    </div>
  )
}
