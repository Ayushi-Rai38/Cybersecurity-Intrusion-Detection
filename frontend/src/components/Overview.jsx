import { useFetch } from '../hooks/useFetch.js'
import { api } from '../api.js'
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend,
} from 'recharts'
import s from './shared.module.css'

const C = { green:'#3fb950', red:'#f85149', orange:'#d29922', blue:'#58a6ff', purple:'#bc8cff', teal:'#39d0d8' }

function KpiCard({ icon, value, label, sub, gradient }) {
  return (
    <div className={s.kpiCard} style={{ background: gradient }}>
      <div className={s.kpiIcon}>{icon}</div>
      <div>
        <div className={s.kpiValue}>{value}</div>
        <div className={s.kpiLabel}>{label}</div>
        {sub && <div className={s.kpiSub}>{sub}</div>}
      </div>
    </div>
  )
}

function MetricPill({ value, label, color }) {
  return (
    <div className={s.pill} style={{ '--pillColor': color }}>
      <div className={s.pillVal}>{typeof value === 'number' ? value.toFixed(4) : value}</div>
      <div className={s.pillLabel}>{label}</div>
      <div className={s.pillBar}>
        <div className={s.pillBarFill} style={{ width: `${(value * 100).toFixed(1)}%` }} />
      </div>
    </div>
  )
}

function SectionTitle({ children }) {
  return (
    <div className={s.sectionTitle}>
      <span className={s.sectionDot} />
      {children}
    </div>
  )
}

export default function Overview({ filters }) {
  const { data: kpis,   loading: lk } = useFetch(api.kpis)
  const { data: metrics,loading: lm } = useFetch(api.metrics)
  const { data: split }               = useFetch(api.predictionSplit)
  const { data: risk }                = useFetch(api.riskDistribution)
  const { data: cats }                = useFetch(api.attackCategories)

  if (lk || lm) return <div className={s.loading}>Loading dashboard data…</div>
  if (!kpis)    return <div className={s.error}>API not reachable. Start the Flask server: <code>python api.py</code></div>

  const alertColor = kpis.alert_rate_pct > 40 ? C.red : C.green

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'1.4rem' }}>

      {/* KPI row */}
      <SectionTitle>Key Performance Indicators</SectionTitle>
      <div className={s.grid4}>
        <KpiCard icon="📦" value={kpis.total_records?.toLocaleString()} label="Total Test Records"
          sub={`${Math.round(kpis.total_records/1000)}k connections analysed`}
          gradient="linear-gradient(135deg,#1f6feb,#388bfd)" />
        <KpiCard icon="🚨" value={kpis.suspicious?.toLocaleString()} label="Suspicious Alerts"
          sub={`${kpis.alert_rate_pct}% of total traffic`}
          gradient="linear-gradient(135deg,#da3633,#f85149)" />
        <KpiCard icon="✅" value={kpis.benign?.toLocaleString()} label="Benign Records"
          sub={`${(100 - kpis.alert_rate_pct).toFixed(1)}% clean`}
          gradient="linear-gradient(135deg,#1a7f37,#3fb950)" />
        <KpiCard icon="🌐" value={kpis.total_bytes_gb > 0 ? `${kpis.total_bytes_gb} GB` : 'N/A'} label="Total Traffic Volume"
          sub="Source + destination bytes"
          gradient="linear-gradient(135deg,#0d7377,#39d0d8)" />
      </div>

      {/* Model metrics */}
      <SectionTitle>Model Performance</SectionTitle>
      <div className={s.grid5}>
        <MetricPill value={metrics?.accuracy ?? 0}         label="Accuracy"        color={C.blue}   />
        <MetricPill value={metrics?.attack_precision ?? 0} label="Attack Precision" color={C.red}    />
        <MetricPill value={metrics?.attack_recall ?? 0}    label="Attack Recall"    color={C.green}  />
        <MetricPill value={metrics?.attack_f1 ?? 0}        label="Attack F1-Score"  color={C.orange} />
        <MetricPill value={metrics?.roc_auc ?? 0}          label="ROC-AUC"          color={C.purple} />
      </div>

      {/* Charts row */}
      <SectionTitle>Traffic Snapshot</SectionTitle>
      <div className={s.grid3}>

        {/* Prediction donut */}
        <div className={s.card}>
          <div className={s.sectionTitle} style={{marginBottom:'0.6rem'}}>
            <span className={s.sectionDot}/> Prediction Split
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={split ?? []} cx="50%" cy="50%" innerRadius={55} outerRadius={80}
                dataKey="value" nameKey="name" paddingAngle={3}>
                {(split ?? []).map(e => (
                  <Cell key={e.name} fill={e.name === 'Benign' ? C.green : C.red} stroke="none" />
                ))}
              </Pie>
              <Tooltip contentStyle={{ background:'#1c2230', border:'1px solid #30363d', borderRadius:8, fontSize:12 }}
                formatter={(v,n) => [v?.toLocaleString(), n]} />
            </PieChart>
          </ResponsiveContainer>
          <div style={{ display:'flex', justifyContent:'center', gap:'1rem', fontSize:'0.75rem', color:'var(--muted)', marginTop:'0.3rem' }}>
            <span style={{ color:C.green }}>● Benign</span>
            <span style={{ color:C.red   }}>● Suspicious</span>
          </div>
        </div>

        {/* Risk distribution */}
        <div className={s.card}>
          <div className={s.sectionTitle} style={{marginBottom:'0.6rem'}}>
            <span className={s.sectionDot}/> Risk Level Distribution
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={risk ?? []} layout="vertical" margin={{ left:10, right:30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#21262d" horizontal={false} />
              <XAxis type="number" tick={{ fill:'#8b949e', fontSize:10 }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="risk" tick={{ fill:'#e6edf3', fontSize:11 }} axisLine={false} tickLine={false} width={55} />
              <Tooltip contentStyle={{ background:'#1c2230', border:'1px solid #30363d', borderRadius:8, fontSize:12 }} />
              <Bar dataKey="count" radius={[0,5,5,0]}>
                {(risk ?? []).map(e => (
                  <Cell key={e.risk} fill={e.risk==='High'?C.red:e.risk==='Medium'?C.orange:C.green} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Top categories */}
        <div className={s.card}>
          <div className={s.sectionTitle} style={{marginBottom:'0.6rem'}}>
            <span className={s.sectionDot}/> Top Attack Categories
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={(cats ?? []).slice(0,8)} layout="vertical" margin={{ left:10, right:30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#21262d" horizontal={false} />
              <XAxis type="number" tick={{ fill:'#8b949e', fontSize:10 }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="category" tick={{ fill:'#e6edf3', fontSize:10 }} axisLine={false} tickLine={false} width={80} />
              <Tooltip contentStyle={{ background:'#1c2230', border:'1px solid #30363d', borderRadius:8, fontSize:12 }} />
              <Bar dataKey="count" fill={C.blue} radius={[0,5,5,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

      </div>
    </div>
  )
}
