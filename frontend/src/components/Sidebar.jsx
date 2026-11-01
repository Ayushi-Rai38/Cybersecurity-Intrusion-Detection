import { useState } from 'react'
import { useFetch } from '../hooks/useFetch.js'
import { api } from '../api.js'
import { Filter, Download, Shield } from 'lucide-react'
import styles from './Sidebar.module.css'

export default function Sidebar({ filters, setFilters }) {
  const { data: opts } = useFetch(api.filterOptions)
  const { data: kpis } = useFetch(api.kpis)

  const set = (key, val) => setFilters(f => ({ ...f, [key]: val }))

  return (
    <aside className={styles.sidebar}>
      {/* Brand */}
      <div className={styles.brand}>
        <Shield size={22} color="var(--accent)" />
        <div>
          <div className={styles.brandName}>CyberShield IDS</div>
          <div className={styles.brandSub}>UNSW-NB15 · Random Forest</div>
        </div>
      </div>

      {/* Filter section */}
      <div className={styles.section}>
        <div className={styles.sectionHdr}><Filter size={12} /> Filters</div>

        <label className={styles.label}>Predicted Status</label>
        <select className={styles.select} value={filters.status} onChange={e => set('status', e.target.value)}>
          <option>All</option>
          {opts?.statuses?.map(s => <option key={s}>{s}</option>)}
        </select>

        <label className={styles.label}>Risk Level</label>
        <select className={styles.select} value={filters.risk} onChange={e => set('risk', e.target.value)}>
          <option>All</option>
          {opts?.risk_levels?.map(r => <option key={r}>{r}</option>)}
        </select>

        <label className={styles.label}>Protocol</label>
        <select className={styles.select} value={filters.proto} onChange={e => set('proto', e.target.value)}>
          <option>All</option>
          {opts?.protocols?.slice(0, 30).map(p => <option key={p}>{p}</option>)}
        </select>
      </div>

      {/* Quick stats */}
      {kpis && (
        <div className={styles.section}>
          <div className={styles.sectionHdr}>📊 Dataset Stats</div>
          {[
            ['Total Records',  kpis.total_records?.toLocaleString()],
            ['Suspicious',     kpis.suspicious?.toLocaleString()],
            ['Benign',         kpis.benign?.toLocaleString()],
            ['Alert Rate',     `${kpis.alert_rate_pct}%`],
            ['Accuracy',       kpis.accuracy?.toFixed(4)],
            ['ROC-AUC',        kpis.roc_auc?.toFixed(4)],
          ].map(([k, v]) => (
            <div key={k} className={styles.statRow}>
              <span className={styles.statKey}>{k}</span>
              <span className={styles.statVal}>{v ?? '—'}</span>
            </div>
          ))}
        </div>
      )}

      <div className={styles.footer}>
        PySpark MLlib 4.x · Python 3.12 · Flask API
      </div>
    </aside>
  )
}
