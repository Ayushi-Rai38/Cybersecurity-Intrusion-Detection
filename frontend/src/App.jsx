import { useState } from 'react'
import { Shield, LayoutDashboard, AlertTriangle, Target, Layers, Radio } from 'lucide-react'
import Sidebar   from './components/Sidebar.jsx'
import Overview  from './components/Overview.jsx'
import Alerts    from './components/Alerts.jsx'
import Evaluation from './components/Evaluation.jsx'
import Categories from './components/Categories.jsx'
import Traffic   from './components/Traffic.jsx'
import StatusBanner from './components/StatusBanner.jsx'
import styles from './App.module.css'

const TABS = [
  { id: 'overview',    label: 'Overview',    icon: LayoutDashboard },
  { id: 'alerts',      label: 'Alerts',      icon: AlertTriangle   },
  { id: 'evaluation',  label: 'Evaluation',  icon: Target          },
  { id: 'categories',  label: 'Categories',  icon: Layers          },
  { id: 'traffic',     label: 'Traffic',     icon: Radio           },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('overview')
  const [filters, setFilters] = useState({ status: 'All', risk: 'All', proto: 'All' })

  return (
    <div className={styles.layout}>
      <Sidebar filters={filters} setFilters={setFilters} />

      <div className={styles.main}>
        {/* Header */}
        <header className={styles.header}>
          <div className={styles.headerLeft}>
            <Shield size={26} color="var(--accent)" />
            <div>
              <h1 className={styles.title}>Cybersecurity Intrusion Detection System</h1>
              <p className={styles.subtitle}>UNSW-NB15 · PySpark MLlib · Random Forest Classifier</p>
            </div>
          </div>
          <StatusBanner />
        </header>

        {/* Tab bar */}
        <nav className={styles.tabBar}>
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`${styles.tab} ${activeTab === id ? styles.tabActive : ''}`}
              onClick={() => setActiveTab(id)}
            >
              <Icon size={15} />
              {label}
            </button>
          ))}
        </nav>

        {/* Page content */}
        <div className={styles.content}>
          {activeTab === 'overview'   && <Overview   filters={filters} />}
          {activeTab === 'alerts'     && <Alerts     filters={filters} />}
          {activeTab === 'evaluation' && <Evaluation />}
          {activeTab === 'categories' && <Categories />}
          {activeTab === 'traffic'    && <Traffic    />}
        </div>

        <footer className={styles.footer}>
          🛡️ CyberShield IDS · UNSW-NB15 Dataset · PySpark MLlib 4.x · Random Forest ·{' '}
          <span className={styles.footerNote}>
            No timestamp or sport/dsport columns in UNSW-NB15 CSVs · is_ftp_login is FTP-specific only
          </span>
        </footer>
      </div>
    </div>
  )
}
