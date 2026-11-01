import { useFetch } from '../hooks/useFetch.js'
import { api } from '../api.js'
import { CheckCircle, AlertCircle, Loader } from 'lucide-react'
import styles from './StatusBanner.module.css'

export default function StatusBanner() {
  const { data, loading } = useFetch(api.health)

  if (loading) return (
    <div className={styles.banner} data-state="loading">
      <Loader size={13} className={styles.spin} />
      <span>Connecting…</span>
    </div>
  )

  const ready = data?.status === 'ready'
  return (
    <div className={styles.banner} data-state={ready ? 'ready' : 'warn'}>
      {ready
        ? <><CheckCircle size={13} /><span>Model outputs loaded</span></>
        : <><AlertCircle size={13} /><span>Run <code>python main.py</code> first</span></>
      }
    </div>
  )
}
