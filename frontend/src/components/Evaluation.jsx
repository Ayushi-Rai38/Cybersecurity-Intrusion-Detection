import { useFetch } from '../hooks/useFetch.js'
import { api } from '../api.js'
import s from './shared.module.css'

export default function Evaluation() {
  const { data: m,   loading: lm } = useFetch(api.metrics)
  const { data: avp, loading: la } = useFetch(api.actualVsPredicted)

  if (lm || la) return <div className={s.loading}>Loading evaluation data…</div>
  if (!m)       return <div className={s.error}>No metrics found. Run <code>python main.py</code> first.</div>

  const cm  = m.confusion_matrix ?? {}
  const tn  = cm.TN ?? 0
  const fp  = cm.FP ?? 0
  const fn  = cm.FN ?? 0
  const tp  = cm.TP ?? 0

  const metrics = [
    { label:'Accuracy',         value: m.accuracy,         color:'#58a6ff', desc:'Fraction of all predictions that are correct' },
    { label:'Attack Precision',  value: m.attack_precision, color:'#f85149', desc:'Of all flagged alerts, how many are real attacks' },
    { label:'Attack Recall',     value: m.attack_recall,    color:'#3fb950', desc:'Of all real attacks, how many were detected' },
    { label:'Attack F1-Score',   value: m.attack_f1,        color:'#d29922', desc:'Harmonic mean of precision and recall' },
    { label:'ROC-AUC',           value: m.roc_auc,          color:'#bc8cff', desc:'Model discrimination across all thresholds' },
  ]

  const cmCells = [
    { label:'TN', full:'True Negative',  val:tn, color:'#3fb950', desc:'Normal → correctly identified' },
    { label:'FP', full:'False Positive', val:fp, color:'#f85149', desc:'Normal → wrongly flagged as attack' },
    { label:'FN', full:'False Negative', val:fn, color:'#d29922', desc:'Attack → missed (most dangerous)' },
    { label:'TP', full:'True Positive',  val:tp, color:'#58a6ff', desc:'Attack → correctly detected' },
  ]

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:'1.4rem' }}>

      {/* Metric pills */}
      <div className={s.sectionTitle}><span className={s.sectionDot}/>Evaluation Metrics</div>
      <div className={s.grid5}>
        {metrics.map(({ label, value, color, desc }) => (
          <div key={label} className={s.pill} style={{ '--pillColor': color }}>
            <div className={s.pillVal}>{(value ?? 0).toFixed(4)}</div>
            <div className={s.pillLabel}>{label}</div>
            <div className={s.pillBar}>
              <div className={s.pillBarFill} style={{ width:`${((value??0)*100).toFixed(1)}%` }} />
            </div>
            <div style={{ fontSize:'0.65rem', color:'var(--muted)', marginTop:'0.4rem', lineHeight:1.4 }}>{desc}</div>
          </div>
        ))}
      </div>

      {/* Confusion matrix + breakdown */}
      <div className={s.sectionTitle}><span className={s.sectionDot}/>Confusion Matrix</div>
      <div className={s.grid2}>

        {/* Visual 2×2 matrix */}
        <div className={s.card}>
          <div style={{ fontSize:'0.72rem', color:'var(--muted)', marginBottom:'0.8rem', textAlign:'center', textTransform:'uppercase', letterSpacing:'0.06em' }}>
            Predicted →
          </div>
          <div style={{ display:'grid', gridTemplateColumns:'auto 1fr 1fr', gap:6, alignItems:'center' }}>
            <div />
            {['Pred: Normal','Pred: Attack'].map(h => (
              <div key={h} style={{ textAlign:'center', fontSize:'0.7rem', color:'var(--accent)', fontWeight:700, textTransform:'uppercase', letterSpacing:'0.05em' }}>{h}</div>
            ))}
            {[['Actual: Normal', tn, fp], ['Actual: Attack', fn, tp]].map(([label, v1, v2]) => (
              <>
                <div key={label} style={{ fontSize:'0.7rem', color:'var(--muted)', fontWeight:700, writingMode:'horizontal-tb', whiteSpace:'nowrap' }}>{label}</div>
                {[v1, v2].map((val, j) => (
                  <div key={j} style={{
                    background: j === 0 ? 'rgba(63,185,80,0.12)' : 'rgba(248,81,73,0.12)',
                    border: `1px solid ${j === 0 ? 'rgba(63,185,80,0.3)' : 'rgba(248,81,73,0.3)'}`,
                    borderRadius: 10, padding: '1rem', textAlign:'center',
                  }}>
                    <div style={{ fontSize:'1.4rem', fontWeight:800, color: j===0 ? '#3fb950':'#f85149' }}>{val.toLocaleString()}</div>
                    <div style={{ fontSize:'0.65rem', color:'var(--muted)', marginTop:'0.2rem' }}>
                      {label.includes('Normal') ? (j===0 ? 'TN':'FP') : (j===0 ? 'FN':'TP')}
                    </div>
                  </div>
                ))}
              </>
            ))}
          </div>
        </div>

        {/* CM breakdown cards */}
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'0.7rem' }}>
          {cmCells.map(({ label, full, val, color, desc }) => (
            <div key={label} className={s.card} style={{ borderTop:`3px solid ${color}` }}>
              <div style={{ display:'flex', alignItems:'baseline', gap:6 }}>
                <span style={{ fontSize:'0.72rem', fontWeight:800, color, background:`${color}18`, border:`1px solid ${color}40`, borderRadius:4, padding:'1px 6px' }}>{label}</span>
                <span style={{ fontSize:'0.7rem', color:'var(--muted)' }}>{full}</span>
              </div>
              <div style={{ fontSize:'1.5rem', fontWeight:800, color, margin:'0.4rem 0 0.1rem' }}>{val.toLocaleString()}</div>
              <div style={{ fontSize:'0.68rem', color:'var(--muted)', lineHeight:1.4 }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Actual vs Predicted */}
      {avp && (
        <>
          <div className={s.sectionTitle}><span className={s.sectionDot}/>Actual vs Predicted Counts</div>
          <div className={s.grid4}>
            {[
              { label:'Actual Normal',   val:avp.actual_normal,   color:'#3fb950' },
              { label:'Actual Attack',   val:avp.actual_attack,   color:'#f85149' },
              { label:'Predicted Normal',val:avp.pred_benign,     color:'#58a6ff' },
              { label:'Predicted Attack',val:avp.pred_suspicious, color:'#d29922' },
            ].map(({ label, val, color }) => (
              <div key={label} className={s.card} style={{ textAlign:'center', borderTop:`3px solid ${color}` }}>
                <div style={{ fontSize:'1.5rem', fontWeight:800, color }}>{val?.toLocaleString() ?? '—'}</div>
                <div style={{ fontSize:'0.7rem', color:'var(--muted)', textTransform:'uppercase', letterSpacing:'0.05em', marginTop:'0.2rem' }}>{label}</div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Class distribution */}
      {m.class_distribution && (
        <>
          <div className={s.sectionTitle}><span className={s.sectionDot}/>Training Class Distribution</div>
          <div className={s.info}>
            The model was trained with class weights to handle imbalance.
            Normal (class 0): <strong>{m.class_distribution.training_normal?.toLocaleString()}</strong> &nbsp;·&nbsp;
            Attack (class 1): <strong>{m.class_distribution.training_attack?.toLocaleString()}</strong>
          </div>
        </>
      )}
    </div>
  )
}
