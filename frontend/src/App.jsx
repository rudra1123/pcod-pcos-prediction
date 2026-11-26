import { useEffect, useMemo, useState } from 'react'
import './App.css'
import {
  FALLBACK_FEATURES,
  SAMPLE_NO_PCOS,
  SAMPLE_PCOS,
  SECTIONS,
  defaultsFromFeatures,
} from './features'

// API base: empty string in dev (Vite proxies /api -> http://localhost:8080).
// Set VITE_API_URL (e.g. http://localhost:8080) to call the backend directly.
const API_BASE = import.meta.env.VITE_API_URL || ''
const BACKEND_DOWN_MSG =
  'Backend not running — start it on port 8080 (run the Spring Boot app in ../backend, then refresh this page).'
const DISCLAIMER = 'Screening aid, not a diagnosis — consult a doctor.'
const N_FEATURES = 41

// Minimum fields the form itself requires before it will send a prediction.
// This list mirrors the backend's required-field list exactly (the backend
// returns 400 listing any missing ones, and that 400 is shown verbatim).
// All 11 are non-invasive — no lab or ultrasound value is required.
const REQUIRED_FIELDS = ['age', 'bmi', 'cycle', 'cycleLength', 'weightGain', 'hairGrowth', 'skinDarkening', 'hairLoss', 'pimples', 'fastFood', 'regularExercise']

// Warn (do not block) when this many fields were imputed: the result then
// rests mostly on training-median values, not on the patient's own data.
const HIGH_IMPUTATION = 21

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return null
  const n = Number(value)
  return `${(n <= 1 ? n * 100 : n).toFixed(1)}%`
}

function fmtNum(value) {
  if (value === null || value === undefined || value === '') return '—'
  const n = Number(value)
  if (Number.isNaN(n)) return String(value)
  return Number.isInteger(n) ? String(n) : String(Math.round(n * 100) / 100)
}

function labelOf(features, key) {
  const f = features.find((x) => x.key === key)
  return f ? f.label : key
}

// --- Backend payload normalisation ---------------------------------------
// /api/models may return either a plain array (the original contract) or a
// full metrics object { models: [...], featureImportance: {...},
// confusionMatrix / rocAuc / thresholds / ... } (the phase-2 contract).
// Everything below renders ONLY when the backend actually provides it;
// nothing is invented when a field is absent.

function asArray(v) {
  return Array.isArray(v) ? v : []
}

function modelAuc(m) {
  const v = m?.rocAuc ?? m?.rocAUC ?? m?.auc ?? m?.roc_auc
  return v === undefined || v === null ? null : Number(v)
}

function modelThreshold(m) {
  const v = m?.threshold ?? m?.decisionThreshold ?? m?.decision_threshold
  return v === undefined || v === null ? null : Number(v)
}

function isScreeningModel(m) {
  if (!m) return false
  return (
    m.mode === 'screening' ||
    m.screening === true ||
    /screening/i.test(String(m.name || ''))
  )
}

// Accepts {tp,fp,fn,tn} (any capitalisation / long names) or a 2x2 array
// [[tn, fp], [fn, tp]] (scikit-learn layout). Returns null when unusable.
function normalizeConfusion(cm) {
  if (!cm) return null
  if (Array.isArray(cm) && cm.length === 2 && Array.isArray(cm[0]) && Array.isArray(cm[1])) {
    const [[tn, fp], [fn, tp]] = cm
    if ([tn, fp, fn, tp].every((v) => Number.isFinite(Number(v)))) {
      return { tp: Number(tp), fp: Number(fp), fn: Number(fn), tn: Number(tn) }
    }
    return null
  }
  if (typeof cm === 'object') {
    const pick = (...keys) => {
      for (const k of keys) if (cm[k] !== undefined && cm[k] !== null) return Number(cm[k])
      return null
    }
    const tp = pick('tp', 'TP', 'truePositive', 'true_positive')
    const fp = pick('fp', 'FP', 'falsePositive', 'false_positive')
    const fn = pick('fn', 'FN', 'falseNegative', 'false_negative')
    const tn = pick('tn', 'TN', 'trueNegative', 'true_negative')
    if ([tp, fp, fn, tn].every((v) => v !== null && Number.isFinite(v))) {
      return { tp, fp, fn, tn }
    }
  }
  return null
}

function modelConfusion(m) {
  return normalizeConfusion(m?.confusionMatrix ?? m?.confusion_matrix ?? m?.confusion)
}

// Accepts [{fpr, tpr}], [[fpr, tpr]], or {fpr: [...], tpr: [...]}.
// Returns sorted [{fpr, tpr}] or null. Curve points are NEVER fabricated:
// without backend points there is no curve, only the AUC value (if given).
function normalizeRocPoints(raw) {
  if (!raw) return null
  let pts = null
  if (Array.isArray(raw)) {
    pts = raw
      .map((p) => {
        if (Array.isArray(p) && p.length >= 2) return { fpr: Number(p[0]), tpr: Number(p[1]) }
        if (p && typeof p === 'object') {
          return {
            fpr: Number(p.fpr ?? p.falsePositiveRate ?? p.x),
            tpr: Number(p.tpr ?? p.truePositiveRate ?? p.y),
          }
        }
        return null
      })
      .filter((p) => p && Number.isFinite(p.fpr) && Number.isFinite(p.tpr))
  } else if (raw && typeof raw === 'object' && Array.isArray(raw.fpr) && Array.isArray(raw.tpr)) {
    pts = raw.fpr
      .map((fpr, i) => ({ fpr: Number(fpr), tpr: Number(raw.tpr[i]) }))
      .filter((p) => Number.isFinite(p.fpr) && Number.isFinite(p.tpr))
  }
  if (!pts || pts.length < 2) return null
  return [...pts].sort((a, b) => a.fpr - b.fpr)
}

function modelRocPoints(m) {
  return normalizeRocPoints(m?.rocCurve ?? m?.rocPoints ?? m?.roc_curve ?? m?.roc)
}

// Feature importance: a map {key: score} or an array of
// {feature|key, importance|score|value}. Returns sorted [{key, score}] desc.
function normalizeImportance(raw) {
  if (!raw) return []
  let entries = []
  if (Array.isArray(raw)) {
    entries = raw
      .map((e) => {
        if (!e || typeof e !== 'object') return null
        const key = e.feature ?? e.key ?? e.name
        const score = Number(e.importance ?? e.score ?? e.value)
        return key && Number.isFinite(score) ? { key, score } : null
      })
      .filter(Boolean)
  } else if (typeof raw === 'object') {
    entries = Object.entries(raw)
      .map(([key, score]) => ({ key, score: Number(score) }))
      .filter((e) => Number.isFinite(e.score))
  }
  return entries.sort((a, b) => b.score - a.score)
}

function extractModelsPayload(data) {
  if (Array.isArray(data)) return { models: data, extra: {} }
  if (data && typeof data === 'object') {
    const models = asArray(data.models)
    const extra = { ...data }
    delete extra.models
    return { models, extra }
  }
  return { models: [], extra: {} }
}

// ROC curve drawn with plain SVG from backend-supplied points only.
function RocChart({ points, auc, modelName }) {
  const W = 360
  const H = 300
  const PAD = 36
  const x = (fpr) => PAD + fpr * (W - 2 * PAD)
  const y = (tpr) => H - PAD - tpr * (H - 2 * PAD)
  const line = points.map((p) => `${x(p.fpr).toFixed(1)},${y(p.tpr).toFixed(1)}`).join(' ')
  return (
    <figure className="roc-figure">
      <figcaption>
        ROC curve — {modelName}
        {auc !== null && auc !== undefined ? ` (ROC-AUC ${Number(auc).toFixed(3)})` : ''}
      </figcaption>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="roc-svg"
        role="img"
        aria-label={`ROC curve for ${modelName}${auc !== null && auc !== undefined ? `, area under the curve ${Number(auc).toFixed(3)}` : ''}`}
      >
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={PAD} stroke="#b9c4ce" strokeDasharray="5 4" />
        <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#5a6b7a" />
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#5a6b7a" />
        <text x={PAD} y={H - 10} fontSize="11" fill="#5a6b7a">False Positive Rate →</text>
        <text x={12} y={H / 2} fontSize="11" fill="#5a6b7a" transform={`rotate(-90 12 ${H / 2})`}>
          True Positive Rate
        </text>
        <text x={W - PAD} y={H - PAD + 16} fontSize="11" fill="#5a6b7a" textAnchor="end">1</text>
        <text x={PAD} y={PAD - 6} fontSize="11" fill="#5a6b7a">1</text>
        <polyline points={line} fill="none" stroke="#123a5c" strokeWidth="2.5" />
        {points.map((p, i) => (
          <circle key={i} cx={x(p.fpr)} cy={y(p.tpr)} r="2.5" fill="#123a5c" />
        ))}
      </svg>
      <p className="muted roc-note">
        Diagonal dashed line = chance (AUC 0.5). Points come from the backend;
        no curve is drawn when the backend supplies no ROC points.
      </p>
    </figure>
  )
}

function ConfusionMatrix({ cm, modelName }) {
  const total = cm.tp + cm.fp + cm.fn + cm.tn
  const accuracy = total > 0 ? ((cm.tp + cm.tn) / total) * 100 : null
  return (
    <figure className="cm-figure">
      <figcaption>Confusion matrix — {modelName} (test set, n = {total})</figcaption>
      <table className="cm-table">
        <thead>
          <tr>
            <th scope="col">Actual \ Predicted</th>
            <th scope="col">Predicted: No PCOS</th>
            <th scope="col">Predicted: PCOS</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">Actual: No PCOS</th>
            <td className="cm-tn">TN: <strong>{cm.tn}</strong><span className="cm-sub">correct</span></td>
            <td className="cm-fp">FP: <strong>{cm.fp}</strong><span className="cm-sub">false alarm</span></td>
          </tr>
          <tr>
            <th scope="row">Actual: PCOS</th>
            <td className="cm-fn">FN: <strong>{cm.fn}</strong><span className="cm-sub">missed case</span></td>
            <td className="cm-tp">TP: <strong>{cm.tp}</strong><span className="cm-sub">correct</span></td>
          </tr>
        </tbody>
      </table>
      {accuracy !== null && (
        <p className="muted">Accuracy from this matrix: {accuracy.toFixed(1)}% ({cm.tp + cm.tn} of {total} correct).</p>
      )}
    </figure>
  )
}

// Animated SVG progress ring for the prediction probability.
// Pure CSS transition on stroke-dashoffset; the ring sweeps from 0 to the
// final value on mount. No numbers are changed — only how they are drawn.
function ProgressRing({ fraction, label, tone }) {
  const [shown, setShown] = useState(0)
  const target = Number.isFinite(fraction) ? Math.max(0, Math.min(1, fraction)) : 0
  useEffect(() => {
    const raf = requestAnimationFrame(() => setShown(target))
    return () => cancelAnimationFrame(raf)
  }, [target])
  const R = 62
  const C = 2 * Math.PI * R
  const offset = C * (1 - shown)
  return (
    <div className="ring-wrap" role="img" aria-label={label}>
      <svg viewBox="0 0 160 160" className="ring-svg" aria-hidden="true">
        <circle cx="80" cy="80" r={R} className="ring-track" />
        <circle
          cx="80" cy="80" r={R}
          className={`ring-fill ring-${tone}`}
          strokeDasharray={C}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="ring-center">
        <span className="ring-value">{label}</span>
        <span className="ring-caption">probability</span>
      </div>
    </div>
  )
}

// Dashboard stat card. Renders nothing when the value is absent —
// no numbers are invented when the backend does not provide them.
function StatCard({ label, value, sub, delay }) {
  if (value === null || value === undefined) return null
  return (
    <div className="stat-card animate-in" style={{ animationDelay: `${delay}ms` }}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  )
}

export default function App() {
  const [features, setFeatures] = useState(FALLBACK_FEATURES)
  const [featuresFromApi, setFeaturesFromApi] = useState(false)
  const [values, setValues] = useState(() => ({ ...SAMPLE_PCOS.values }))
  // Optional patient details (name + ID). Shown on the result card and the
  // printout, sent with the predict request and stored with the history
  // record — never sent to the model.
  const [patientName, setPatientName] = useState('')
  const [patientId, setPatientId] = useState('')
  const [submittedPatient, setSubmittedPatient] = useState(null)
  const [backendStatus, setBackendStatus] = useState('checking') // checking | online | offline
  const [models, setModels] = useState(null)
  const [modelsError, setModelsError] = useState('')
  // Full /api/models payload when the backend returns an object (not just a
  // list): featureImportance, per-model confusion/ROC/AUC, thresholds, etc.
  const [metricsExtra, setMetricsExtra] = useState(null)
  const [result, setResult] = useState(null)
  const [resultAt, setResultAt] = useState(null)
  const [submittedValues, setSubmittedValues] = useState(null)
  const [submittedImputedKeys, setSubmittedImputedKeys] = useState([])
  const [predicting, setPredicting] = useState(false)
  const [error, setError] = useState('')
  const [fieldErrors, setFieldErrors] = useState({})
  const [loadedSample, setLoadedSample] = useState(SAMPLE_PCOS.name)
  const [cmModelName, setCmModelName] = useState('')
  // Prediction mode for the form: "full" (41-feature deployed model) or
  // "screening" (23 non-invasive features). Sent as ?mode= to /api/predict.
  const [predictMode, setPredictMode] = useState('full')
  // Prediction history (admin-only): the admin key lives in component
  // state only — it is never written to localStorage or any other storage.
  const [adminKey, setAdminKey] = useState('')
  const [history, setHistory] = useState(null)
  const [historyError, setHistoryError] = useState('')
  const [historyLoading, setHistoryLoading] = useState(false)
  // Per-row verified-outcome selection for the history table (id -> "0" | "1").
  const [verifyValues, setVerifyValues] = useState({})
  const [verifyBusy, setVerifyBusy] = useState({})

  const featureByKey = useMemo(() => {
    const map = {}
    for (const f of features) map[f.key] = f
    return map
  }, [features])

  // On load: check backend health, then fetch the feature list and models.
  // Any fetch failure falls back to the hardcoded copy — the page never
  // goes blank because the backend is down.
  useEffect(() => {
    let cancelled = false
    let healthTimer = null
    let healthTries = 0

    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE}/api/health`)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        if (cancelled) return
        setBackendStatus('online')
        // Backend just came up (a one-click run opens the page before the
        // backend finishes starting) — load the live data now.
        loadFeatures()
        loadModels()
        loadMetrics()
      } catch {
        if (cancelled) return
        setBackendStatus('offline')
        // Retry while the backend is still starting (one-click run):
        // every 3s, up to 20 tries (~60s).
        healthTries += 1
        if (healthTries < 20) {
          healthTimer = setTimeout(checkHealth, 3000)
        }
      }
    }

    async function loadFeatures() {
      try {
        const res = await fetch(`${API_BASE}/api/features`)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        const list = Array.isArray(data) ? data : data.features
        if (Array.isArray(list) && list.length > 0 && list.every((f) => f.key)) {
          if (cancelled) return
          // /api/features does not send units; keep the fallback unit when
          // the backend gives none, so units/ranges still show.
          const merged = list.map((f) => {
            const fb = FALLBACK_FEATURES.find((x) => x.key === f.key)
            return {
              ...f,
              unit: f.unit ?? fb?.unit ?? '',
              min: f.min ?? fb?.min,
              max: f.max ?? fb?.max,
              default: f.default ?? fb?.default ?? '',
              type: f.type ?? fb?.type ?? 'number',
            }
          })
          setFeatures(merged)
          setFeaturesFromApi(true)
          // Keep the prefilled PCOS sample; only fill keys the API adds.
          setValues((prev) => {
            const next = { ...prev }
            for (const f of merged) if (next[f.key] === undefined) next[f.key] = f.default ?? ''
            return next
          })
        }
      } catch {
        // Offline fallback: FALLBACK_FEATURES (already in state).
      }
    }

    async function loadModels() {
      try {
        const res = await fetch(`${API_BASE}/api/models`)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        const { models: list, extra } = extractModelsPayload(data)
        if (Array.isArray(list) && list.length > 0) {
          if (cancelled) return
          setModels(list)
          if (Object.keys(extra).length > 0) setMetricsExtra(extra)
        } else {
          if (!cancelled) setModelsError('The backend returned an empty model list.')
        }
      } catch {
        if (!cancelled) setModelsError(BACKEND_DOWN_MSG)
      }
    }

    // Best-effort: some backend builds expose the full metrics.json at
    // /api/metrics. A 404 is fine — /api/models is the documented source
    // and everything below works without this call.
    async function loadMetrics() {
      try {
        const res = await fetch(`${API_BASE}/api/metrics`)
        if (!res.ok) return
        const data = await res.json()
        if (cancelled || !data || typeof data !== 'object') return
        const { models: list, extra } = extractModelsPayload(data)
        if (list.length > 0) {
          setModels((prev) => {
            if (prev) return prev
            return list
          })
        }
        setMetricsExtra((prev) => ({ ...(prev || {}), ...extra }))
      } catch {
        // Not exposed — ignore silently.
      }
    }

    checkHealth()
    loadFeatures()
    loadModels()
    loadMetrics()
    return () => {
      cancelled = true
      if (healthTimer) clearTimeout(healthTimer)
    }
  }, [])

  // Smooth-scroll the result into view when a new prediction arrives.
  useEffect(() => {
    if (result) {
      const el = document.getElementById('result-panel')
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [result])

  // --- Validation --------------------------------------------------------
  function validateValue(f, raw) {
    const label = f.label
    const unit = f.unit ? ` ${f.unit}` : ''
    if (raw === '' || raw === null || raw === undefined) {
      if (REQUIRED_FIELDS.includes(f.key)) {
        return `${label} is required — enter a value before predicting.`
      }
      return null // blank non-required field: imputed, not an error
    }
    const n = Number(raw)
    if (!Number.isFinite(n)) return `${label} must be a number.`
    if (f.min !== undefined && f.min !== null && n < f.min) {
      return `${label} must be at least ${f.min}${unit} — you entered ${raw}.`
    }
    if (f.max !== undefined && f.max !== null && n > f.max) {
      return `${label} must be at most ${f.max}${unit} — you entered ${raw}.`
    }
    return null
  }

  function validateAll(currentValues) {
    const errs = {}
    for (const f of features) {
      const msg = validateValue(f, currentValues[f.key])
      if (msg) errs[f.key] = msg
    }
    return errs
  }

  function setField(key, raw) {
    // Keep the raw text so out-of-range / partial input is visible and can
    // be validated; numbers are converted only when the form is sent.
    setValues((prev) => ({ ...prev, [key]: raw }))
    setFieldErrors((prev) => {
      if (!prev[key]) return prev
      const f = featureByKey[key]
      const msg = f ? validateValue(f, raw) : null
      const next = { ...prev }
      if (msg) next[key] = msg
      else delete next[key]
      return next
    })
  }

  function loadSample(sample) {
    setValues({ ...sample.values })
    setLoadedSample(sample.name)
    setPatientName('') // never pre-filled: a stored name must be one the user typed
    setPatientId('')
    setSubmittedPatient(null)
    setResult(null)
    setResultAt(null)
    setSubmittedValues(null)
    setError('')
    setFieldErrors({})
  }

  function resetForm() {
    setValues(defaultsFromFeatures(features))
    setLoadedSample('Dataset-median defaults')
    setPatientName('')
    setPatientId('')
    setSubmittedPatient(null)
    setResult(null)
    setResultAt(null)
    setSubmittedValues(null)
    setError('')
    setFieldErrors({})
  }

  function clearForm() {
    const empty = {}
    for (const f of features) empty[f.key] = ''
    setValues(empty)
    setLoadedSample('Cleared — no values entered')
    setPatientName('')
    setPatientId('')
    setSubmittedPatient(null)
    setResult(null)
    setResultAt(null)
    setSubmittedValues(null)
    setError('')
    setFieldErrors({})
  }

  async function handlePredict(event) {
    event.preventDefault()
    setError('')
    setResult(null)
    setResultAt(null)
    setSubmittedPatient(null)

    // 1) Empty input must never produce a confident prediction.
    const filledCount = features.filter((f) => {
      const v = values[f.key]
      return v !== '' && v !== null && v !== undefined
    }).length
    if (filledCount === 0) {
      setError(
        'No values entered — a prediction from no data would not be meaningful. ' +
          `Enter at least the required fields (${REQUIRED_FIELDS.map((k) => labelOf(features, k)).join(', ')}), ` +
          'or load a sample patient.',
      )
      return
    }

    // 2) Range / required validation — block before anything is sent.
    const errs = validateAll(values)
    if (Object.keys(errs).length > 0) {
      setFieldErrors(errs)
      const missingRequired = REQUIRED_FIELDS.filter((k) => errs[k])
      setError(
        (missingRequired.length > 0
          ? `Missing required field(s): ${missingRequired.map((k) => labelOf(features, k)).join(', ')}. `
          : '') +
          `Please fix ${Object.keys(errs).length} field(s) before predicting — ` +
          'each one shows its valid range below its input. Nothing was sent to the backend.',
      )
      const firstKey = Object.keys(errs)[0]
      const el = document.querySelector(`[data-field="${firstKey}"]`)
      if (el) el.focus()
      return
    }
    setFieldErrors({})

    // 3) Send: blank fields go as null so the backend median-imputes them
    // (and, once that backend change is live, counts them in imputedCount).
    const payload = {}
    const imputedKeys = []
    for (const f of features) {
      const v = values[f.key]
      if (v === '' || v === null || v === undefined) {
        payload[f.key] = null
        imputedKeys.push(f.key)
      } else {
        payload[f.key] = Number(v)
      }
    }
    // Optional patient details — stored with the history record, never
    // model inputs.
    const patientNameTrimmed = patientName.trim()
    const patientIdTrimmed = patientId.trim()
    if (patientNameTrimmed) payload.patientName = patientNameTrimmed
    if (patientIdTrimmed) payload.patientId = patientIdTrimmed

    setPredicting(true)
    try {
      const res = await fetch(`${API_BASE}/api/predict?mode=${predictMode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        let detail = ''
        try {
          const data = await res.json()
          detail = data.error || data.message || JSON.stringify(data)
        } catch {
          detail = await res.text().catch(() => '')
        }
        // A 400 from the backend (e.g. missing required fields) is shown
        // verbatim — it is the authoritative list.
        throw new Error(
          `The backend rejected the request (HTTP ${res.status})` +
            (detail ? `: ${detail}` : '.'),
        )
      }
      const data = await res.json()
      setResult(data)
      setResultAt(new Date())
      setSubmittedValues(payload)
      setSubmittedPatient({ name: patientNameTrimmed || null, id: patientIdTrimmed || null })
      setSubmittedImputedKeys(imputedKeys)
    } catch (err) {
      setError(
        err.message && err.message.includes('Failed to fetch')
          ? `${BACKEND_DOWN_MSG} (Could not reach ${API_BASE || ''}/api/predict.)`
          : `Prediction failed: ${err.message || err}. ${BACKEND_DOWN_MSG}`,
      )
    } finally {
      setPredicting(false)
    }
  }

  function handlePrint() {
    window.print()
  }

  // --- Prediction history (admin key) ------------------------------------
  async function loadHistory() {
    setHistoryError('')
    setHistory(null)
    if (!adminKey) {
      setHistoryError('Enter the admin key (ADMIN_KEY) to view prediction history — history is admin-only.')
      return
    }
    setHistoryLoading(true)
    try {
      const res = await fetch(`${API_BASE}/api/predictions?limit=20`, {
        headers: { 'X-Admin-Key': adminKey },
      })
      const data = await res.json().catch(() => null)
      if (!res.ok) {
        throw new Error(
          data?.error || `History request failed (HTTP ${res.status}).`,
        )
      }
      setHistory(Array.isArray(data) ? data : [])
    } catch (err) {
      setHistoryError(
        err.message && err.message.includes('Failed to fetch')
          ? `${BACKEND_DOWN_MSG} (Could not reach ${API_BASE || ''}/api/predictions.)`
          : `Could not load history: ${err.message || err}`,
      )
    } finally {
      setHistoryLoading(false)
    }
  }

  async function verifyOutcome(id) {
    const value = verifyValues[id]
    if (value !== '0' && value !== '1') {
      setHistoryError(`Choose 0 (no PCOS) or 1 (PCOS confirmed) for prediction #${id} first.`)
      return
    }
    setHistoryError('')
    setVerifyBusy((prev) => ({ ...prev, [id]: true }))
    try {
      const res = await fetch(`${API_BASE}/api/predictions/${id}/outcome`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', 'X-Admin-Key': adminKey },
        body: JSON.stringify({ verifiedOutcome: Number(value) }),
      })
      const data = await res.json().catch(() => null)
      if (!res.ok) {
        throw new Error(data?.error || `Verify failed (HTTP ${res.status}).`)
      }
      setHistory((prev) =>
        prev ? prev.map((r) => (r.id === id ? { ...r, verifiedOutcome: data.verifiedOutcome, verifiedAt: data.verifiedAt } : r)) : prev,
      )
    } catch (err) {
      setHistoryError(`Could not record the verified outcome: ${err.message || err}`)
    } finally {
      setVerifyBusy((prev) => ({ ...prev, [id]: false }))
    }
  }

  // The deployed model is the one the BACKEND chose on train CV
  // (metrics.bestModel), matched by name — never the highest test
  // accuracy. Gradient Boosting and the Soft Voting Ensemble tie on test
  // accuracy (both 88.89%), and picking by test accuracy would put the
  // star on the wrong model. Falls back to test accuracy only when the
  // backend does not expose /api/metrics.
  const bestModel = useMemo(() => {
    if (!models) return null
    const deployedName = metricsExtra?.bestModel
    if (deployedName) {
      const match = models.find((m) => m.name === deployedName)
      if (match) return match
    }
    return models.reduce((best, m) =>
      Number(m.accuracy ?? 0) > Number(best.accuracy ?? 0) ? m : best, models[0])
  }, [models, metricsExtra])

  // --- Metrics section data (all backend-provided, or absent) -------------
  const hasAuc = models ? models.some((m) => modelAuc(m) !== null) : false
  const hasThreshold = models ? models.some((m) => modelThreshold(m) !== null) : false
  const screeningModel = models ? models.find(isScreeningModel) : null

  // Feature importance: from /api/models (or /api/metrics) when present.
  const importanceRaw =
    metricsExtra?.featureImportance ??
    metricsExtra?.feature_importance ??
    bestModel?.featureImportance ??
    null
  const importanceList = useMemo(
    () => normalizeImportance(importanceRaw),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [JSON.stringify(importanceRaw)],
  )
  const topImportance = importanceList.slice(0, 10)
  const maxImportance = topImportance.length > 0 ? topImportance[0].score : 0

  // Confusion matrices / ROC curves, per model when the backend provides them.
  const modelsWithCm = models ? models.filter((m) => modelConfusion(m)) : []
  const topLevelCm = normalizeConfusion(
    metricsExtra?.confusionMatrix ?? metricsExtra?.confusion_matrix,
  )
  const cmByModelName = {}
  for (const m of modelsWithCm) cmByModelName[m.name] = modelConfusion(m)
  if (topLevelCm && bestModel) cmByModelName[bestModel.name] = cmByModelName[bestModel.name] || topLevelCm
  const cmNames = Object.keys(cmByModelName)
  const activeCmName =
    (cmModelName && cmByModelName[cmModelName] ? cmModelName : null) ||
    (bestModel && cmByModelName[bestModel.name] ? bestModel.name : cmNames[0] || '')
  const activeCm = activeCmName ? cmByModelName[activeCmName] : null

  const modelsWithRoc = models ? models.filter((m) => modelRocPoints(m)) : []
  const topLevelRoc = normalizeRocPoints(
    metricsExtra?.rocCurve ?? metricsExtra?.rocPoints ?? metricsExtra?.roc_curve,
  )
  const rocByModelName = {}
  for (const m of modelsWithRoc) rocByModelName[m.name] = modelRocPoints(m)
  if (topLevelRoc && bestModel) rocByModelName[bestModel.name] = rocByModelName[bestModel.name] || topLevelRoc
  const rocNames = Object.keys(rocByModelName)
  const activeRocName =
    (bestModel && rocByModelName[bestModel.name] ? bestModel.name : rocNames[0] || '')
  const activeRocPoints = activeRocName ? rocByModelName[activeRocName] : null
  const activeRocAuc =
    activeRocName && models
      ? modelAuc(models.find((m) => m.name === activeRocName)) ?? metricsExtra?.rocAuc ?? metricsExtra?.auc ?? null
      : null
  const anyRocAuc = hasAuc || metricsExtra?.rocAuc !== undefined || metricsExtra?.auc !== undefined

  const probability = result
    ? result.probability ?? result.confidence ?? null
    : null
  const probabilityPct = pct(probability)
  const confidencePct = pct(result?.confidence ?? result?.probability)
  const isPcos = result
    ? result.prediction === 1 || result.prediction === true ||
      (result.prediction === undefined &&
        String(result.label || '').toLowerCase().includes('pcos') &&
        !String(result.label || '').toLowerCase().includes('no'))
    : false

  // Imputed count: backend's imputedCount when live; otherwise the count of
  // fields left blank in this browser (those were sent as null and the
  // backend median-imputes them). Never invented — the source is labelled.
  const backendImputed =
    result?.imputedCount ?? result?.imputed ?? result?.imputedFieldsCount ?? null
  const imputedFromList = Array.isArray(result?.imputedFields)
    ? result.imputedFields.length
    : null
  const imputedCount =
    backendImputed !== null && backendImputed !== undefined
      ? Number(backendImputed)
      : imputedFromList !== null
        ? imputedFromList
        : submittedImputedKeys.length
  const imputedSource =
    backendImputed !== null && backendImputed !== undefined
      ? 'reported by the backend'
      : 'counted in the browser (fields left blank, sent as missing)'

  // Top factors for this result: per-prediction factors first, else the
  // model's overall importance (labelled as overall, not per-patient).
  const resultTopFactors = useMemo(() => {
    if (!result) return []
    const raw =
      result.topFactors ?? result.top_factors ?? result.featureImportance ??
      result.explanation ?? null
    const list = normalizeImportance(raw)
    if (list.length > 0) return { items: list.slice(0, 5), perPatient: true }
    if (topImportance.length > 0) return { items: topImportance.slice(0, 5), perPatient: false }
    return { items: [], perPatient: false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result, JSON.stringify(topImportance)])

  const resultDateStr = resultAt ? resultAt.toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' }) : ''

  return (
    <div className="page">
      <header className="header animate-in">
        <div className="header-main">
          <span className="header-eyebrow">Galgotias College of Engineering and Technology · AKTU</span>
          <h1>Cognitive Analytics in Healthcare</h1>
          <p className="header-sub">PCOD / PCOS prediction with ensemble machine learning</p>
          <p className="team">
            Team: Priyansh Kr. Yadav, Rudra Pratap Singh, Rajneesh Pal
            {' | '}Guide: Mrs. Rashi Chaudhary
          </p>
        </div>
        <div className="header-status">
          <span className={`status-pill ${backendStatus === 'online' ? 'status-online' : backendStatus === 'offline' ? 'status-offline' : 'status-checking'}`}>
            <span className="status-dot" aria-hidden="true" />
            {backendStatus === 'online' ? 'Backend connected' : backendStatus === 'offline' ? 'Backend offline' : 'Checking backend…'}
          </span>
          <span className="header-model">{bestModel ? bestModel.name : 'Soft Voting Ensemble'}</span>
        </div>
      </header>

      <div className="banner banner-disclaimer" role="note">
        <strong>⚠ {DISCLAIMER}</strong>
        {' '}This tool is a screening aid for a student project. It does not
        diagnose PCOS, and its output must never replace a doctor&apos;s
        examination, tests, or advice.
      </div>

      {backendStatus === 'offline' && (
        <div className="banner banner-warn" role="alert">
          <strong>Backend not running</strong> — start it on port 8080
          (run the Spring Boot app in <code>../backend</code>, then refresh this page).
          The form below still works with the built-in feature list, but
          prediction needs the backend.
        </div>
      )}
      {backendStatus === 'online' && (
        <div className="banner banner-ok animate-in" role="status">
          Backend connected at {API_BASE || 'http://localhost:8080'} — ready to predict.
        </div>
      )}

      {models && bestModel && (
        <div className="stat-row" aria-label="Model overview">
          <StatCard label="Deployed model" value={bestModel.name} sub="Chosen on cross-validation" delay={0} />
          <StatCard label="Test accuracy" value={pct(bestModel.accuracy)} sub="Held-out test set, n = 108" delay={50} />
          <StatCard label="ROC-AUC" value={modelAuc(bestModel) !== null ? modelAuc(bestModel).toFixed(3) : null} sub="Discrimination" delay={100} />
          {screeningModel && (
            <StatCard label="Screening recall" value={pct(screeningModel.recall)} sub="23 non-invasive features" delay={150} />
          )}
        </div>
      )}

      <main>
        <section className="card animate-in" style={{ animationDelay: '80ms' }}>
          <span className="card-eyebrow">Screening tool</span>
          <h2>Predict PCOD / PCOS</h2>
          <p className="muted">
            Enter the patient&apos;s 41 clinical features, or load a real sample
            patient from the dataset. Currently loaded: <strong>{loadedSample}</strong>
            {featuresFromApi
              ? ' — feature list loaded live from the backend.'
              : ' — feature list from the built-in copy (backend feature list unavailable).'}
          </p>
          <p className="muted">
            Required fields: <strong>{REQUIRED_FIELDS.map((k) => labelOf(features, k)).join(', ')}</strong>.
            Other fields may be left blank — they are filled with the training
            median (median imputation) and counted in the result.
            Every number shows its unit and valid range; values outside the
            range are rejected before anything is sent.
          </p>

          <div className="sample-buttons">
            <button type="button" onClick={() => loadSample(SAMPLE_PCOS)}>
              Load Sample: PCOS case
            </button>
            <button type="button" onClick={() => loadSample(SAMPLE_NO_PCOS)}>
              Load Sample: No-PCOS case
            </button>
            <button type="button" className="secondary" onClick={resetForm}>
              Reset
            </button>
            <button type="button" className="secondary" onClick={clearForm}>
              Clear all
            </button>
          </div>

          <form onSubmit={handlePredict} noValidate>
            <fieldset className="section animate-in" style={{ animationDelay: '90ms' }}>
              <legend>Patient Details</legend>
              <div className="fields">
                <label className="field">
                  <span className="field-label">Patient Name</span>
                  <input
                    type="text"
                    data-field="patientName"
                    value={patientName}
                    onChange={(e) => setPatientName(e.target.value)}
                    placeholder="e.g. Patient A"
                    maxLength={100}
                  />
                  <span className="range">Optional — shown on the result and printout, saved with the record.</span>
                </label>
                <label className="field">
                  <span className="field-label">Patient ID</span>
                  <input
                    type="text"
                    data-field="patientId"
                    value={patientId}
                    onChange={(e) => setPatientId(e.target.value)}
                    placeholder="e.g. P-1024"
                    maxLength={64}
                  />
                  <span className="range">Optional — shown on the result and printout, saved with the record.</span>
                </label>
              </div>
            </fieldset>
            {SECTIONS.map((section, sIdx) => (
              <fieldset key={section.id} className="section animate-in" style={{ animationDelay: `${140 + sIdx * 50}ms` }}>
                <legend>{section.title}</legend>
                <div className="fields">
                  {section.keys.map((key) => {
                    const f = featureByKey[key]
                    if (!f) return null
                    const value = values[key] ?? ''
                    const err = fieldErrors[key]
                    return (
                      <label key={key} className="field">
                        <span className="field-label">
                          {f.label}
                          {f.unit ? <span className="unit"> ({f.unit})</span> : null}
                          {REQUIRED_FIELDS.includes(key) ? <span className="required-mark"> *</span> : null}
                        </span>
                        {f.type === 'binary' ? (
                          <select
                            data-field={key}
                            value={value === '' || value === null ? '' : String(value)}
                            onChange={(e) => setField(key, e.target.value)}
                            aria-invalid={Boolean(err)}
                            className={err ? 'invalid' : ''}
                          >
                            <option value="">— select —</option>
                            <option value="0">No (0)</option>
                            <option value="1">Yes (1)</option>
                          </select>
                        ) : (
                          <input
                            type="number"
                            step="any"
                            min={f.min}
                            max={f.max}
                            data-field={key}
                            value={value}
                            onChange={(e) => setField(key, e.target.value)}
                            aria-invalid={Boolean(err)}
                            aria-describedby={err ? `err-${key}` : undefined}
                            className={err ? 'invalid' : ''}
                          />
                        )}
                        <span className="range">
                          Valid range: {f.min} – {f.max}
                          {f.unit ? ` ${f.unit}` : ''}
                        </span>
                        {err && (
                          <span className="field-error" id={`err-${key}`} role="alert">{err}</span>
                        )}
                      </label>
                    )
                  })}
                </div>
              </fieldset>
            ))}

            <div className="mode-toggle" role="group" aria-label="Prediction mode">
              <span className="mode-toggle-label">Prediction mode</span>
              <div className="segmented">
                <label className={predictMode === 'full' ? 'seg-option seg-active' : 'seg-option'}>
                  <input
                    type="radio"
                    name="predict-mode"
                    value="full"
                    checked={predictMode === 'full'}
                    onChange={() => setPredictMode('full')}
                  />
                  <span className="seg-title">Full</span>
                  <span className="seg-sub">41 features · deployed model</span>
                </label>
                <label className={predictMode === 'screening' ? 'seg-option seg-active' : 'seg-option'}>
                  <input
                    type="radio"
                    name="predict-mode"
                    value="screening"
                    checked={predictMode === 'screening'}
                    onChange={() => setPredictMode('screening')}
                  />
                  <span className="seg-title">Screening</span>
                  <span className="seg-sub">23 non-invasive features</span>
                </label>
              </div>
              <p className="muted mode-toggle-note">
                Screening needs no lab or ultrasound values — any of those
                entered above are ignored in screening mode. The screening
                model is less accurate (test accuracy 82.4%, 95% CI 75.0–88.9%)
                but catches more cases (recall 82.9%).
              </p>
            </div>

            <button type="submit" className="predict-btn" disabled={predicting}>
              {predicting ? (<><span className="spinner" aria-hidden="true" /> Predicting…</>) : 'Predict'}
            </button>
          </form>

          {predicting && (
            <div className="predicting-panel" role="status" aria-live="polite">
              <span className="spinner spinner-lg" aria-hidden="true" />
              <p><strong>Running the ensemble…</strong></p>
              <p className="muted">
                {predictMode === 'screening'
                  ? 'Scoring 23 non-invasive features against the screening model.'
                  : 'Scoring 41 features against the deployed model.'}
              </p>
              <div className="skeleton skeleton-line" />
              <div className="skeleton skeleton-line short" />
            </div>
          )}

          {error && (
            <div className="banner banner-error animate-in" role="alert">{error}</div>
          )}

          {result && (
            <div
              id="result-panel"
              className={`result ${isPcos ? 'result-pcos' : 'result-no-pcos'}`}
              role="status"
            >
              <div className="result-top">
                {probabilityPct && (
                  <ProgressRing
                    fraction={probability ?? 0}
                    label={probabilityPct}
                    tone={isPcos ? 'danger' : 'ok'}
                  />
                )}
                <div className="result-head">
                  <span className="result-eyebrow">Screening result</span>
                  <h3>{isPcos ? 'High risk \u2014 refer for testing' : 'Low risk'}</h3>
                  <p className="result-label">
                    {result.label ? `Model output: ${result.label}` : ''}
                  </p>
                </div>
              </div>
              {submittedPatient && (submittedPatient.name || submittedPatient.id) && (
                <p className="result-patient">
                  Patient: <strong>{submittedPatient.name || '—'}</strong>
                  {submittedPatient.id ? ` (ID: ${submittedPatient.id})` : ''}
                </p>
              )}
              {probabilityPct && (
                <p className="result-prob">
                  Probability: <strong>{probabilityPct}</strong>
                  {confidencePct && confidencePct !== probabilityPct
                    ? ` (confidence ${confidencePct})`
                    : ''}
                </p>
              )}
              {typeof result.threshold === 'number' && typeof result.probability === 'number' && (
                <div
                  className="prob-gauge"
                  role="img"
                  aria-label={`Probability ${probabilityPct}, decision threshold ${pct(result.threshold)}`}
                >
                  <div className="prob-gauge-track">
                    <div
                      className="prob-gauge-fill"
                      style={{ width: `${Math.max(0, Math.min(100, result.probability * 100))}%` }}
                    />
                    <div
                      className="prob-gauge-threshold"
                      style={{ left: `${Math.max(0, Math.min(100, result.threshold * 100))}%` }}
                    />
                  </div>
                  <div className="prob-gauge-labels">
                    <span>0%</span>
                    <span>Decision threshold {pct(result.threshold)}</span>
                    <span>100%</span>
                  </div>
                </div>
              )}
              <p className="result-model">
                Model: <strong>{result.model || '—'}</strong>
                {result.modelAccuracy !== undefined && result.modelAccuracy !== null
                  ? ` — test accuracy ${pct(result.modelAccuracy)}`
                  : ''}
                {result.modelAccuracyCiLow !== undefined && result.modelAccuracyCiHigh !== undefined
                  ? ` (95% CI ${pct(result.modelAccuracyCiLow)}–${pct(result.modelAccuracyCiHigh)})`
                  : ''}
                {result.mode || result.screeningMode
                  ? ` — mode: ${result.mode || (result.screeningMode ? 'screening' : '')}`
                  : ''}
              </p>
              {(result.modelAccuracyNote || result.modelAccuracy !== undefined) && (
                <p className="muted result-accuracy-note">
                  {result.modelAccuracyNote ||
                    'Test accuracy describes the model on its 108-row held-out test set — not this individual prediction.'}
                </p>
              )}
              <p className="result-imputed">
                {imputedCount} of {N_FEATURES} fields were imputed
                <span className="muted"> ({imputedSource}).</span>
                {imputedCount === 0
                  ? ' All values were entered, so no training-median filling was needed.'
                  : ' Missing values were filled with the training median.'}
              </p>
              {result.warning && (
                <p className="result-warning" role="alert">
                  ⚠ {result.warning}
                </p>
              )}
              {!result.warning && imputedCount >= HIGH_IMPUTATION && (
                <p className="result-warning" role="alert">
                  ⚠ More than half the fields were imputed — this result rests
                  mostly on typical values, not on the patient&apos;s own data.
                  Treat it with extra caution and consult a doctor.
                </p>
              )}
              {resultTopFactors.items.length > 0 ? (
                <div className="top-factors">
                  <h4>
                    {resultTopFactors.perPatient
                      ? 'Top factors for this prediction'
                      : 'Top factors for the model overall'}
                  </h4>
                  {!resultTopFactors.perPatient && (
                    <p className="muted">
                      Overall Random Forest importance on the training data —
                      not a per-patient explanation.
                    </p>
                  )}
                  <ul>
                    {resultTopFactors.items.map((item) => (
                      <li key={item.key}>
                        <span className="tf-label">{labelOf(features, item.key)}</span>
                        <span
                          className="tf-bar"
                          style={{ width: `${maxImportance ? (item.score / maxImportance) * 100 : 0}%` }}
                        />
                        <span className="tf-score">{item.score.toFixed(1)}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="result-note muted">
                  Feature-importance data comes from the backend — it is not
                  shown here because the backend has not provided it yet.
                  No importance values have been guessed.
                </p>
              )}
              <p className="result-date muted">Predicted on {resultDateStr}</p>
              <button type="button" className="download-btn" onClick={handlePrint}>
                Download result (PDF)
              </button>
              <p className="muted print-hint">
                Opens the print dialog — choose “Save as PDF” to download.
                The printout contains the patient summary, prediction,
                probability, model, imputed count, date, and the disclaimer.
              </p>
              <p className="result-disclaimer"><strong>{DISCLAIMER}</strong></p>
            </div>
          )}
        </section>

        <section className="card animate-in">
          <span className="card-eyebrow">Evidence</span>
          <h2>How it works / Models compared</h2>
          {screeningModel && (
            <p className="screening-note" role="note">
              <strong>Screening-mode model: {screeningModel.name}</strong>
              {screeningModel.description ? ` — ${screeningModel.description}` : ''}
              {screeningModel.threshold !== undefined && screeningModel.threshold !== null
                ? ` (decision threshold ${screeningModel.threshold})`
                : ''}
            </p>
          )}
          {models ? (
            <table className="models-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  {hasAuc && <th>ROC-AUC</th>}
                  {hasThreshold && <th>Threshold</th>}
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr
                    key={m.name}
                    className={bestModel && m.name === bestModel.name ? 'best-row' : ''}
                  >
                    <td>
                      {m.name}
                      {bestModel && m.name === bestModel.name ? ' ★ Best' : ''}
                      {isScreeningModel(m) ? ' — Screening mode' : ''}
                    </td>
                    <td>{pct(m.accuracy) ?? '—'}</td>
                    <td>{pct(m.precision) ?? '—'}</td>
                    <td>{pct(m.recall) ?? '—'}</td>
                    <td>{pct(m.f1) ?? '—'}</td>
                    {hasAuc && <td>{modelAuc(m) !== null ? modelAuc(m).toFixed(3) : '—'}</td>}
                    {hasThreshold && <td>{modelThreshold(m) !== null ? modelThreshold(m) : '—'}</td>}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">
              {modelsError || 'Loading models…'}
              {' '}Deployed model: Soft Voting Ensemble — test accuracy 88.9%
              (95% CI 82.4–94.4%, test set n = 108). The full comparison table
              loads here automatically once the backend is running.
            </p>
          )}

          <h3>Model performance charts</h3>
          {activeCm ? (
            <>
              {cmNames.length > 1 && (
                <label className="chart-select">
                  Confusion matrix for:{' '}
                  <select value={activeCmName} onChange={(e) => setCmModelName(e.target.value)}>
                    {cmNames.map((n) => (
                      <option key={n} value={n}>{n}</option>
                    ))}
                  </select>
                </label>
              )}
              <ConfusionMatrix cm={activeCm} modelName={activeCmName} />
            </>
          ) : (
            <p className="metrics-note muted">
              Confusion matrix (TP / FP / FN / TN) not shown: the backend does
              not expose it yet — neither per model in <code>/api/models</code> nor
              in <code>/api/metrics</code>. No matrix has been guessed from the
              accuracy/precision/recall values. It will appear here automatically
              once the backend returns a <code>confusionMatrix</code> per model.
            </p>
          )}
          {activeRocPoints ? (
            <RocChart points={activeRocPoints} auc={activeRocAuc} modelName={activeRocName} />
          ) : anyRocAuc ? (
            <p className="metrics-note muted">
              ROC-AUC is reported in the table above where the backend provides
              it. The ROC curve itself is not drawn: the backend supplies no ROC
              curve points, and curve points are never fabricated.
            </p>
          ) : (
            <p className="metrics-note muted">
              ROC curve / ROC-AUC not shown: the backend does not expose ROC
              points or an AUC value yet (in <code>/api/models</code> or{' '}
              <code>/api/metrics</code>). The curve will be drawn automatically
              once the backend returns <code>rocCurve</code> points and/or{' '}
              <code>rocAuc</code> per model.
            </p>
          )}

          {topImportance.length > 0 ? (
            <div className="top-factors">
              <h3>Top factors (model overall)</h3>
              <p className="muted">
                Random Forest feature importance on the training data, from the
                backend — the 10 most influential of the 41 features.
              </p>
              <ul>
                {topImportance.map((item) => (
                  <li key={item.key}>
                    <span className="tf-label">{labelOf(features, item.key)}</span>
                    <span
                      className="tf-bar"
                      style={{ width: `${maxImportance ? (item.score / maxImportance) * 100 : 0}%` }}
                    />
                    <span className="tf-score">{item.score.toFixed(1)}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="metrics-note muted">
              Feature importance is not shown: the backend does not expose it
              yet in <code>/api/models</code> (or <code>/api/metrics</code>).
              No importance values have been guessed.
            </p>
          )}

          <p className="muted">
            Dataset: 541 patients, 45 attributes — Kaggle PCOS dataset by
            Prasoon Kottarathil (PCOS data without infertility, 10 hospitals in
            Kerala). Preprocessing: mean imputation + standardisation, 80/20
            train/test split. The model uses 41 input features (identifiers and
            the target column excluded).
          </p>
        </section>

        <section className="card animate-in">
          <span className="card-eyebrow">Admin</span>
          <h2>Prediction history (admin only)</h2>
          <p className="muted">
            Saved predictions, latest first. History contains patient health
            data, so it needs the admin key (ADMIN_KEY) — the key is kept in
            this page&apos;s memory only and is never stored in the browser.
            A doctor can record a verified outcome per row; only verified rows
            with no imputed fields are ever used for retraining.
          </p>
          <div className="admin-key-row">
            <label>
              Admin key:{' '}
              <input
                type="password"
                value={adminKey}
                onChange={(e) => setAdminKey(e.target.value)}
                placeholder="ADMIN_KEY"
                autoComplete="off"
              />
            </label>
            <button type="button" onClick={loadHistory} disabled={historyLoading}>
              {historyLoading ? 'Loading…' : 'Load history'}
            </button>
          </div>
          {historyError && (
            <div className="banner banner-error" role="alert">{historyError}</div>
          )}
          {history && history.length === 0 && (
            <p className="muted">No predictions saved yet.</p>
          )}
          {history && history.length > 0 && (
            <table className="models-table history-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Patient</th>
                  <th>Date</th>
                  <th>Prediction</th>
                  <th>Probability</th>
                  <th>Model</th>
                  <th>Imputed</th>
                  <th>Verified outcome</th>
                </tr>
              </thead>
              <tbody>
                {history.map((r) => (
                  <tr key={r.id}>
                    <td>{r.id}</td>
                    <td>{r.patientName || '—'}{r.patientId ? ` (${r.patientId})` : ''}</td>
                    <td>{r.createdAt ? new Date(r.createdAt).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' }) : '—'}</td>
                    <td>{r.label || (r.prediction === 1 ? 'High risk \u2014 refer for testing' : 'Low risk')}</td>
                    <td>{pct(r.probability) ?? '—'}</td>
                    <td>{r.model || '—'}</td>
                    <td>{Array.isArray(r.imputedFields) ? r.imputedFields.length : '—'}</td>
                    <td>
                      {r.verifiedOutcome === 0 || r.verifiedOutcome === 1 ? (
                        <span>
                          {r.verifiedOutcome === 1 ? 'PCOS confirmed' : 'No PCOS'}
                          {r.verifiedAt ? ` (${new Date(r.verifiedAt).toLocaleDateString('en-IN')})` : ''}
                        </span>
                      ) : (
                        <span className="verify-row">
                          <select
                            value={verifyValues[r.id] ?? ''}
                            onChange={(e) => setVerifyValues((prev) => ({ ...prev, [r.id]: e.target.value }))}
                            aria-label={`Verified outcome for prediction ${r.id}`}
                          >
                            <option value="">— select —</option>
                            <option value="0">0 — No PCOS</option>
                            <option value="1">1 — PCOS confirmed</option>
                          </select>
                          <button
                            type="button"
                            className="secondary"
                            onClick={() => verifyOutcome(r.id)}
                            disabled={verifyBusy[r.id]}
                          >
                            {verifyBusy[r.id] ? 'Saving…' : 'Verify'}
                          </button>
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </main>

      <footer className="footer">
        {DISCLAIMER} Educational demo — not a medical diagnosis.
      </footer>

      {/* Print-only result summary: the "Download result (PDF)" printout.
          window.print() + the print stylesheet shows only this block. */}
      {result && submittedValues && (
        <div id="result-print" className="print-only" aria-hidden="true">
          <h1>PCOD / PCOS Screening Result</h1>
          <p>Cognitive Analytics in Healthcare — Galgotias College of Engineering and Technology, AKTU</p>
          <p>Team: Priyansh Kr. Yadav, Rudra Pratap Singh, Rajneesh Pal | Guide: Mrs. Rashi Chaudhary</p>
          <p>Date: {resultDateStr}</p>
          <hr />
          <h2>{isPcos ? 'High risk \u2014 refer for testing' : 'Low risk'}</h2>
          {result.label && <p>Model output: {result.label}</p>}
          {probabilityPct && <p>Probability: {probabilityPct}{confidencePct && confidencePct !== probabilityPct ? ` (confidence ${confidencePct})` : ''}</p>}
          <p>
            Model: {result.model || '—'}
            {result.modelAccuracy !== undefined && result.modelAccuracy !== null
              ? ` — test accuracy ${pct(result.modelAccuracy)}`
              : ''}
            {result.modelAccuracyCiLow !== undefined && result.modelAccuracyCiHigh !== undefined
              ? ` (95% CI ${pct(result.modelAccuracyCiLow)}–${pct(result.modelAccuracyCiHigh)}, test set n = 108)`
              : ''}
          </p>
          {result.warning && <p>Warning: {result.warning}</p>}
          <p>
            {imputedCount} of {N_FEATURES} fields were imputed ({imputedSource}).
            Missing values were filled with the training median.
          </p>
          <h2>Patient summary (inputs)</h2>
          {submittedPatient && (submittedPatient.name || submittedPatient.id) ? (
            <p>
              Patient: {submittedPatient.name || '—'}
              {submittedPatient.id ? ` (ID: ${submittedPatient.id})` : ''}
            </p>
          ) : (
            <p>No name or ID was given for this prediction.</p>
          )}
          <table>
            <thead>
              <tr><th>Feature</th><th>Value</th><th>Unit</th><th>Imputed?</th></tr>
            </thead>
            <tbody>
              {features.map((f) => {
                const sent = submittedValues[f.key]
                const wasImputed = sent === null || sent === undefined
                return (
                  <tr key={f.key}>
                    <td>{f.label}</td>
                    <td>{wasImputed ? '—' : fmtNum(sent)}</td>
                    <td>{f.unit || '—'}</td>
                    <td>{wasImputed ? 'Yes (training median)' : 'No'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          <hr />
          <p><strong>{DISCLAIMER}</strong></p>
          <p>
            Educational demo — not a medical diagnosis. This screening aid does
            not replace a doctor&apos;s examination, tests, or advice.
          </p>
        </div>
      )}
    </div>
  )
}
