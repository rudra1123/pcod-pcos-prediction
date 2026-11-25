// ---------------------------------------------------------------------------
// Feature definitions + sample patients (fallback copy of the backend contract)
// ---------------------------------------------------------------------------
// Source: ../data/SCHEMA.md (verbatim Kaggle PCOS column names) and
// ../data/sample_patients.json (3 real, de-identified rows).
// min/max/default below are computed from the 541-row dataset
// (pcos_dataset.csv). Two clearly erroneous extremes in the raw data were
// clipped to plausible clinical bounds and are marked: pulse rate (raw min
// 13 bpm) and blood pressure (raw min 12/8 mmHg).
// The UI first tries GET /api/features and uses the backend's list when it
// is available; this copy is only the offline fallback, and it must stay in
// sync with the backend's 41 clean keys.
// ---------------------------------------------------------------------------

export const FALLBACK_FEATURES = [
  { key: 'age', label: 'Age', unit: 'yrs', min: 20, max: 48, default: 31.43, type: 'number' },
  { key: 'weightKg', label: 'Weight', unit: 'Kg', min: 31, max: 108, default: 59.64, type: 'number' },
  { key: 'heightCm', label: 'Height', unit: 'cm', min: 137, max: 180, default: 156.49, type: 'number' },
  { key: 'bmi', label: 'BMI', unit: 'kg/m²', min: 12.42, max: 38.9, default: 24.31, type: 'number' },
  { key: 'bloodGroup', label: 'Blood Group', unit: 'dataset code 11–18', min: 11, max: 18, default: 14, type: 'number' },
  { key: 'pulseRate', label: 'Pulse rate', unit: 'bpm', min: 40, max: 120, default: 73.25, type: 'number' },
  { key: 'rr', label: 'Respiratory Rate (RR)', unit: 'breaths/min', min: 16, max: 28, default: 19.24, type: 'number' },
  { key: 'hb', label: 'Haemoglobin (Hb)', unit: 'g/dl', min: 8.5, max: 14.8, default: 11.16, type: 'number' },
  { key: 'cycle', label: 'Cycle (R/I)', unit: '2 = Regular, 4/5 = Irregular', min: 2, max: 5, default: 2, type: 'number' },
  { key: 'cycleLength', label: 'Cycle length', unit: 'days', min: 0, max: 12, default: 4.94, type: 'number' },
  { key: 'marriageYears', label: 'Marriage Status', unit: 'yrs', min: 0, max: 30, default: 7.68, type: 'number' },
  { key: 'pregnant', label: 'Pregnant', unit: 'Y/N', min: 0, max: 1, default: 0, type: 'binary' },
  { key: 'abortions', label: 'No. of abortions', unit: '', min: 0, max: 5, default: 0, type: 'number' },
  { key: 'betaHcg1', label: 'beta-HCG (I)', unit: 'mIU/mL', min: 1.3, max: 32460.97, default: 664.55, type: 'number' },
  { key: 'betaHcg2', label: 'beta-HCG (II)', unit: 'mIU/mL', min: 0.99, max: 25000, default: 238.67, type: 'number' },
  { key: 'fsh', label: 'FSH', unit: 'mIU/mL', min: 0.21, max: 5052, default: 14.6, type: 'number' },
  { key: 'lh', label: 'LH', unit: 'mIU/mL', min: 0.02, max: 2018, default: 6.47, type: 'number' },
  { key: 'fshLh', label: 'FSH/LH ratio', unit: '', min: 0, max: 1372.83, default: 6.91, type: 'number' },
  { key: 'hip', label: 'Hip', unit: 'inch', min: 26, max: 48, default: 37.99, type: 'number' },
  { key: 'waist', label: 'Waist', unit: 'inch', min: 24, max: 47, default: 33.84, type: 'number' },
  { key: 'waistHipRatio', label: 'Waist:Hip Ratio', unit: '', min: 0.76, max: 0.98, default: 0.89, type: 'number' },
  { key: 'tsh', label: 'TSH', unit: 'mIU/L', min: 0.04, max: 65, default: 2.98, type: 'number' },
  { key: 'amh', label: 'AMH', unit: 'ng/mL', min: 0.1, max: 66, default: 5.62, type: 'number' },
  { key: 'prl', label: 'PRL', unit: 'ng/mL', min: 0.4, max: 128.24, default: 24.32, type: 'number' },
  { key: 'vitD3', label: 'Vit D3', unit: 'ng/mL', min: 0, max: 6014.66, default: 49.92, type: 'number' },
  { key: 'prg', label: 'PRG', unit: 'ng/mL', min: 0.05, max: 85, default: 0.61, type: 'number' },
  { key: 'rbs', label: 'RBS', unit: 'mg/dl', min: 60, max: 350, default: 99.84, type: 'number' },
  { key: 'weightGain', label: 'Weight gain', unit: 'Y/N', min: 0, max: 1, default: 0, type: 'binary' },
  { key: 'hairGrowth', label: 'Hair growth', unit: 'Y/N', min: 0, max: 1, default: 0, type: 'binary' },
  { key: 'skinDarkening', label: 'Skin darkening', unit: 'Y/N', min: 0, max: 1, default: 0, type: 'binary' },
  { key: 'hairLoss', label: 'Hair loss', unit: 'Y/N', min: 0, max: 1, default: 0, type: 'binary' },
  { key: 'pimples', label: 'Pimples', unit: 'Y/N', min: 0, max: 1, default: 0, type: 'binary' },
  { key: 'fastFood', label: 'Fast food', unit: 'Y/N', min: 0, max: 1, default: 0, type: 'binary' },
  { key: 'regularExercise', label: 'Regular Exercise', unit: 'Y/N', min: 0, max: 1, default: 0, type: 'binary' },
  { key: 'bpSystolic', label: 'BP Systolic', unit: 'mmHg', min: 80, max: 180, default: 114.66, type: 'number' },
  { key: 'bpDiastolic', label: 'BP Diastolic', unit: 'mmHg', min: 40, max: 120, default: 76.93, type: 'number' },
  { key: 'follicleL', label: 'Follicle No. (L)', unit: '', min: 0, max: 22, default: 6.13, type: 'number' },
  { key: 'follicleR', label: 'Follicle No. (R)', unit: '', min: 0, max: 20, default: 6.64, type: 'number' },
  { key: 'avgFollicleSizeL', label: 'Avg. Follicle Size (L)', unit: 'mm', min: 0, max: 24, default: 15.02, type: 'number' },
  { key: 'avgFollicleSizeR', label: 'Avg. Follicle Size (R)', unit: 'mm', min: 0, max: 24, default: 15.45, type: 'number' },
  { key: 'endometrium', label: 'Endometrium', unit: 'mm', min: 0, max: 18, default: 8.48, type: 'number' },
]

// Form sections (each lists feature keys, in display order).
export const SECTIONS = [
  {
    id: 'basic',
    title: '1) Basic',
    keys: ['age', 'weightKg', 'heightCm', 'bmi', 'bloodGroup', 'pulseRate', 'rr', 'hb'],
  },
  {
    id: 'hormonal',
    title: '2) Hormonal & Lab',
    keys: ['fsh', 'lh', 'fshLh', 'tsh', 'amh', 'prl', 'vitD3', 'prg', 'rbs', 'betaHcg1', 'betaHcg2'],
  },
  {
    id: 'symptoms',
    title: '3) Symptoms & Lifestyle',
    keys: ['cycle', 'cycleLength', 'marriageYears', 'pregnant', 'abortions', 'weightGain', 'hairGrowth', 'skinDarkening', 'hairLoss', 'pimples', 'fastFood', 'regularExercise'],
  },
  {
    id: 'vitals',
    title: '4) Vitals & Ultrasound',
    keys: ['bpSystolic', 'bpDiastolic', 'hip', 'waist', 'waistHipRatio', 'follicleL', 'follicleR', 'avgFollicleSizeL', 'avgFollicleSizeR', 'endometrium'],
  },
]

// Real sample patients from ../data/sample_patients.json, converted from the
// verbatim dataset headers to the 41 clean keys above. True labels are kept
// here only so the demo can say what the correct answer is.
export const SAMPLE_PCOS = {
  name: 'Sample Patient 2 — PCOS case (true label: PCOS)',
  values: {
    age: 33, weightKg: 68.8, heightCm: 165, bmi: 25.27089073, bloodGroup: 11,
    pulseRate: 72, rr: 18, hb: 11.8, cycle: 2, cycleLength: 5, marriageYears: 10,
    pregnant: 1, abortions: 0, betaHcg1: 494.08, betaHcg2: 494.08, fsh: 5.54,
    lh: 0.88, fshLh: 6.295454545, hip: 40, waist: 36, waistHipRatio: 0.9,
    tsh: 2.54, amh: 6.63, prl: 10.52, vitD3: 49.7, prg: 0.36, rbs: 84,
    weightGain: 0, hairGrowth: 0, skinDarkening: 0, hairLoss: 1, pimples: 1,
    fastFood: 1, regularExercise: 0, bpSystolic: 120, bpDiastolic: 80,
    follicleL: 13, follicleR: 15, avgFollicleSizeL: 18, avgFollicleSizeR: 20,
    endometrium: 10,
  },
}

export const SAMPLE_NO_PCOS = {
  name: 'Sample Patient 1 — No-PCOS case (true label: No PCOS)',
  values: {
    age: 28, weightKg: 44.6, heightCm: 152, bmi: 19.3, bloodGroup: 15,
    pulseRate: 78, rr: 22, hb: 10.48, cycle: 2, cycleLength: 5, marriageYears: 7,
    pregnant: 0, abortions: 0, betaHcg1: 1.99, betaHcg2: 1.99, fsh: 7.95,
    lh: 3.68, fshLh: 2.160326087, hip: 36, waist: 30, waistHipRatio: 0.8333333333,
    tsh: 0.68, amh: 2.07, prl: 45.16, vitD3: 17.1, prg: 0.57, rbs: 92,
    weightGain: 0, hairGrowth: 0, skinDarkening: 0, hairLoss: 0, pimples: 0,
    fastFood: 1, regularExercise: 0, bpSystolic: 110, bpDiastolic: 80,
    follicleL: 3, follicleR: 3, avgFollicleSizeL: 18, avgFollicleSizeR: 18,
    endometrium: 8.5,
  },
}

export function defaultsFromFeatures(features) {
  const values = {}
  for (const f of features) values[f.key] = f.default ?? ''
  return values
}
