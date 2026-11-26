# PCOD Prediction — Frontend

React + Vite (JavaScript, plain CSS) single-page app for the final-year project
**Cognitive Analytics in Healthcare — PCOD Prediction**.

Team: Priyansh Kr. Yadav, Rudra Pratap Singh, Rajneesh Pal |
Guide: Mrs. Rashi Chaudhary |
Galgotias College of Engineering and Technology, AKTU.

## Run

```bash
cd frontend
npm install
npm run dev
```

Then open **http://localhost:5173**.

The backend (Spring Boot, in `../backend`) must be running on
**http://localhost:8080** for live prediction. If it is not running, the page
still loads and shows a clear
"Backend not running — start it on port 8080" banner; the form works from its
built-in feature list, and prediction is disabled until the backend is up.

Production build:

```bash
npm run build        # output in dist/
npm run preview      # serves the built output on http://localhost:5173
```

## How it works

- On load the app calls `GET /api/health`, `GET /api/features` and
  `GET /api/models` on the backend. If `/api/features` fails, it falls back
  to a hardcoded copy of the same 41 features (see `src/features.js`).
  A best-effort `GET /api/metrics` is also tried (ignored on 404) in case
  the backend exposes the full `metrics.json` there.
- A prominent disclaimer banner sits under the header on every page load:
  **"Screening aid, not a diagnosis — consult a doctor."** The same wording
  is in the footer and on the result card / printout.
- **Patient Details (optional):** name and patient ID, entered at the top of
  the form. They show on the result card and the printout, are sent with the
  predict request, and are stored with the history record in MySQL — they
  are never sent to the model.
- The prediction form is rendered dynamically from that feature list, grouped
  into 4 sections: Basic, Hormonal & Lab, Symptoms & Lifestyle, Vitals &
  Ultrasound. Binary (Y/N) fields render as No/Yes dropdowns; the rest are
  number inputs with min/max from the dataset.
- **Validation:** every number input shows its unit and valid
  range, and values outside the range are rejected live (red border + a
  message such as "Age must be at most 48 yrs — you entered 99.") and again
  on submit — nothing out of range is ever sent. Required fields are
  the backend's 11 non-invasive fields — **Age, BMI, Cycle, Cycle length,
  Weight gain, Hair growth, Skin darkening, Hair loss, Pimples, Fast food,
  Regular exercise** (marked `*`); the
  form lists any missing required fields before sending. Any other field may
  be left blank: it is sent as `null` so the backend median-imputes it, and it
  is counted in the result (see below). **Empty input is blocked outright:**
  with no values entered at all, no request is sent — a prediction from no
  data would not be meaningful. A backend 400 (e.g. its own missing-field
  list, once live) is shown verbatim. **Clear all** empties the form so this
  can be tried; **Reset** restores dataset-median defaults.
- The form starts with **Sample Patient 2** values (a true PCOS case from
  `../data/sample_patients.json`) but the patient name field is left blank — a stored name must be one the user typed. Buttons load the PCOS sample, the No-PCOS
  sample (Sample Patient 1), or reset to dataset-median defaults. Both
  samples match `sample_patients.json` field by field.
- **Predict** sends `POST /api/predict` with a flat JSON object of the 41
  clean keys (blank fields as `null`) and shows a result card: "High risk — refer for testing" (red) or
  "Low risk" (green) — screening language, not a diagnosis — probability/confidence as a percentage with a
  ring and a gauge with the decision threshold marked, and the model name + its accuracy. Errors (e.g. backend down) are
  shown in plain text.
- **Result card:** shows **"X of 41 fields were imputed"** —
  the backend's `imputedCount` when it returns one, otherwise the count of
  fields left blank in the browser (the source is labelled on the card).
  More than half imputed (≥21) adds an extra caution warning. The card also
  shows **top factors**: per-prediction factors from the predict response
  when the backend provides them, otherwise the model's overall top 5 from
  the backend's feature-importance data (labelled as overall, not
  per-patient). If the backend provides no importance data, the card says so
  instead of guessing values. A **"Download result (PDF)"** button opens the
  print dialog — choose "Save as PDF". It prints only a result summary
  (print stylesheet, no new dependency): patient summary of the
  non-identifying inputs with an imputed? flag per field, prediction,
  probability, model, imputed count, date, and the disclaimer.
- The "How it works / Models compared" section renders the `GET /api/models`
  table (Model | Accuracy | Precision | Recall | F1) with the best model
  highlighted. If the backend is down it shows the synopsis result instead
  (Gradient Boosting, 91.7% accuracy, F1 92%).
- **Metrics section:** ROC-AUC and decision-threshold columns
  appear in the models table when the backend provides them per model, and a
  screening-mode model is badged "Screening mode" when the backend marks one
  (`mode: "screening"` or a screening name). The **confusion matrix**
  (TP/FP/FN/TN, as a labelled table/heatmap, with a per-model selector when
  several models provide one) and the **ROC curve** (plain SVG drawn from the
  backend's ROC points, with the chance diagonal) render only from data the
  backend actually returns (`confusionMatrix` / `rocCurve` points /
  `rocAuc`, per model in `/api/models` or top-level in `/api/metrics`).
  If the backend provides AUC but no curve points, only the AUC value is
  shown — curve points are never fabricated. If it provides neither, the
  section says exactly which fields are missing instead of showing guessed
  numbers. **Feature importance** (top 10, Random Forest, training data)
  renders from the backend's `featureImportance` the same way.

## API contract (backend, port 8080)

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Backend liveness check |
| `GET /api/models` | List of `{name, accuracy, precision, recall, f1}` — or a full metrics object `{models: [...], featureImportance: {key: score}, ...}`. Per model the frontend also reads, when present: `rocAuc` / `auc`, `threshold` / `decisionThreshold`, `mode: "screening"` (or a screening name), `confusionMatrix` (`{tp, fp, fn, tn}` or a 2×2 array), `rocCurve` / `rocPoints` (`[{fpr, tpr}]`). Top-level `confusionMatrix` / `rocCurve` / `rocAuc` are also honoured. |
| `GET /api/metrics` | Best-effort only (404 ignored): full `metrics.json` if the backend serves it — same fields as above. |
| `GET /api/features` | List of 41 features `{key, label, min, max, default, type}` (units come from the frontend fallback copy when the backend sends none) |
| `POST /api/predict` | Flat JSON of the 41 clean keys, blank fields as `null` (or `{"features": {...}}`); returns `{prediction: 0\|1, label, probability, confidence, model, modelAccuracy}` and, once live, `imputedCount` (or `imputedFields`), `topFactors`, and a 400 listing missing required fields. A 400 body is displayed verbatim. |

In dev, Vite proxies `/api` to `http://localhost:8080` (see
`vite.config.js`), so the app uses relative URLs. To call the backend
directly instead, set `VITE_API_URL`, e.g.:

```bash
VITE_API_URL=http://localhost:8080 npm run dev
```

## Feature keys (41)

`age, weightKg, heightCm, bmi, bloodGroup, pulseRate, rr, hb, cycle,
cycleLength, marriageYears, pregnant, abortions, betaHcg1, betaHcg2, fsh, lh,
fshLh, hip, waist, waistHipRatio, tsh, amh, prl, vitD3, prg, rbs, weightGain,
hairGrowth, skinDarkening, hairLoss, pimples, fastFood, regularExercise,
bpSystolic, bpDiastolic, follicleL, follicleR, avgFollicleSizeL,
avgFollicleSizeR, endometrium`

Labels/units and min/max/defaults come from the dataset
(`../data/SCHEMA.md`, `../data/pcos_dataset.csv`, 541 rows). Two clearly
erroneous raw extremes were clipped to plausible clinical bounds in the
fallback copy: pulse rate (raw min 13 bpm → 40) and blood pressure
(raw min 12/8 mmHg → 80/40).

## Sources consulted

- Project dataset/schema: `../data/SCHEMA.md` and
  `../data/sample_patients.json` (Kaggle PCOS dataset by Prasoon
  Kottarathil — https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos).
- Vite + React docs for the dev-server proxy and build setup
  (https://vite.dev/guide/ — server.proxy, preview config).

## Disclaimer

Screening aid, not a diagnosis — consult a doctor. Educational demo.

## Test record

- `npm install` — ok.
- `npm run build` — 0 errors.
- The app loads on http://localhost:5173 with the backend running (verified on the laptop). The confusion matrix, ROC curve and feature importance render from the backend's `/api/models` data; if the backend is down they show a note instead of guessed numbers.
