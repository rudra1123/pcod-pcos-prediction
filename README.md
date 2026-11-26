# Cognitive Analytics in Healthcare — PCOD Prediction

A working web application that predicts Polycystic Ovarian Disorder (PCOD) from patient clinical attributes using ensemble machine learning.

- **Team:** Priyansh Kr. Yadav (2300970130098), Rudra Pratap Singh (2300970130107), Rajneesh Pal (2300970130103)
- **Guide:** Mrs. Rashi Chaudhary
- **College:** Galgotias College of Engineering and Technology, Greater Noida — Dr. APJ Abdul Kalam Technical University, Lucknow
- **Stack:** React (Vite) frontend · Java 25 (LTS) + Spring Boot 4.1.2 backend · MySQL 8.x · Smile 6.3.0 (Java ML library) · IntelliJ IDEA

> Screening aid, not a medical diagnosis. Consult a doctor.

## Project structure

```
pcod-working-project/
├── README.md                  # this file
├── docker-compose.yml         # MySQL + backend + frontend
├── data/
│   ├── pcos_dataset.csv       # 541 rows × 45 columns
│   ├── PCOS_data_without_infertility.xlsx   # original Kaggle file
│   ├── SCHEMA.md              # dataset provenance and columns
│   ├── LARGE_DATASET.md       # larger-dataset search and verdict
│   └── sample_patients.json   # 3 sample patients for testing
├── backend/                   # Java + Spring Boot (Maven), port 8080
└── frontend/                  # React + Vite, port 5173
```

See `backend/README.md` for the API, method and test details, and `frontend/README.md` for the UI.

## Dataset

- **Source:** Kaggle dataset "Polycystic Ovary Syndrome (PCOS)" by Prasoon Kottarathil — https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos (data from 10 hospitals in Kerala, India). File used: `PCOS_data_without_infertility.xlsx`, sheet `Full_new`.
- **541 patients × 45 columns.** Target `PCOS (Y/N)`: 0 = no PCOS (364, 67.3%), 1 = PCOS (177, 32.7%). After dropping 2 ID columns and 1 junk column, **41 input features**. Only 2 cells are genuinely missing (median-imputed). Full column list: `data/SCHEMA.md`.
- **Larger datasets:** a search across Kaggle, UCI, GitHub, IEEE DataPort, Zenodo and PhysioNet found no genuine larger real-patient PCOS dataset. The 2,000-row and 2,539-row Kaggle files are augmented copies of the same 541 patients, so they were not used — merging them would leak duplicates across train/test. Details: `data/LARGE_DATASET.md`.

## Results (held-out test set, n = 108: 73 no-PCOS / 35 PCOS)

Stratified 80/20 split, seed 42 (train 433 / test 108). Preprocessing was fitted on the training set only (median imputation → 1st/99th-percentile clipping → standardisation, re-fitted per CV fold). Hyperparameters and thresholds were picked on 5-fold train-only cross-validation; the test set was evaluated once. Full details (ROC-AUC, PR-AUC, Brier score, confusion matrices, bootstrap 95% CI, thresholds): `backend/src/main/resources/metrics.json` and `backend/src/main/resources/models/MODEL_CARD.md`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TP/FP/FN/TN | 95% CI |
|---|---|---|---|---|---|---|---|
| **Soft Voting Ensemble — deployed** | **88.89%** | 89.66% | 74.29% | 81.25% | 0.927 | 26/3/9/70 | 82.4–94.4% |
| Gradient Boosting | 87.96% | 82.35% | 80.00% | 81.16% | 0.943 | 28/6/7/67 | 81.5–93.5% |
| Random Forest | 88.89% | 96.00% | 68.57% | 80.00% | 0.930 | 24/1/11/72 | 81.5–94.4% |
| Bagging | 87.04% | 86.21% | 71.43% | 78.13% | 0.900 | 25/4/10/69 | 80.6–92.6% |
| AdaBoost | 87.04% | 92.00% | 65.71% | 76.67% | 0.919 | 23/2/12/71 | 79.6–92.6% |
| Gradient Boosting — Screening (23 non-invasive features) | 64.81% | 47.83% | 94.29% | 63.46% | 0.879 | 33/36/2/37 | 55.6–74.1% |

Baselines (same split/CV): Majority class 67.59% · Logistic Regression 85.19% · Decision Tree 82.41% · SVM (Linear) 88.89% · KNN 84.26%.

**Note on accuracy:** the 99% target was not reached — the best measured test accuracy is 88.89% (train CV 91.92%). With only 108 test rows the 95% CI is wide (82.4–94.4%), so 86% vs 89% is a difference of 2–3 patients. The screening model trades precision for recall (threshold 0.08, chosen for ≥90% out-of-fold recall — it catches 94.29% of cases but flags about half of healthy patients; see MODEL_CARD.md). The shipped models were trained on the 433-row train split only, so the reported test metrics describe the exact model being served.

## How to run

### Option A — Docker Compose

```bash
cd pcod-working-project
# First create a .env file with MYSQL_ROOT_PASSWORD and ADMIN_KEY
# (both are required — the stack will not start without them).
docker compose up --build
# Frontend: http://localhost:5173   Backend: http://localhost:8080
```

### Option B — IntelliJ IDEA + MySQL (for the viva demo)

1. Install Java 25 (Temurin 25 LTS), Maven, Node.js and MySQL Server 8.x; start MySQL.
2. Set the credentials (never hard-code a real password):
   ```bash
   export MYSQL_URL='jdbc:mysql://localhost:3306/pcod_db?createDatabaseIfNotExist=true&useSSL=false&serverTimezone=UTC'
   export MYSQL_USERNAME=root
   export MYSQL_PASSWORD='your-password'
   export ADMIN_KEY='your-admin-key'   # required for POST /api/retrain
   ```
   The `pcod_db` database and the `prediction_records` table are created automatically (`schema.sql`, `ddl-auto=none`).
3. Backend — open the project root in IntelliJ IDEA and run the **Run Full Project** configuration (runs backend + frontend), or from a terminal:
   ```bash
   cd backend
   mvn clean package
   java -jar target/pcod-prediction-backend-1.0.0.jar   # port 8080
   ```
4. Frontend:
   ```bash
   cd frontend
   npm install
   npm run dev    # http://localhost:5173
   ```

> Without a database the backend can still serve predictions (start it with the JPA auto-configuration excluded — see `backend/README.md`); history endpoints then return a clear "database not available" message instead of breaking predictions.

## API (base: http://localhost:8080)

Swagger UI: http://localhost:8080/swagger-ui.html

| Method & path | What it does |
|---|---|
| `GET /api/health` | `{"status":"UP","service":"PCOD Prediction API"}` |
| `GET /api/features` | The 41 features (`key, label, min, max, default, type, screening, required`) |
| `GET /api/models` | All models with measured metrics, confusion matrices, ROC curves, thresholds |
| `GET /api/models/{name}/importance` | Top feature importances (Random Forest / Gradient Boosting / screening) |
| `POST /api/predict[?mode=full\|screening]` | Predict. 11 non-invasive fields are required (age, bmi, cycle, cycleLength, weightGain, hairGrowth, skinDarkening, hairLoss, pimples, fastFood, regularExercise); missing any → HTTP 400 listing them. Other missing fields are median-imputed and listed in `imputedFields` / `imputedCount`. Values outside a feature's plausible range → HTTP 400 (they are rejected, not silently clipped). POST /api/predict is rate-limited to 120 requests/minute per IP. `confidence` is the probability of the predicted class. Optional `patientName` / `patientId` in the body are stored with the history record (never model inputs). Empty `{}` → 400. |
| `GET /api/predictions?limit=N` | Prediction history, latest first (needs MySQL + `X-Admin-Key` header; 401 without/wrong key, 503 if `ADMIN_KEY` is unset) |
| `GET /api/predictions/{id}` | One saved prediction (needs MySQL + `X-Admin-Key`) |
| `PATCH /api/predictions/{id}/outcome` | Set a doctor-verified outcome (`{"verifiedOutcome": 0\|1}`) — retraining uses only verified rows with no imputed fields. Needs the `X-Admin-Key` header. |
| `POST /api/retrain` | Start a retrain in the background — returns **202 Accepted** with `{jobId, status}` (header `X-Admin-Key`; 409 if a retrain is already running) |
| `GET /api/retrain/status` | Retrain job status (`RUNNING` / `DONE` / `FAILED` + metrics summary). Needs the `X-Admin-Key` header. |

### Example — predict (sample patient 2, a true PCOS case)

```bash
curl -X POST http://localhost:8080/api/predict -H 'Content-Type: application/json' -d '{"age":33,"weightKg":68.8,"heightCm":165,"bmi":25.27,"bloodGroup":11,"pulseRate":72,"rr":18,"hb":11.8,"cycle":2,"cycleLength":5,"marriageYears":10,"pregnant":1,"abortions":0,"betaHcg1":494.08,"betaHcg2":494.08,"fsh":5.54,"lh":0.88,"fshLh":6.3,"hip":40,"waist":36,"waistHipRatio":0.9,"tsh":2.54,"amh":6.63,"prl":10.52,"vitD3":49.7,"prg":0.36,"rbs":84,"weightGain":0,"hairGrowth":0,"skinDarkening":0,"hairLoss":1,"pimples":1,"fastFood":1,"regularExercise":0,"bpSystolic":120,"bpDiastolic":80,"follicleL":13,"follicleR":15,"avgFollicleSizeL":18,"avgFollicleSizeR":20,"endometrium":10}'
```

Response (verified live):

```json
{"prediction":1,"label":"High risk — refer for testing","probability":0.8493,"confidence":0.8493,"model":"Soft Voting Ensemble","modelAccuracy":0.8889,"modelAccuracyCiLow":0.8241,"modelAccuracyCiHigh":0.9444,"modelAccuracyNote":"Test accuracy of Soft Voting Ensemble on the 108-row held-out test set (95% CI 82.4%–94.4%); it describes the model, not this individual prediction.","mode":"full","threshold":0.5,"imputedCount":0,"imputedOf":41,"imputedFields":[]}
```

Screening mode (`?mode=screening`) on the same patient returns `1` (probability 0.1033 > threshold 0.08) — the recall-target threshold catches this case from the non-invasive profile alone, at the cost of flagging about half of healthy patients. That is the trade-off of screening.

### Tests

```bash
cd backend && mvn test   # 52 tests, 0 failures (2 live-MySQL tests skip when MySQL is unreachable)
cd frontend && npm run build   # 0 errors
```

## Privacy

Prediction records are stored only in the local MySQL database (`prediction_records`, with a nullable doctor-verified `verified_outcome`). The form collects an optional patient name and ID — they are stored only with the history record, never sent to the model and never used in training; leave them blank and nothing identifying is stored. Each record also carries the model version that made the prediction, so a retrain is auditable. Any real deployment would need patient consent, anonymisation, and compliance with India's DPDP Act 2023 — see `backend/README.md`.

## Sources

- Dataset: Kaggle, Prasoon Kottarathil, "Polycystic Ovary Syndrome (PCOS)" (see `data/SCHEMA.md`).
- Smile 6.3.0 (https://haifengl.github.io/smile/) — Random Forest, Gradient Boosting, AdaBoost, Bagging, Logistic Regression, SVM, KNN.
- Spring Boot 4.1 / Spring Data JPA / springdoc-openapi 3.2.0, React 18 + Vite, MySQL 8 — official documentation.

## Limitations

- 541 patients is a small dataset; the 99% accuracy target was not reached (best measured test accuracy 88.89%, 95% CI 82.4–94.4%). A larger verified labelled dataset would be needed to do better — none compatible was publicly available.
- The screening (non-invasive) model catches 94.29% of cases at its recall-target threshold but flags about half of healthy patients (precision 47.83%, accuracy 64.81%).
- Single-centre region (10 hospitals in Kerala) and a single dataset — no external validation cohort yet.
- The API is a screening aid, not a diagnosis; a clinician's judgement is required for any real decision.
