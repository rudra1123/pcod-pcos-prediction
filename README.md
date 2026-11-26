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
- **541 patients × 45 columns.** Target `PCOS (Y/N)`: 0 = no PCOS (364, 67.3%), 1 = PCOS (177, 32.7%). After dropping 2 ID columns and 1 junk column, **41 input features**. Only 2 cells are genuinely missing (mean-imputed). Full column list: `data/SCHEMA.md`.
- **Larger datasets:** a search across Kaggle, UCI, GitHub, IEEE DataPort, Zenodo and PhysioNet found no genuine larger real-patient PCOS dataset. The 2,000-row and 2,539-row Kaggle files are augmented copies of the same 541 patients, so they were not used — merging them would leak duplicates across train/test. Details: `data/LARGE_DATASET.md`.

## Results (held-out test set, n = 108: 73 no-PCOS / 35 PCOS)

Stratified 80/20 split, seed 42 (train 433 / test 108). Preprocessing was fitted on the training set only (mean imputation → 1st/99th-percentile clipping → standardisation, re-fitted per CV fold). Hyperparameters and thresholds were picked on 5-fold train-only cross-validation; the test set was evaluated once. Full details (ROC-AUC, PR-AUC, Brier score, confusion matrices, bootstrap 95% CI, thresholds): `backend/src/main/resources/metrics.json` and `backend/src/main/resources/models/MODEL_CARD.md`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TP/FP/FN/TN | 95% CI |
|---|---|---|---|---|---|---|---|
| **Soft Voting Ensemble — deployed** | **88.89%** | 92.59% | 71.43% | 80.65% | 0.927 | 25/2/10/71 | 82.4–94.4% |
| Gradient Boosting | 87.96% | 84.38% | 77.14% | 80.60% | 0.945 | 27/5/8/68 | 81.5–93.5% |
| Random Forest | 87.96% | 95.83% | 65.71% | 77.97% | 0.930 | 23/1/12/72 | 80.6–93.5% |
| Bagging | 87.04% | 86.21% | 71.43% | 78.13% | 0.900 | 25/4/10/69 | 80.6–92.6% |
| AdaBoost | 86.11% | 81.25% | 74.29% | 77.61% | 0.928 | 26/6/9/67 | 78.7–91.7% |
| Gradient Boosting — Screening (23 non-invasive features) | 82.41% | 72.22% | 74.29% | 73.24% | 0.879 | 26/10/9/63 | 75.0–88.9% |

Baselines (same split/CV): Majority class 67.59% · Logistic Regression 85.19% · Decision Tree 82.41% · SVM (Linear) 87.96% · KNN 84.26%.

**Note on accuracy:** the 99% target was not reached — the best measured test accuracy is 88.89% (train CV 91.92%). With only 108 test rows the 95% CI is wide (82.4–94.4%), so 86% vs 89% is a difference of 2–3 patients. The screening model trades precision for recall (threshold 0.30, tuned on CV only). The shipped models were trained on the 433-row train split only, so the reported test metrics describe the exact model being served.

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
| `POST /api/predict[?mode=full\|screening]` | Predict. 11 non-invasive fields are required (age, bmi, cycle, cycleLength, weightGain, hairGrowth, skinDarkening, hairLoss, pimples, fastFood, regularExercise); missing any → HTTP 400 listing them. Other missing fields are mean-imputed and listed in `imputedFields` / `imputedCount`. `confidence` is the probability of the predicted class. Empty `{}` → 400. |
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
{"prediction":1,"label":"PCOS Detected","probability":0.8453,"confidence":0.8453,"model":"Soft Voting Ensemble","modelAccuracy":0.8889,"modelAccuracyCiLow":0.8241,"modelAccuracyCiHigh":0.9444,"modelAccuracyNote":"Test accuracy of Soft Voting Ensemble on the 108-row held-out test set (95% CI 82.4%–94.4%); it describes the model, not this individual prediction.","mode":"full","threshold":0.55,"imputedCount":0,"imputedOf":41,"imputedFields":[]}
```

Screening mode (`?mode=screening`) on the same patient returns `0` (probability 0.1033 < threshold 0.30) — its non-invasive profile is mild and the case is only visible via follicle counts / AMH. That is the trade-off of screening.

### Tests

```bash
cd backend && mvn test   # 52 tests, 0 failures (2 live-MySQL tests skip when MySQL is unreachable)
cd frontend && npm run build   # 0 errors
```

## Privacy

Prediction records are stored only in the local MySQL database (`prediction_records`, with a nullable doctor-verified `verified_outcome`). No patient-identifying information is stored. Any real deployment would need patient consent, anonymisation, and compliance with India's DPDP Act 2023 — see `backend/README.md`.

## Sources

- Dataset: Kaggle, Prasoon Kottarathil, "Polycystic Ovary Syndrome (PCOS)" (see `data/SCHEMA.md`).
- Smile 6.3.0 (https://haifengl.github.io/smile/) — Random Forest, Gradient Boosting, AdaBoost, Bagging, Logistic Regression, SVM, KNN.
- Spring Boot 4.1 / Spring Data JPA / springdoc-openapi 3.2.0, React 18 + Vite, MySQL 8 — official documentation.

## Limitations

- 541 patients is a small dataset; the 99% accuracy target was not reached (best measured test accuracy 88.89%, 95% CI 82.4–94.4%). A larger verified labelled dataset would be needed to do better — none compatible was publicly available.
- The screening (non-invasive) model is less accurate (82.41%) and misses cases that need lab/ultrasound findings.
- Single-centre region (10 hospitals in Kerala) and a single dataset — no external validation cohort yet.
- The API is a screening aid, not a diagnosis; a clinician's judgement is required for any real decision.
