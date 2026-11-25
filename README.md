# Cognitive Analytics in Healthcare — PCOD Prediction (Working Project)

A full working web application that predicts Polycystic Ovarian Disorder (PCOD)
from patient clinical attributes using ensemble machine learning.

- **Team:** Priyansh Kr. Yadav (2300970130098), Rudra Pratap Singh (2300970130107),
  Rajneesh Pal (2300970130103)
- **Guide:** Mrs. Rashi Chaudhary
- **College:** Galgotias College of Engineering and Technology, Greater Noida —
  Dr. APJ Abdul Kalam Technical University, Lucknow
- **Stack:** Frontend HTML, CSS, JavaScript, React (Vite) · Backend Java 17 +
  Spring Boot 3.2.5 · Database MySQL 8.x · IDE IntelliJ IDEA · ML library
  Smile 3.1.1 (Java — no Python anywhere in this project)

> Educational demo — a screening aid, not a medical diagnosis. Consult a doctor.

## Project structure

```
pcod-working-project/
├── README.md                  # this file
├── docker-compose.yml         # one-command start: MySQL + backend + frontend
├── data/
│   ├── pcos_dataset.csv       # verified dataset, 541 rows × 45 columns
│   ├── PCOS_data_without_infertility.xlsx   # original Kaggle file
│   ├── SCHEMA.md              # dataset provenance, columns, missing values
│   ├── LARGE_DATASET.md       # larger-dataset search: what was checked, verdict
│   └── sample_patients.json   # 3 real sample patients for testing
├── backend/                   # Java + Spring Boot (Maven), port 8080
│   ├── pom.xml
│   ├── src/main/java/com/pcod/
│   │   ├── PcodApplication.java
│   │   ├── controller/        # REST API
│   │   ├── db/                # PredictionRecord entity + repository (MySQL)
│   │   ├── ml/                # preprocessing, ModelStore, TrainingService
│   │   └── trainer/ModelTrainer.java   # reproducible training (CLI)
│   ├── src/main/resources/
│   │   ├── application.properties
│   │   ├── metrics.json       # measured metrics for every model (run 8)
│   │   ├── models/            # trained Smile models (*.ser) + MODEL_CARD.md
│   │   ├── preprocessing.json # fitted imputer means, clip bounds, scaler
│   │   └── schema.sql         # prediction_records table
│   └── README.md              # backend details, API reference, MySQL setup
└── frontend/                  # React + Vite, port 5173
    ├── src/App.jsx            # prediction form, result card, metrics charts
    ├── src/features.js        # 41-feature fallback copy + sample patients
    └── README.md              # frontend details
```

## Dataset

- **Source:** Kaggle dataset "Polycystic Ovary Syndrome (PCOS)" by Prasoon
  Kottarathil — https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos
  (data from 10 hospitals in Kerala, India). File used:
  `PCOS_data_without_infertility.xlsx`, sheet `Full_new`.
- **Verified on disk: 541 patients × 45 columns** — exactly as specified.
  Target `PCOS (Y/N)`: 0 = no PCOS (364, 67.3%), 1 = PCOS (177, 32.7%).
  After dropping 2 ID columns and 1 junk trailing column, **41 input features**.
  Only 2 genuinely missing cells (mean-imputed). Full column list and
  provenance: `data/SCHEMA.md`.
- **Larger datasets:** a dedicated search (Kaggle, UCI, GitHub, IEEE DataPort,
  Zenodo, PhysioNet) found **no genuine larger real-patient PCOS dataset**.
  The 2,000-row and 2,539-row Kaggle files are augmented copies of the same
  541 patients (proven by repeated patient IDs and near-duplicate rows) and
  were rejected — merging them would leak duplicates across train/test. A
  10,000-row file exists but its provenance is unverified, so it was not used.
  Details: `data/LARGE_DATASET.md`. Training therefore uses the verified
  541-row dataset only; no patient counts are inflated anywhere.

## Measured results (held-out test set, n = 108: 73 no-PCOS / 35 PCOS)

Stratified 80/20 split, seed 42 (train 433 / test 108). Preprocessing fitted
on the training set only (mean imputation → 1st/99th-percentile clipping →
standardisation, re-fitted per CV fold). Hyperparameters and thresholds were
chosen on 5-fold train-only cross-validation; the test set was evaluated once.
Full per-model details (ROC-AUC, PR-AUC, Brier score, confusion matrix,
bootstrap 95% CI, ROC-curve points, thresholds): `backend/src/main/resources/metrics.json`
and `backend/src/main/resources/models/MODEL_CARD.md`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TP/FP/FN/TN | 95% CI |
|---|---|---|---|---|---|---|---|
| **Soft Voting Ensemble — deployed** | **88.89%** | 89.66% | 74.29% | 81.25% | 0.925 | 26/3/9/70 | 82.4–94.4% |
| Gradient Boosting (best single) | 88.89% | 87.10% | 77.14% | 81.82% | 0.933 | 27/4/8/69 | 82.4–94.4% |
| Random Forest | 87.04% | 92.00% | 65.71% | 76.67% | 0.925 | 23/2/12/71 | 79.6–93.5% |
| Bagging | 87.04% | 86.21% | 71.43% | 78.13% | 0.900 | 25/4/10/69 | 80.6–92.6% |
| AdaBoost | 86.11% | 85.71% | 68.57% | 76.19% | 0.917 | 24/4/11/69 | 78.7–92.6% |
| Gradient Boosting — Screening (23 non-invasive features) | 82.41% | 69.05% | **82.86%** | 75.32% | 0.886 | 29/13/6/60 | 75.0–88.9% |

Baselines (same split/CV): Majority class 67.59% · Logistic Regression 85.19% ·
Decision Tree 82.41% · SVM (Linear) 83.33% · KNN 84.26%.

**Honest note on accuracy:** the 99% target was **not** reached. The measured
best test accuracy is 88.89% (train CV 93.07%). With only 108 test rows, the
95% confidence interval is wide (82.4–94.4%) and 86% vs 89% is a difference of
2–3 patients — not significant. The screening model trades precision for
recall (threshold 0.25, tuned on CV only): recall rises from 60.00% to 82.86%.
The screening model misses some cases that are only visible via follicle
counts / AMH (e.g. sample patient 2) — documented in `backend/README.md`.

**No refit on all 541 rows — by design:** the shipped models were trained on
the 433-row train split only, so the reported test metrics describe the exact
artefact being served. Refitting on all 541 rows after evaluation would use
~25% more data but then the test metrics would no longer describe the shipped
model — see `backend/README.md` (Limitations) and `MODEL_CARD.md`.

**Follicle counts are near-diagnostic:** antral follicle counts differ
sharply between the classes (follicleL mean 9.79 with PCOS vs 4.35 without;
follicleR 10.76 vs 4.64) — Rotterdam-adjacent morphology features. The
23-feature screening model (no follicle counts, no AMH, no labs) exists
precisely to answer that, and still reaches 82.41% test accuracy with
82.86% recall.

## How to run

### Option A — Docker Compose (one command)

```bash
cd pcod-working-project
# Required first: create a .env file with MYSQL_ROOT_PASSWORD and
# ADMIN_KEY (both are required — the stack will not start without them).
docker compose up --build
# Frontend: http://localhost:5173   Backend: http://localhost:8080
```

MySQL's port 3306 is **not** published to the host (the backend reaches it
over the Compose network; use `docker exec -it pcod-mysql mysql -uroot -p`
for host access). Retrained models are kept in the named volume
`pcod-resources` (`/app/resources` in the backend container), so they
survive container recreation.

### Option B — IntelliJ IDEA + MySQL (recommended for the viva demo)

1. Install Java 17, Maven, Node.js, and MySQL Server 8.x; start MySQL.
2. Set the database credentials (never hard-code a real password):
   ```bash
   export MYSQL_URL='jdbc:mysql://localhost:3306/pcod_db?createDatabaseIfNotExist=true&useSSL=false&serverTimezone=UTC'
   export MYSQL_USERNAME=root
   export MYSQL_PASSWORD='your-password'
   export ADMIN_KEY='your-admin-key'   # required for POST /api/retrain
   ```
   The `pcod_db` database and the `prediction_records` table are created
   automatically (`schema.sql`, `ddl-auto=none`).
3. Backend — open `backend/` in IntelliJ IDEA as a Maven project (JDK 17)
   and run `PcodApplication`, or from a terminal:
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

> Prediction-only run (no database): if MySQL is not available, the backend
> can still serve predictions with JPA excluded:
> `java -jar target/pcod-prediction-backend-1.0.0.jar --spring.autoconfigure.exclude=org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration,org.springframework.boot.autoconfigure.orm.jpa.HibernateJpaAutoConfiguration,org.springframework.boot.autoconfigure.data.jpa.JpaRepositoriesAutoConfiguration`
> — history endpoints then return a clear "database not available" message
> instead of breaking predictions. Full details in `backend/README.md`.

## API (base: http://localhost:8080)

API docs (Swagger UI): http://localhost:8080/swagger-ui.html

| Method & path | What it does |
|---|---|
| `GET /api/health` | `{"status":"UP","service":"PCOD Prediction API"}` |
| `GET /api/features` | The 41 features (`key, label, min, max, default, type, screening, required`) |
| `GET /api/models` | All models with measured metrics, confusion matrices, ROC curves, thresholds |
| `GET /api/models/{name}/importance` | Top feature importances (Random Forest / Gradient Boosting / screening) |
| `POST /api/predict[?mode=full\|screening]` | Predict. 11 non-invasive fields are required (age, bmi, cycle, cycleLength, weightGain, hairGrowth, skinDarkening, hairLoss, pimples, fastFood, regularExercise); missing any → HTTP 400 listing them. Other missing fields are mean-imputed and listed in `imputedFields` / counted in `imputedCount`. `confidence` is the probability of the predicted class. `modelAccuracy` is the model's test accuracy (108-row test set, with 95% CI + note) — it describes the model, not this prediction. A `warning` is added when 21+ fields were imputed. Empty `{}` → 400. |
| `GET /api/predictions?limit=N` | Prediction history, latest first (needs MySQL + `X-Admin-Key` header; 401 without/wrong key, 503 if `ADMIN_KEY` is unset) |
| `GET /api/predictions/{id}` | One saved prediction (needs MySQL + `X-Admin-Key`, same 401/503 rules) |
| `PATCH /api/predictions/{id}/outcome` | Set a doctor-verified outcome (`{"verifiedOutcome": 0\|1}`) — retraining uses only verified rows, and only verified rows with no imputed fields. Needs the `X-Admin-Key` header. |
| `POST /api/retrain` | Start a retrain in the background — returns **202 Accepted** with `{jobId, status: "RUNNING", statusUrl}` (header `X-Admin-Key: $ADMIN_KEY`; 409 if a retrain is already running) |
| `GET /api/retrain/status` / `GET /api/retrain/{jobId}` | Retrain job status (`RUNNING` / `DONE` / `FAILED` + metrics summary). Needs the `X-Admin-Key` header. |

### Example — predict (sample patient 2, a true PCOS case)

```bash
curl -X POST http://localhost:8080/api/predict -H 'Content-Type: application/json' -d '{"age":33,"weightKg":68.8,"heightCm":165,"bmi":25.27,"bloodGroup":11,"pulseRate":72,"rr":18,"hb":11.8,"cycle":2,"cycleLength":5,"marriageYears":10,"pregnant":1,"abortions":0,"betaHcg1":494.08,"betaHcg2":494.08,"fsh":5.54,"lh":0.88,"fshLh":6.3,"hip":40,"waist":36,"waistHipRatio":0.9,"tsh":2.54,"amh":6.63,"prl":10.52,"vitD3":49.7,"prg":0.36,"rbs":84,"weightGain":0,"hairGrowth":0,"skinDarkening":0,"hairLoss":1,"pimples":1,"fastFood":1,"regularExercise":0,"bpSystolic":120,"bpDiastolic":80,"follicleL":13,"follicleR":15,"avgFollicleSizeL":18,"avgFollicleSizeR":20,"endometrium":10}'
```

Response (verified live):

```json
{"prediction":1,"label":"PCOS Detected","probability":0.8566,"confidence":0.8566,"model":"Soft Voting Ensemble","modelAccuracy":0.8889,"modelAccuracyCiLow":0.8241,"modelAccuracyCiHigh":0.9444,"modelAccuracyNote":"Test accuracy of Soft Voting Ensemble on the 108-row held-out test set (95% CI 82.4%–94.4%); it describes the model, not this individual prediction.","mode":"full","threshold":0.5,"imputedCount":0,"imputedOf":41,"imputedFields":[]}
```

Sample patient 1 (true label 0) predicts `0` with probability `0.099`.
Screening mode (`?mode=screening`) on sample 2 returns `0` (probability
0.1214 < threshold 0.25) — its non-invasive profile is mild; the case is only
visible via follicle counts / AMH. That is the honest trade-off of screening.

### Tests

```bash
cd backend && mvn test   # 52 tests, 0 failures (2 live-MySQL tests skip when MySQL is unreachable)
cd frontend && npm run build   # 0 errors
```

## Privacy

Prediction records are stored only in the local MySQL database
(`prediction_records`, with a nullable doctor-verified `verified_outcome`).
No patient-identifying information is stored. Any real deployment would need
patient consent, anonymisation, and compliance with India's DPDP Act 2023 —
see `backend/README.md`.

## Sources consulted

- Dataset: Kaggle, Prasoon Kottarathil, "Polycystic Ovary Syndrome (PCOS)"
  (see `data/SCHEMA.md`).
- Java ML: Smile 3.1.1 (https://haifengl.github.io/smile/) — Random Forest,
  Gradient Boosting, AdaBoost, Bagging, Logistic Regression, SVM, KNN.
- Spring Boot 3.2 / Spring Data JPA / springdoc-openapi, React 18 + Vite,
  MySQL 8 — official documentation.

## Limitations (honest)

- 541 patients is a small dataset; the 99% accuracy target was not reached
  (best measured test accuracy 88.89%, 95% CI 82.4–94.4%). A larger verified labelled dataset
  would be needed to do better — none compatible was publicly available.
- The screening (non-invasive) model is less accurate (82.41%) and misses
  cases that need lab/ultrasound findings.
- In the development VM used to build this, Java could not open TCP
  connections to MySQL, so the live app→MySQL write path and Docker Compose
  could not be live-tested there: the schema was verified against real MySQL
  8.0.46 with the native client, the JPA layer is covered by tests, and the
  live jar was tested prediction-only. Run `docker compose up --build` or the
  IntelliJ steps above on a normal machine to verify the full path.
