# PCOD Prediction Backend

Java 17 + Spring Boot 3.2.5 REST API for PCOD/PCOS prediction, built for the
final-year project **Cognitive Analytics in Healthcare**. It serves six
models trained in Java with the Smile machine-learning library
(`com.github.haifengl:smile-core`, version **3.1.1**, verified on Maven
Central): Random Forest, Bagging, AdaBoost, Gradient Boosting, a soft-voting
ensemble of those four, and a screening Gradient Boosting model trained on
non-invasive features only. Five baselines (majority class, Logistic
Regression, single Decision Tree, linear SVM, KNN) are trained and reported
the same way for comparison. Predictions are stored in MySQL
(`prediction_records` table); the API is documented with Swagger/OpenAPI.

## Measured results (held-out test set, 108 patients)

Test set: 73 patients without PCOS, 35 with PCOS. All numbers below are read
from `src/main/resources/metrics.json`, produced by the trainer
(`ModelTrainer`, run 2026-11-25) — measured values, not targets. The 95% CI
is a bootstrap interval for test accuracy (1000 resamples, seed 42).

| Model | Test accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Train CV | TP/FP/FN/TN | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| **Soft Voting Ensemble — deployed** | **88.89%** | 89.66% | 74.29% | 81.25% | 0.925 | 0.879 | 0.0992 | 93.07% | 26/3/9/70 | 82.4–94.4% |
| Gradient Boosting | 88.89% | 87.10% | 77.14% | 81.82% | 0.933 | 0.890 | 0.0963 | 92.15% | 27/4/8/69 | 82.4–94.4% |
| Random Forest | 87.04% | 92.00% | 65.71% | 76.67% | 0.925 | 0.879 | 0.1028 | 90.76% | 23/2/12/71 | 79.6–93.5% |
| Bagging | 87.04% | 86.21% | 71.43% | 78.13% | 0.900 | 0.848 | 0.1075 | 89.15% | 25/4/10/69 | 80.6–92.6% |
| AdaBoost | 86.11% | 85.71% | 68.57% | 76.19% | 0.917 | 0.864 | 0.1811 | 89.38% | 24/4/11/69 | 78.7–92.6% |
| Gradient Boosting (Screening) | 82.41% | 69.05% | 82.86% | 75.32% | 0.886 | 0.777 | 0.1331 | 82.45% | 29/13/6/60 | 75.0–88.9% |

The deployed model is the **Soft Voting Ensemble** (average P(PCOS) of the
four tuned models): it had the best 5-fold cross-validation accuracy on the
training split (93.07%), and it ties Gradient Boosting on the held-out test
set (88.89%). The model was chosen by train CV, not by test accuracy.
Gradient Boosting is the best single model (test 88.89%, CV 92.15%).

**Read the small differences honestly.** With only 108 test rows, one
patient is worth 0.93 percentage points, so 86% vs 89% is a difference of
2–3 patients and is **not significant** — the bootstrap confidence intervals
overlap heavily (e.g. AdaBoost 78.7–92.6% vs Gradient Boosting 82.4–94.4%).
The honest claim is that all four ensembles and the voting ensemble sit in
the same high-80s band, clearly above the baselines below.

### Baselines (same split, same train-only CV protocol)

| Model | Test accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Train CV | TP/FP/FN/TN | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| Majority Class (always "no PCOS") | 67.59% | 0.00% | 0.00% | 0.00% | 0.500 | 0.324 | 0.3241 | 67.21% | 0/0/35/73 | 58.3–75.9% |
| Logistic Regression | 85.19% | 82.76% | 68.57% | 75.00% | 0.880 | 0.860 | 0.1151 | 87.07% | 24/5/11/68 | 77.8–91.7% |
| Decision Tree (single) | 82.41% | 80.77% | 60.00% | 68.85% | 0.819 | 0.696 | 0.1428 | 84.76% | 21/5/14/68 | 75.0–89.8% |
| SVM (Linear) | 83.33% | 79.31% | 65.71% | 71.88% | 0.788 | 0.632 | 0.1667 | 86.84% | 23/6/12/67 | 75.0–89.8% |
| KNN | 84.26% | 82.14% | 65.71% | 73.02% | 0.814 | 0.716 | 0.1430 | 85.91% | 23/5/12/68 | 77.8–90.7% |

The majority-class baseline scores 67.59% by predicting the majority class
every time — any useful model must beat it, and all of them do. The linear
SVM outputs hard decisions only (decision-function sign, no calibrated
probabilities), so its ROC-AUC/Brier reflect 0/1 scores; see Method.

### Decision thresholds (tuned on train CV, never on the test set)

For each deployed model the decision threshold was chosen from
out-of-fold train probabilities (5-fold CV on train only), maximising F1,
with ties broken toward the lower threshold (higher recall). Recall matters
most for a screening tool: a missed case costs more than a false alarm.

| Model | Tuned threshold | OOF precision / recall / F1 | Test at tuned threshold (acc / P / R / F1) | Test at 0.5 (acc / P / R / F1) |
|---|---|---|---|---|
| Soft Voting Ensemble (deployed) | 0.50 | 94.44% / 83.80% / 88.81% | 88.89% / 89.66% / 74.29% / 81.25% | 88.89% / 89.66% / 74.29% / 81.25% |
| Gradient Boosting (Screening) | 0.25 | 64.80% / 81.69% / 72.27% | 82.41% / 69.05% / 82.86% / 75.32% | 79.63% / 72.41% / 60.00% / 65.62% |

For the deployed ensemble the tuned threshold came out at 0.50 — the
default was already optimal on out-of-fold train data, so both columns are
identical. For the screening model the tuned threshold (0.25) is the whole
point: compared with 0.5 it trades 3.4 points of precision for **22.9
points of recall** (60.00% → 82.86%) and gains 2.8 points of accuracy.
The API applies the tuned threshold automatically (`threshold` in the
predict response); the 0.5 column is shown so the trade-off is visible.

### Screening model

`POST /api/predict?mode=screening` uses a Gradient Boosting model trained
on **23 non-invasive features only** — age, body measurements (weight,
height, BMI, hip, waist, waist:hip ratio), pulse rate, respiratory rate,
cycle regularity and length, marriage duration, pregnancy/abortions, weight
gain, hair growth, skin darkening, hair loss, pimples, fast food, regular
exercise, blood pressure. **No** follicle counts, **no** AMH/FSH/LH/TSH/PRL
or other hormone/blood labs, **no** ultrasound measures. It is the model
for early detection before a doctor visit, when labs and ultrasound are not
available yet. It scores 82.41% test accuracy with 82.86% recall at its
tuned threshold — lower accuracy than the full 41-feature model, as
expected from fewer features, but with the highest recall of any model
here. The full model remains the default (`mode=full`).

**About the 99% target:** it was not reached. With 541 patients the test
set has only 108 rows, so one patient is worth 0.93 percentage points and
the test accuracy carries roughly ±3 percentage points of sampling
variation; train CV (82–93%) and test accuracy (82–89%) agree that the
honest level for this data with these models is high-80s to low-90s, not
99%. An earlier internal figure of 91.7% accuracy for Gradient Boosting on
this dataset was reported before this build; the independently measured
held-out result here is 88.89%, and 88.89% is the number to use. Reaching
99% would need substantially more verified labelled patient data (see
"Dataset" below) or a different data source — not more tuning on 541 rows.

## Dataset

`../data/pcos_dataset.csv` — **541 rows × 45 columns**, from the Kaggle
dataset by Prasoon Kottarathil, "Polycystic Ovary Syndrome (PCOS)"
(10 Kerala hospitals). Target `PCOS (Y/N)`: 0 = 364 patients (67.3%),
1 = 177 patients (32.7%). Full column list, provenance and checksums are in
`../data/SCHEMA.md`.

- Dropped before training: `Sl. No`, `Patient File No.` (identifiers) and
  `Unnamed: 44` (junk column, 539 of 541 cells empty). That leaves **41 input
  features** (see `GET /api/features`, or `FeatureSpec.java` for the exact
  clean snake_case keys, in training order). The screening model uses the 23
  non-invasive ones of these (see `FeatureSpec.SCREENING_FEATURES`).
- Missing data: only 2 cells are genuinely missing (Marriage duration: 1,
  Fast food: 1) — filled with the training mean. Two further cells are
  malformed in the source file and were found while loading: beta-HCG II =
  `1.99.` (trailing dot, read as 1.99) and AMH = `a` (stray letter, treated
  as missing and mean-imputed). Both are logged by the trainer.
- Outliers: FSH up to 5052, Vitamin D3 up to 6014.66, beta-HCG up to
  32460.97 — handled by clipping (see Method).

**Larger data — searched, not found.** A dedicated search of Kaggle, UCI,
IEEE DataPort, Zenodo, PhysioNet, Harvard Dataverse, GitHub and the
literature (full record: `../data/LARGE_DATASET.md`) found **no genuine
larger real-patient labelled tabular PCOS dataset**. The apparently larger
same-schema files (2,000 and 2,539 rows on Kaggle) are proven augmented
copies of the same 541 patients (only 530/541 distinct Patient File Nos,
near-duplicate rows, the same two typos reproduced) — merging them would
leak near-duplicates across train/test, so they were **not used**. A
10,000-row file exists but its provenance is unverified (uploaded
2026-11-21, 4 downloads, no hospital, source or paper) and it was **not
used** as real-patient training data. There is no public 50,000- or
1-lakh-patient labelled tabular PCOS dataset. Training therefore uses the
verified 541 rows only, and no larger patient count is claimed anywhere.

## Method (exactly as implemented in `TrainingService.java`)

- **Split:** stratified 80/20 train/test, seed 42. Train 433 rows
  (291 without PCOS, 142 with PCOS); test 108 rows (73 without, 35 with).
  The test split is evaluated exactly **once**, after all tuning.
- **Preprocessing, fitted on training data only:** (1) mean imputation for
  missing values; (2) outlier clipping to the 1st/99th training percentiles
  per feature; (3) standardisation, z = (x − mean) / std, with mean/std from
  the imputed + clipped training data. This is the same arithmetic as
  Smile's `Standardizer` (per-column mean and standard deviation); it is
  implemented explicitly because the fitted mean/std must be stored and
  re-applied verbatim at inference time, including for single-row
  predictions. Inside cross-validation the preprocessing is re-fitted on
  each fold's training part only.
- **Tuning (81 candidate evaluations, train CV only):** 5-fold stratified
  cross-validation on the train split, seed 42, comparing hyperparameter
  grids per model (Random Forest: 16 configs — trees 300–800, mtry 0/6,
  depth 10/20, node size 1/2/5, subsample 0.8/1.0; Bagging: 14 configs —
  100–300 bootstrap trees, depth 10/20, node size 1/5; AdaBoost: 15 configs
  — 100–500 rounds, depth 1–3; Gradient Boosting: 16 configs — 200–800
  trees, depth 2/5, shrinkage 0.03–0.1, subsample 0.7–1.0), feature subsets
  from Random Forest importance rankings re-computed inside each CV fold
  (ranked on that fold's training part only; all 41, top 25/20/15/10),
  clipping on/off, and minority-class random oversampling
  on/off (applied to training folds only); baselines: Logistic Regression
  (lambda 0.01/0.1/1.0), single Decision Tree (4 depth/node-size combos),
  linear SVM (C 0.1/1.0/10.0), KNN (k 3/5/9/15); screening Gradient
  Boosting (6 configs on the 23 non-invasive features). Full per-candidate
  log: `metrics.json` → `cvLog`. An earlier version ranked once on the
  whole train split before CV, which made the subset CV scores slightly
  optimistic (the test split was never involved in either version); the
  ranking is now re-computed per fold. The all-41 subset still wins for
  every model, so no deployed number changed.
- **Bagging:** Smile 3.1.1 ships no Bagging classifier, so it is implemented
  here from its definition (`BaggingClassifier.java`): bootstrap-sampled
  Smile decision trees with majority vote; the vote fraction is the
  probability.
- **Baselines:** Logistic Regression, KNN and the single Decision Tree use
  Smile directly on the same preprocessed matrices. The linear SVM needs
  labels −1/+1 (Smile's LASVM rejects 0/1) and its `predict(x, posteriori)`
  throws (hard classifier), so `TrainingService` maps the labels and wraps
  the model in `HardBinaryClassifier`, which returns the decision-function
  sign and a one-hot posterior — its ROC-AUC/Brier therefore reflect hard
  decisions, not calibrated probabilities. The majority baseline predicts 0
  for every row (probability 0).
- **Best configurations found (by train CV):**
  - Random Forest: 500 trees, mtry 6, maxDepth 20, nodeSize 1, all 41
    features, clipping on (CV 90.76%).
  - Bagging: 100 trees, maxDepth 20, nodeSize 1, all 41 features, clipping
    on (CV 89.15%).
  - AdaBoost: 300 rounds, maxDepth 3, maxNodes 8, all 41 features, clipping
    off (CV 89.38%).
  - Gradient Boosting: 500 trees, maxDepth 5, maxNodes 31, nodeSize 5,
    shrinkage 0.05, subsample 0.7, all 41 features, clipping on
    (CV 92.15%).
  - Soft Voting Ensemble: average P(PCOS) of the four tuned models
    (CV 93.07%).
  - Gradient Boosting (Screening): 800 trees, maxDepth 2, maxNodes 6,
    nodeSize 5, shrinkage 0.03, subsample 0.7, 23 non-invasive features,
    clipping on (CV 82.45%).
  - Baselines: Logistic Regression lambda 0.1 (CV 87.07%); Decision Tree
    maxDepth 5, maxNodes 31, nodeSize 1 (CV 84.76%); SVM C 0.1 (CV 86.84%);
    KNN k 3 (CV 85.91%).
  - Oversampling did not win for any model on CV and is not used in the
    final models. Feature subsets did not win for any model; all final
    models use all of their features (41 full, 23 screening).
- **Threshold tuning:** after the final fit, out-of-fold train
  probabilities (5-fold CV on train only) are scanned over thresholds
  0.05–0.95 (step 0.05); the threshold maximising F1 is kept, ties broken
  toward the lower threshold (higher recall). See the thresholds table
  above. The test set is never used to pick a threshold.
- **Extra metrics:** ROC-AUC by the Mann–Whitney statistic (tie-averaged
  ranks), PR-AUC by average precision grouped on distinct probabilities,
  Brier score, the full ROC curve points (stored per model in
  `metrics.json`, so the frontend draws a real curve and never an invented
  one), and a 95% bootstrap confidence interval for test accuracy (1000
  resamples, seed 42).
- **Top features** by importance (mean decrease in impurity) on train:
  - Random Forest (all 41): follicleR 286.9, follicleL 210.9, AMH 160.8,
    Vit D3 157.3, TSH 151.7, FSH/LH 141.7, FSH 140.0, PRL 139.0, BMI 138.1,
    LH 137.2. (Full ranking: `metrics.json` → `featureImportance`,
    also `GET /api/models/random-forest/importance`.)
  - Gradient Boosting (all 41): follicleR 1259.3, follicleL 314.0,
    hair growth 195.3, weight gain 137.7, LH 123.3.
    (`metrics.json` → `gbmImportance`,
    `GET /api/models/gradient-boosting/importance`.)
  - Screening model (23 non-invasive): skin darkening 776.2, hair growth
    417.7, cycle length 416.9, cycle regularity 386.9, weight gain 373.7.
    (`metrics.json` → `screeningImportance`,
    `GET /api/models/gradient-boosting-screening/importance`.)
  Importance values are sums over the fitted forest — the ranking is stable
  across runs, but the absolute numbers depend on the fitted trees and are
  not comparable across runs.
- **Reproducibility:** seed 42 everywhere (split, CV folds, Smile RNG,
  per-tree seeds, bootstrap sampling, threshold scan, bootstrap CI).
  One subtlety found while building: `MathEx.setSeed` is thread-local and
  Smile builds Random Forest trees on a parallel stream, so a single
  `setSeed(42)` at training start is not enough — the trainer now calls
  `MathEx.setSeed(seed)` immediately before every fit, and hyperparameter
  descriptions are sorted so the metrics file is byte-stable. Verified:
  trainer runs 3–6 produced identical numeric metrics, and the final run
  reproduces the run-6 ensemble metrics exactly (same CV and test accuracy
  for all four ensembles). Re-running the trainer on the same CSV
  reproduces the same metrics.
- **Serving:** the Spring Boot app loads the saved models at startup
  (`src/main/resources/models/*.ser`, Smile models are `Serializable`) and
  never retrains on boot. Each model is served with its own fitted
  preprocessing and feature subset, stored in `metrics.json`
  (`modelPreprocessing`). The deployed model and its threshold come from
  `metrics.json` (`bestModel`, `deployedThreshold`); the screening model
  and threshold from `screeningThreshold`.
- **Model card:** `src/main/resources/models/MODEL_CARD.md` is generated
  from `metrics.json` — every number in it comes from `metrics.json`, so
  the card cannot drift from the measured results. Regenerate it from
  `metrics.json` after retraining. It records version, training date,
  dataset, split/seed, per-model metrics + thresholds + baselines,
  limitations, intended use.

## Database (MySQL)

MySQL 8.x stores every prediction. **MySQL must be installed and running
before the backend starts** (the app fails fast at startup if it cannot
connect — predictions are the one exception: see below).

1. Install MySQL Server 8.x and start it (Ubuntu:
   `sudo apt-get install -y mysql-server && sudo service mysql start`).
2. Set a root password (or create a dedicated user) — the app never
   hard-codes one:
   ```sql
   ALTER USER 'root'@'localhost' IDENTIFIED BY 'your-password';
   ```
3. Set the environment variables (or edit `application.properties`):
   ```bash
   export MYSQL_URL='jdbc:mysql://localhost:3306/pcod_db?createDatabaseIfNotExist=true&useSSL=false&serverTimezone=UTC'
   export MYSQL_USERNAME=root
   export MYSQL_PASSWORD='your-password'
   ```
   Defaults when unset: URL as above, username `root`, empty password.
   The database `pcod_db` is created automatically on first connect
   (`createDatabaseIfNotExist=true`).
4. The table is created automatically on startup from
   `src/main/resources/schema.sql` (`CREATE TABLE IF NOT EXISTS
   prediction_records ...`). `spring.jpa.hibernate.ddl-auto=none` — the
   schema file, not Hibernate auto-DDL, owns the table.
5. Start the backend (see "How to run"). `GET /api/predictions` then lists
   history, latest first.

`PredictionRecord` (`prediction_records` table) stores the 41 input
features as JSON (`features_json`), prediction (0/1), label, probability,
confidence, model name and timestamp, plus the doctor-verification fields
below. Saving happens inside `POST /api/predict` after a successful
prediction; **a database failure never breaks the prediction response** —
it is logged and the prediction is still returned. If the database is not
available at all, the history endpoints answer 503 with a clear message
and predictions still work (the app can also be run prediction-only with
the JPA auto-configuration excluded — see "How to run").

### Verified outcomes — why retraining never learns from its own predictions

A model that retrains on its own predictions reinforces its own mistakes.
So the table has two nullable doctor-entered fields, `verified_outcome`
(0/1) and `verified_at`, set through
`PATCH /api/predictions/{id}/outcome` with body
`{"verifiedOutcome": 0}` or `{"verifiedOutcome": 1}`. **Retraining uses
only rows where `verified_outcome` is filled in** — never unverified model
predictions. `POST /api/retrain` loads those rows from the database,
appends them to the training split only (the test split never changes),
and records how many were used in `metrics.json` → `verifiedRowsUsed`
(0 in the shipped run — no verified rows existed yet).

A verified row is used only if its stored inputs are complete: each
prediction record stores which fields were mean-imputed at prediction
time (`imputed_fields_json` column, also returned as `imputedFields` in
the history JSON). **Verified rows with any imputed field are excluded
from retraining** — a doctor can verify the outcome, but the stored
feature values for the imputed fields are training averages, not the
patient's real values, so treating them as ground truth would teach the
model fake data. Rows written before this field was tracked (null) are
excluded too, for the same reason: they cannot be proven complete.

### Admin key — fail-closed protection for history and retraining

Patient history and retraining are protected by the `X-Admin-Key` header.
The key comes from the `ADMIN_KEY` environment variable
(`pcod.admin.key=${ADMIN_KEY:}`). Protected endpoints:

- `GET /api/predictions`, `GET /api/predictions/{id}` (patient health data)
- `PATCH /api/predictions/{id}/outcome` (retraining data — an open endpoint
  would let anyone poison future models with false verified outcomes)
- `POST /api/retrain`, `GET /api/retrain/status`, `GET /api/retrain/{jobId}`

Without the header, or with the wrong key, a protected endpoint returns
401. **If `ADMIN_KEY` is unset, every protected endpoint refuses with 503
(fail-closed)** — the app logs an ERROR at startup and the endpoints stay
disabled until the key is set and the app restarted. There is no
"unprotected" mode: an earlier version only warned and left retraining
open, which was a security hole. `docker-compose.yml` requires `ADMIN_KEY`
(` ${ADMIN_KEY:?set ADMIN_KEY in .env}`) so the stack cannot start without
one. Never hard-code a real secret in the code or in
`application.properties`.

Retraining is asynchronous: `POST /api/retrain` returns **202 Accepted**
immediately with `{jobId, status: "RUNNING", statusUrl}` and trains on a
background thread (about 2.5–12 minutes), so the HTTP request never times
out. Poll `GET /api/retrain/status` (or `GET /api/retrain/{jobId}`) for
`RUNNING` / `DONE` / `FAILED` plus the resulting metrics summary
(`bestModel`, `bestModelTestAccuracy`, `verifiedRowsUsed`, `generatedAt`).
A second retrain while one is running is rejected with **409**. When a
retrain finishes, the new models are loaded from the external resources
directory (see below) and served without a restart.

Retrained artefacts are written to the external resources directory
(`PCOD_RESOURCES_DIR` / `pcod.resources.dir`, default
`src/main/resources` when running from source) and **loaded from that
directory first at startup** — a restart after retraining serves the new
models instead of the ones baked into the jar. In Docker Compose the
directory is a named volume (`pcod-resources`, mounted at
`/app/resources`), so retrained artefacts survive container recreation.
If the external directory has no `metrics.json` (or is corrupt), the app
falls back to the models baked into the jar and logs a warning.

### Privacy

Prediction records are stored in the project's own MySQL database and
nowhere else. Patient data must be **anonymised before entry**, and
**consent must be obtained** for storing it — real deployment would need a
consent workflow, access control on the history endpoints, and retention
rules. This project follows India's **Digital Personal Data Protection
(DPDP) Act, 2023**: personal health data is sensitive, so collect the
minimum needed, keep it only in the local database, and do not share or
export it without consent.

## How to run

Prerequisites: **Java 17**, **Maven 3.9+**, **MySQL 8.x** (running),
**Node.js/npm** for the frontend. IntelliJ IDEA: open this `backend/`
folder as a Maven project, JDK 17, run `PcodApplication`.

```bash
cd pcod-working-project/backend

# 0. MySQL running + env vars set (see "Database (MySQL)").

# 1. Train (reads ../data/pcos_dataset.csv; writes metrics.json,
#    preprocessing.json and models/*.ser into src/main/resources).
#    Takes ~12 minutes: 5-fold CV over 81 candidate configurations.
mvn exec:java -Dexec.mainClass=com.pcod.trainer.ModelTrainer

# 2. Build + test (52 tests: 50 pass, 2 skipped when MySQL is unreachable
#    — see "Tests" below).
mvn clean package

# 3. Run (port 8080).
java -jar target/pcod-prediction-backend-1.0.0.jar
```

The API base URL is `http://localhost:8080`. CORS allows
`http://localhost:5173` (Vite) and `http://localhost:3000` (Create React App).
Swagger UI: `http://localhost:8080/swagger-ui.html`
(OpenAPI JSON: `/v3/api-docs`).

Prediction-only mode (no database, for a quick smoke test): start with the
JPA auto-configuration excluded — predictions, models, metrics, features
and importance all work; history endpoints answer 503:

```bash
java -jar target/pcod-prediction-backend-1.0.0.jar \
  --spring.autoconfigure.exclude=org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration,org.springframework.boot.autoconfigure.orm.jpa.HibernateJpaAutoConfiguration,org.springframework.boot.autoconfigure.data.jpa.JpaRepositoriesAutoConfiguration
```

### API

| Endpoint | Description |
|---|---|
| `GET /api/health` | `{"status":"UP","service":"PCOD Prediction API"}` |
| `GET /api/models` | 6 models with measured `{name, accuracy, precision, recall, f1, rocAuc, prAuc, brierScore, cvAccuracy, confusionMatrix, rocCurve, threshold, hyperparameters, nFeatures, mode, screening, ...}` |
| `GET /api/metrics` | The full `metrics.json`: models, baselines, thresholds, importances, `cvLog`, `verifiedRowsUsed` |
| `GET /api/features` | The 41 features as `{key, label, min, max, default, type, screening, required}` for the frontend form (`default` = training mean; `screening` = used by the screening model; `required` = one of the 11 required fields) |
| `GET /api/models/{name}/importance` | Top feature importances for Random Forest, Gradient Boosting and the screening model (`{model, importance: [{feature, importance}]}`); 404 for Bagging/AdaBoost/voting/baselines, which have no impurity importance |
| `POST /api/predict` | Prediction from the deployed model. Body: flat JSON with the 41 keys, or `{"features": {...}}`. Query `?mode=full` (default) or `?mode=screening`. **11 non-invasive fields are required** (age, bmi, cycle, cycleLength, weightGain, hairGrowth, skinDarkening, hairLoss, pimples, fastFood, regularExercise) — missing any returns 400 listing them. Other missing features default to training means, and the response reports `imputedCount` / `imputedFields` (how many of the 41 fields were imputed, and which). Returns `{"prediction": 0\|1, "label", "probability", "confidence", "model", "modelAccuracy", "modelAccuracyCiLow", "modelAccuracyCiHigh", "modelAccuracyNote", "mode", "threshold", "imputedCount", "imputedOf", "imputedFields", "warning?"}`. `confidence` is the probability of the **predicted** class — for a screening prediction below 0.5 it is the (low) PCOS probability itself, not `max(p, 1-p)`. `modelAccuracy` is the model's test accuracy on the 108-row held-out set (with its 95% CI and a note saying so) — it describes the model, not this one prediction. When 21+ fields were imputed the response also carries a `warning` that the result rests mostly on training averages. Non-numeric values return 400 with a clear message, never 500. Empty `{}` returns 400 (it must not return a confident prediction). |
| `GET /api/predictions?limit=20` | Prediction history, latest first (limit 1–100, default 20). Requires the `X-Admin-Key` header (401 without/wrong key; 503 if `ADMIN_KEY` is unset). 503 when the database is unavailable. |
| `GET /api/predictions/{id}` | One history record. Requires the `X-Admin-Key` header (same 401/503 rules). 404 for an unknown id, 503 without a database. |
| `PATCH /api/predictions/{id}/outcome` | Set the doctor-verified outcome: body `{"verifiedOutcome": 0\|1}`. Requires the `X-Admin-Key` header (same 401/503 rules). Re-PATCHing overwrites the previous value (latest value wins; `verifiedAt` updated). 400 for any other value, 404 for an unknown id, 503 without a database. |
| `POST /api/retrain` | Starts a retrain in the background (all models, on `../data/pcos_dataset.csv` plus verified history rows) and returns **202 Accepted** immediately with `{jobId, status: "RUNNING", statusUrl}`. Requires the `X-Admin-Key` header (401 without/wrong key; 503 if `ADMIN_KEY` is unset). A second retrain while one runs returns **409**. Takes ~2.5–12 minutes; poll the status endpoints below. |
| `GET /api/retrain/status` | The latest retrain job: `{jobId, status: "RUNNING"\|"DONE"\|"FAILED", startedAt, finishedAt, message, bestModel, bestModelTestAccuracy, verifiedRowsUsed, generatedAt}`. Requires the `X-Admin-Key` header (401/503 rules as above). |
| `GET /api/retrain/{jobId}` | One retrain job by id (only the latest job is retained). Requires the `X-Admin-Key` header. 404 for an unknown id. |

### Tests

`mvn test` runs 52 tests, 0 failures (2 aborted = live-MySQL tests skipped
when MySQL is unreachable from Java — see "Notes for this development VM
only"):

| Test class | Tests | What it covers |
|---|---|---|
| `PcodControllerTest` | 21 | MockMvc: health, models (6 entries), metrics, features, importance endpoints, predict (samples, modes, 400s, imputed-fields + warning, accuracy note + CI), history 503 without DB, predictions/PATCH auth (401 without/wrong key) |
| `PreprocessingTest` | 5 | Mean imputation, standardisation, 99th-percentile clipping, constant-feature handling |
| `PredictorTest` | 8 | Real `ModelStore`: full + screening predictions on samples 1/2, screening ignores lab/ultrasound features, thresholds valid, screening confidence = P(predicted class) incl. the p≈0.26 bug case, reload keeps predictions consistent, concurrent predict during reload never sees a mixed state |
| `PredictionRecordRepositoryTest` | 4 | JPA layer on H2 (**test scope only** — runtime DB stays MySQL): save/find, latest-first ordering, verified-only query, `imputedFieldsJson` round-trip |
| `RetrainSecurityTest` | 4 | `POST /api/retrain` without key → 401, wrong key → 401; with key → 202 with job id + status URL; status endpoint reports a failed job (bad dataset path) and 404s for unknown ids, all behind the admin key |
| `AdminKeyFailClosedTest` | 4 | With no admin key configured: retrain, predictions, PATCH outcome and retrain status all → 503 (fail-closed), even with a key supplied |
| `TrainingServiceTest` | 4 | Quoted-field CSV parsing (`splitCsvLine`, `loadCsv` — quoted commas, `""` escapes, malformed-cell counter reset per load), per-fold feature rankings (permutation + signal feature first in every fold), `cvAccuracyTopK` uses only the top-k features |
| `PredictionHistoryTest` | 2 | **Live MySQL**: POST a prediction → history lists it → get by id → PATCH verified outcome → 404s. Skipped (not failed) when MySQL is unreachable |

The live MySQL test is designed to skip cleanly: its context loads without
a live database (pinned MySQL dialect, Hikari no-fail-fast, no DDL at
startup), the table is created from `schema.sql` after a reachability
probe, and the probe result is cached with a 5-second login timeout so a
black-holed connection does not stall the suite.

### Verified end-to-end (final run, 2026-11-25, live jar, run-8 models)

The jar was run prediction-only (JPA excluded — see "Notes for this
development VM only" for why MySQL is not reachable from Java here) with
`ADMIN_KEY=test-admin-key`. Sample patients are from
`../data/sample_patients.json` (real rows, clean keys; true labels are
recorded there for checking only — the predictions below are endpoint smoke
tests, not accuracy evidence). Sample 1 = Sl.No 1 (true 0), sample 2 =
Sl.No 3 (true 1), sample 3 = Sl.No 151 (true 1); values verified against
the CSV (e.g. sample 2: age 33, marriageYears 10.0, betaHcg1 494.08).

```bash
curl http://localhost:8080/api/health
# {"status":"UP","service":"PCOD Prediction API"}

curl http://localhost:8080/api/models
# 6 entries. Soft Voting Ensemble: accuracy 0.8889, precision 0.8966,
# recall 0.7429, f1 0.8125, rocAuc 0.9252, prAuc 0.8791, brierScore 0.0992,
# cvAccuracy 0.9307, threshold 0.5, confusionMatrix {tp:26, fp:3, fn:9, tn:70}
# Gradient Boosting (Screening): accuracy 0.8241, precision 0.6905,
# recall 0.8286, f1 0.7532, threshold 0.25, nFeatures 23, mode "screening"

curl http://localhost:8080/api/metrics
# bestModel "Soft Voting Ensemble", 6 models, 5 baselines,
# deployedThreshold {model: "Soft Voting Ensemble", threshold: 0.5},
# screeningThreshold {model: "Gradient Boosting (Screening)", threshold: 0.25},
# verifiedRowsUsed 0

curl http://localhost:8080/api/features
# 41 entries; first {"key":"age","label":"Age (years)","min":20.0,"max":48.0,
# "default":31.214780600461893,"type":"number","screening":true,"required":true}
# last {"key":"endometrium","label":"Endometrium (mm)","min":0.0,"max":18.0,
# "default":8.477413394919166,"type":"number","screening":false,"required":false}
# 11 required fields, 23 screening fields.

curl http://localhost:8080/api/models/random-forest/importance
# {"model":"Random Forest","importance":[{"feature":"follicleR","importance":286.93}, ... 41 entries]}
curl http://localhost:8080/api/models/gradient-boosting-screening/importance
# {"model":"Gradient Boosting (Screening)","importance":[... 23 entries, top skinDarkening 776.24]}
curl http://localhost:8080/api/models/bagging/importance
# 404 (no impurity importance for Bagging)

# Sample 2 (true label 1), full mode:
curl -X POST http://localhost:8080/api/predict -H "Content-Type: application/json" -d \
'{"age":33,"weightKg":68.8,"heightCm":165.0,"bmi":25.27089073,"bloodGroup":11,"pulseRate":72,"rr":18,"hb":11.8,"cycle":2,"cycleLength":5,"marriageYears":10.0,"pregnant":1,"abortions":0,"betaHcg1":494.08,"betaHcg2":494.08,"fsh":5.54,"lh":0.88,"fshLh":6.295454545,"hip":40,"waist":36,"waistHipRatio":0.9,"tsh":2.54,"amh":6.63,"prl":10.52,"vitD3":49.7,"prg":0.36,"rbs":84.0,"weightGain":0,"hairGrowth":0,"skinDarkening":0,"hairLoss":1,"pimples":1,"fastFood":1.0,"regularExercise":0,"bpSystolic":120,"bpDiastolic":80,"follicleL":13,"follicleR":15,"avgFollicleSizeL":18.0,"avgFollicleSizeR":20.0,"endometrium":10.0}'
# {"prediction":1,"label":"PCOS Detected","probability":0.8566,"confidence":0.8566,"model":"Soft Voting Ensemble","modelAccuracy":0.8889,"modelAccuracyCiLow":0.8241,"modelAccuracyCiHigh":0.9444,"modelAccuracyNote":"Test accuracy of Soft Voting Ensemble on the 108-row held-out test set (95% CI 82.4%–94.4%); it describes the model, not this individual prediction.","mode":"full","threshold":0.5,"imputedCount":0,"imputedOf":41,"imputedFields":[]}

# Sample 1 (true label 0), full mode:
# {"prediction":0,"label":"No PCOS Detected","probability":0.099,"confidence":0.901,"model":"Soft Voting Ensemble","modelAccuracy":0.8889,"modelAccuracyCiLow":0.8241,"modelAccuracyCiHigh":0.9444,"modelAccuracyNote":"Test accuracy of Soft Voting Ensemble on the 108-row held-out test set (95% CI 82.4%–94.4%); it describes the model, not this individual prediction.","mode":"full","threshold":0.5,"imputedCount":0,"imputedOf":41,"imputedFields":[]}

# Sample 3 (true label 1), full mode:
# {"prediction":1,"label":"PCOS Detected","probability":0.8314,"confidence":0.8314,"model":"Soft Voting Ensemble","modelAccuracy":0.8889,"modelAccuracyCiLow":0.8241,"modelAccuracyCiHigh":0.9444,"modelAccuracyNote":"Test accuracy of Soft Voting Ensemble on the 108-row held-out test set (95% CI 82.4%–94.4%); it describes the model, not this individual prediction.","mode":"full","threshold":0.5,"imputedCount":0,"imputedOf":41,"imputedFields":[]}

# Sample 2 in screening mode — the screening model MISSES this patient
# (probability 0.1214 < threshold 0.25): its non-invasive profile is mild
# (no weight gain, no hair growth, no skin darkening) and the case is only
# visible through follicle counts / AMH, which screening cannot see. This
# is the expected behaviour difference between the modes, and why the
# screening model's measured recall is 82.86%, not 100%:
curl -X POST "http://localhost:8080/api/predict?mode=screening" -H "Content-Type: application/json" -d '<sample 2>'
# {"prediction":0,"label":"No PCOS Detected","probability":0.1214,"confidence":0.8786,"model":"Gradient Boosting (Screening)","modelAccuracy":0.8241,"modelAccuracyCiLow":0.75,"modelAccuracyCiHigh":0.8889,"modelAccuracyNote":"Test accuracy of Gradient Boosting (Screening) on the 108-row held-out test set (95% CI 75.0%–88.9%); it describes the model, not this individual prediction.","mode":"screening","threshold":0.25,"imputedCount":0,"imputedOf":41,"imputedFields":[]}
# Sample 1 screening: {"prediction":0,...,"probability":0.0764,...}
# Sample 3 screening: {"prediction":1,...,"probability":0.8513,...}

# Confidence is the probability of the PREDICTED class, even below 0.5:
# dataset row 9 (true label 0) in screening mode predicts PCOS with
# probability 0.2577 — above the 0.25 threshold, below 0.5 — and the
# response reports confidence 0.2577 (the PCOS side), not 0.7423. An
# earlier version reported max(p, 1-p) = 0.7423 here, which was the
# No-PCOS side's probability. Verified live against the running jar.

# Only the 11 required fields (30 of 41 imputed):
curl -X POST http://localhost:8080/api/predict -H "Content-Type: application/json" -d \
'{"age":33,"bmi":25.0,"cycle":2,"cycleLength":5,"weightGain":0,"hairGrowth":0,"skinDarkening":0,"hairLoss":1,"pimples":1,"fastFood":1,"regularExercise":0}'
# {"prediction":0,"label":"No PCOS Detected","probability":0.2358,"confidence":0.7642,"model":"Soft Voting Ensemble","modelAccuracy":0.8889,"modelAccuracyCiLow":0.8241,"modelAccuracyCiHigh":0.9444,"modelAccuracyNote":"Test accuracy of Soft Voting Ensemble on the 108-row held-out test set (95% CI 82.4%–94.4%); it describes the model, not this individual prediction.","mode":"full","threshold":0.5,"imputedCount":30,"imputedOf":41,"imputedFields":["weightKg","heightCm","bloodGroup","pulseRate","rr","hb","marriageYears","pregnant","abortions","betaHcg1","betaHcg2","fsh","lh","fshLh","hip","waist","waistHipRatio","tsh","amh","prl","vitD3","prg","rbs","bpSystolic","bpDiastolic","follicleL","follicleR","avgFollicleSizeL","avgFollicleSizeR","endometrium"],"warning":"More than half of the 41 input fields were missing and were filled with training averages — this result rests mostly on average values, not on the patient's own data. Treat it with extra caution; it is not a diagnosis."}

# Error cases (all HTTP 400, never 500):
curl -X POST http://localhost:8080/api/predict -H "Content-Type: application/json" -d '{}'
# {"error":"Missing required fields: age, bmi, cycle, cycleLength, weightGain, hairGrowth, skinDarkening, hairLoss, pimples, fastFood, regularExercise. These 11 non-invasive fields are required for a meaningful prediction."}  (400)
curl -X POST http://localhost:8080/api/predict -H "Content-Type: application/json" -d '{"age":33}'
# {"error":"Missing required fields: bmi, cycle, cycleLength, weightGain, hairGrowth, skinDarkening, hairLoss, pimples, fastFood, regularExercise. These 11 non-invasive fields are required for a meaningful prediction."}  (400)
curl -X POST "http://localhost:8080/api/predict?mode=bogus" -H "Content-Type: application/json" -d '<11 required fields>'
# {"error":"Query parameter 'mode' must be 'full' or 'screening', got: bogus"}  (400)
curl -X POST http://localhost:8080/api/predict -H "Content-Type: application/json" -d '{"age":"twenty-eight", ...}'
# {"error":"Feature 'age' must be a number, got: \"twenty-eight\""}  (400)

# History without a database (prediction-only run):
curl http://localhost:8080/api/predictions
# {"error":"Missing or invalid X-Admin-Key header. This endpoint requires the admin key (ADMIN_KEY)."}  (401 — the admin key is checked before the database)
curl http://localhost:8080/api/predictions -H "X-Admin-Key: test-admin-key"
# {"error":"Prediction history database is not available — check that MySQL is running and the datasource is configured (see README)."}  (503)

# Retrain auth (ADMIN_KEY=test-admin-key in this run):
curl -X POST http://localhost:8080/api/retrain
# {"error":"Missing or invalid X-Admin-Key header. POST /api/retrain requires the admin key (ADMIN_KEY)."}  (401)
curl -X POST http://localhost:8080/api/retrain -H "X-Admin-Key: wrong-key"
# (same 401)
# With the right key the endpoint answers 202 immediately and trains in
# the background (verified live in this session):
curl -X POST http://localhost:8080/api/retrain -H "X-Admin-Key: test-admin-key"
# {"jobId":"a5070e68-af4c-4aa1-ad55-f8b8bfae077b","status":"RUNNING","message":"Retraining started in the background — poll GET /api/retrain/status for the result.","statusUrl":"/api/retrain/status"}  (202)
curl -X POST http://localhost:8080/api/retrain -H "X-Admin-Key: test-admin-key"
# {"error":"A retrain is already running (job a5070e68-...). Wait for it to finish — poll GET /api/retrain/status — before starting another."}  (409)
curl http://localhost:8080/api/retrain/status -H "X-Admin-Key: test-admin-key"
# {"jobId":"a5070e68-...","status":"RUNNING","startedAt":"2026-11-25T19:31:15.47...","finishedAt":null,"message":"Retraining started in the background — poll GET /api/retrain/status for the result.","bestModel":null,"bestModelTestAccuracy":null,"verifiedRowsUsed":null,"generatedAt":null}  (200)
# (The background job above ran to completion in this session: DONE,
# best model Soft Voting Ensemble, test accuracy 0.8889, verified rows
# used 0, generatedAt 2026-11-25T19:34:51 — the retrained metrics are
# identical to the frozen run-8 metrics, and the app reloaded the new
# models from PCOD_RESOURCES_DIR (/tmp/smoke-retrain in this run) and
# kept serving predictions (sample 2 still predicts 1 at p=0.8566) with
# no restart. The frozen run-8 artefacts in src/main/resources were
# untouched throughout.)

# Swagger:
curl http://localhost:8080/v3/api-docs        # 200
curl -L http://localhost:8080/swagger-ui.html  # 200
```

MySQL was verified separately against the real server (MySQL 8.0.46) with
the native `mysql` client: `schema.sql` runs cleanly (and idempotently —
running it twice is fine), the `prediction_records` table is created with
all 10 columns including nullable `verified_outcome` / `verified_at`, an
insert + verified-outcome update + `SELECT ... WHERE verified_outcome IS
NOT NULL` (the exact query retraining uses) all work. What was **not**
verified live in this VM: the app writing a prediction to MySQL and the
`PredictionHistoryTest` running against it — Java processes in this VM
cannot open TCP connections at all (see "Notes for this development VM
only"), so no Java code here can reach MySQL over TCP. The JPA wiring is
covered by `PredictionRecordRepositoryTest` (H2, in-JVM, no TCP) and the
live history test runs on any machine where Java can reach MySQL (the
user's laptop, or Docker Compose below).

## Docker Compose (project root)

`../docker-compose.yml` starts MySQL + backend + frontend with one command
(`docker compose up --build` from `pcod-working-project/`):

- `mysql`: `mysql:8.0`, database `pcod_db`, data in the `mysql-data`
  volume, healthcheck `mysqladmin ping`. Root password from
  `MYSQL_ROOT_PASSWORD` — **required** (`${MYSQL_ROOT_PASSWORD:?set
  MYSQL_ROOT_PASSWORD in .env}`); there is no default password. MySQL's
  port **3306 is not published to the host** — the backend reaches MySQL
  over the Compose network; host access needs
  `docker exec -it pcod-mysql mysql -uroot -p`.
- `backend`: builds `./backend` (multi-stage Maven build; the trained
  models are baked into the jar), port 8080, `MYSQL_URL` pointing at host
  `mysql`, `ADMIN_KEY` from the environment (**required** —
  `${ADMIN_KEY:?set ADMIN_KEY in .env}`; the stack refuses to start
  without it, and the app itself fails closed without it), `./data`
  mounted read-only at `/app/data` for `POST /api/retrain`, and the named
  volume `pcod-resources` mounted at `/app/resources` (`PCOD_RESOURCES_DIR`)
  so retrained artefacts survive container recreation.
- `frontend`: multi-stage build (Node builds the Vite bundle, then
  **nginx:alpine serves `dist/`** on port 5173 and proxies `/api/` to
  `http://backend:8080` inside the Compose network — not the Vite dev or
  preview server). Locally (outside Docker) the Vite proxy still defaults
  to `http://localhost:8080` — `frontend/vite.config.js` reads
  `VITE_BACKEND_URL` with that default, so local dev is unchanged.

Set `MYSQL_ROOT_PASSWORD` and `ADMIN_KEY` in a `.env` file next to
`docker-compose.yml` (or in the shell) before starting — both are required
and the stack will not start without them. Stop with
`docker compose down` (add `-v` to delete the MySQL data volume; the
`pcod-resources` volume is kept unless removed explicitly, so retrained
models survive a `down`). Then
open `http://localhost:5173`; Swagger UI is at
`http://localhost:8080/swagger-ui.html`.

**Not live-tested in this VM:** Docker is installed here (29.1.3,
Compose 2.40.3) but no container can start (mount-namespace creation is
blocked), so the Compose stack could not be started and the frontend image
could not be built here. The YAML was validated
(services mysql/backend/frontend parse correctly) and both Dockerfiles were written from the project's actual
build steps (Maven package for the backend, `npm ci && npm run build`
then nginx for the frontend), but the one-command start must be
verified on a machine with Docker before it is relied on.

## Project layout

```
pcod-working-project/
  docker-compose.yml                     MySQL + backend + frontend (one-command start)
  backend/
    Dockerfile                           Multi-stage Maven build + JRE run
    pom.xml                              Spring Boot 3.2.5, Java 17, smile-core 3.1.1,
                                         spring-boot-starter-data-jpa, mysql-connector-j,
                                         springdoc-openapi 2.3.0, H2 (test scope only)
    src/main/java/com/pcod/
      PcodApplication.java               Spring Boot entry point + CORS (5173, 3000)
      controller/PcodController.java     /api/health, /models, /metrics, /features,
                                         /models/{name}/importance, /predict, /predictions,
                                         /predictions/{id}, /predictions/{id}/outcome, /retrain
      db/PredictionRecord.java           JPA entity: prediction_records (+ verified fields)
      db/PredictionRecordRepository.java JpaRepository + findByVerifiedOutcomeNotNull
      ml/FeatureSpec.java                The 41 features: keys, labels, CSV headers;
                                         SCREENING_FEATURES (23), REQUIRED_FEATURES (11)
      ml/Preprocessing.java              Imputation / clipping / standardisation (fitted, stored)
      ml/Metrics.java                    Metrics JSON model (+ thresholds, baselines, CIs)
      ml/TrainingService.java            Load, split, CV tuning, baselines, screening,
                                         threshold tuning, train, evaluate, persist
      ml/BaggingClassifier.java          Bagging (bootstrap DecisionTrees, majority vote)
      ml/ModelStore.java                 Loads saved models at startup; prediction (full/screening)
      trainer/ModelTrainer.java          Standalone trainer main class
    src/main/resources/
      application.properties             server.port=8080, MySQL (env vars), ADMIN_KEY, dataset path
      schema.sql                         prediction_records table (owns the schema; ddl-auto=none)
      metrics.json                       Measured metrics (generated by the trainer)
      preprocessing.json                 Fitted preprocessing (generated)
      models/*.ser                       The five trained models (generated)
      models/MODEL_CARD.md               Model card (generated from metrics.json)
    src/test/java/com/pcod/
      PcodControllerTest.java            21 MockMvc tests
      PreprocessingTest.java             5 unit tests
      PredictorTest.java                 8 tests on the real ModelStore
      PredictionRecordRepositoryTest.java  4 JPA tests on H2 (test scope only)
      PredictionHistoryTest.java         2 live-MySQL tests (skip when MySQL unreachable)
      RetrainSecurityTest.java           4 tests (401 without/wrong key; 202 + status)
      AdminKeyFailClosedTest.java        4 tests (503 fail-closed, no key configured)
      TrainingServiceTest.java           4 tests (CSV parsing, per-fold ranking)
    src/test/resources/application.properties   JPA excluded (unit tests need no DB)
  frontend/
    Dockerfile                           node:20 build + vite preview on 5173
    vite.config.js                       /api proxy target from VITE_BACKEND_URL (default localhost:8080)
```

## Notes for this development VM only

On a normal machine (and in IntelliJ IDEA on the user's computer) plain
`mvn clean package` downloads everything from Maven Central and just
works, and the app connects to MySQL normally. This VM is different in two
ways:

1. **Maven cannot reach the internet from Java here** (raw TCP from Java
   processes is blocked), so dependencies were populated with `curl` by
   dev-only helper scripts used during development in this VM — those
   scripts are not shipped with the project. The build runs here with `mvn -o` (offline)
   against the already-populated local Maven repository. None of this
   affects the project: the `pom.xml` is a standard Spring Boot 3.2.5
   project and IntelliJ will resolve it normally.
2. **Java cannot open TCP connections here at all — including to
   localhost MySQL.** Proven directly: a Java socket to 127.0.0.1:3306
   receives the sandbox's block message ("Database connections is turned
   off...") instead of the MySQL handshake, and the MySQL connector then
   fails with a misleading `PacketTooBigException`. The native `mysql`
   client connects fine (socket login as root, no password), and MySQL
   8.0.46 is installed and running in this VM. Consequences: the live
   history test skips here (by design), the app was live-tested
   prediction-only (JPA excluded), and the MySQL schema was verified with
   the native client instead. On the user's laptop none of this applies —
   Java reaches MySQL normally.

Two Smile API facts found while building (verified against the Smile 3.1.1
sources on Maven Central): there is no separate `smile-data` 3.1.1
artifact — the DataFrame classes live in `smile-base`, pulled in
transitively by `smile-core` — and Smile's `DoubleVector.toIntArray()` is
unsupported, so the target column is built as an `IntVector`
(`TrainingService.buildDataFrame`). Smile's linear SVM takes labels −1/+1
only and its soft-predict throws; see Method for how that is handled.

## Limitations and future work

- The honest accuracy ceiling on 541 patients is high-80s to low-90s; the
  single largest improvement would come from more verified labelled patient
  data, which no public source currently provides (see Dataset).
- **No refit on all 541 rows — by design.** The shipped models were
  trained on the 433-row train split only, so the reported test metrics
  (88.89% for the deployed ensemble, 95% CI 82.4–94.4%) describe the exact
  artefact being served. Refitting on all 541 rows after evaluation would
  use ~25% more data and probably help slightly, but then the test metrics
  would no longer describe the shipped model — there would be no honest
  held-out number left for it. A refit is a deliberate future decision,
  not a silent step.
- **Follicle counts are near-diagnostic.** Antral follicle counts differ
  sharply between the classes (follicleL mean 9.79 with PCOS vs 4.35
  without; follicleR 10.76 vs 4.64) — they are Rotterdam-adjacent
  morphology features (ovarian morphology is one of the Rotterdam
  criteria), so a model that sees them is partly re-deriving the
  diagnosis. That is why the 23-feature screening model exists: it uses
  no follicle counts, no AMH and no other labs, and still reaches 82.41%
  test accuracy with 82.86% recall — the honest answer to "does this work
  before any lab or ultrasound?".
- Recall is the weaker side for every model (deployed ensemble 74.29% on
  the test set; screening 82.86% at its tuned threshold). The deployed
  model must not be used to rule PCOS out.
- Probabilities are not perfectly calibrated (Brier 0.10 for the deployed
  model, 0.18 for AdaBoost); they are risk scores, not clinical
  probabilities. Calibration on a larger sample is future work.
- The models use only the 41 tabular clinical features. Ultrasound images
  were out of scope for this Java tabular ensemble.
- Single-centre region (10 hospitals in Kerala) and a single dataset — no
  external validation cohort yet. Performance on other populations is
  unknown.
- The API is a screening aid, not a diagnosis: predictions are
  probabilities from a model trained on 541 patients, and a clinician's
  judgement is required for any real decision. The screening model is for
  early, pre-visit risk indication without labs or ultrasound; it can miss
  cases that only labs/ultrasound reveal (sample 2 above is a live example).
- Possible next steps: collect doctor-verified outcomes through the PATCH
  endpoint and retrain on them (the mechanism is built); probability
  calibration; a held-out external validation cohort if one becomes
  available.

## Sources

- Dataset: Kaggle, Prasoon Kottarathil, "Polycystic Ovary Syndrome (PCOS)",
  https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos
  (file `PCOS_data_without_infertility.xlsx`, sheet `Full_new`). Local
  verification and provenance: `../data/SCHEMA.md`. Larger-dataset search:
  `../data/LARGE_DATASET.md`.
- Smile (Statistical Machine Intelligence and Learning Engine) 3.1.1:
  https://github.com/haifengl/smile (Maven Central
  `com.github.haifengl:smile-core:3.1.1`); API details verified from the
  published `smile-core` / `smile-base` 3.1.1 sources jars.
- Spring Boot 3.2.5: https://spring.io/projects/spring-boot
- springdoc-openapi 2.3.0 (Swagger/OpenAPI): https://springdoc.org
- MySQL 8.0: https://dev.mysql.com/doc/refman/8.0/en/
- H2 database (test scope only — the runtime database is MySQL):
  https://www.h2database.com
- India's Digital Personal Data Protection Act, 2023:
  https://www.meity.gov.in (privacy note in "Database (MySQL)").
