# Model Card — PCOD/PCOS Prediction

| | |
|---|---|
| Model version | 1.2.0 (trainer run 2026-11-26, Java 25 / Smile 6.3.0) |
| Training date | 2026-11-26T15:44:00.651177400 |
| Dataset | Kaggle, Prasoon Kottarathil, "Polycystic Ovary Syndrome (PCOS)" — 541 patients from 10 hospitals in Kerala, 45 columns. 41 input features after dropping identifiers and the junk column. Target `PCOS (Y/N)`: 0 = 364 (67.3%), 1 = 177 (32.7%). |
| Split | Stratified 80/20 train/test, seed 42. Train 433 rows, test 108 rows. Test split evaluated once per model, after all tuning. |
| Tuning | 5-fold stratified cross-validation on the train split only (seed 42): hyperparameter grids, feature subsets from Random Forest importance rankings re-computed inside each CV fold (ranked on that fold's training part only), clipping on/off, minority oversampling on/off. Missing values are filled with the training median (not the mean — a few extreme outliers inflate the raw mean far above the typical value). Preprocessing re-fitted inside every fold. The deployed threshold maximises F1 on out-of-fold probabilities; the screening threshold is the highest threshold reaching ≥90% out-of-fold recall. Repeated CV (10×5), McNemar tests, ablation and calibration reuse the fixed tuned hyperparameters — extra analyses of the same run, not new tuning. |
| Deployed model | Soft Voting Ensemble — chosen by best train CV, not by test accuracy. |
| Verified history rows used | 0 (only doctor-verified outcomes are ever used for training) |
| Library / stack | Java 25 (LTS), Spring Boot 4.1.2, Smile 6.3.0 (Java ML). The metrics table below is the 1.2.0 run (2026-11-26): median imputation + recall-target screening threshold, same split, seed and protocol as run 8 and the 1.1.0 retrain. Previous runs: run 8 (2026-11-25, Java 17 / Spring Boot 3.2.5 / Smile 3.1.1) — deployed ensemble 88.89% (P 89.66%, R 74.29%, ROC-AUC 0.9252, threshold 0.50, CV 93.07%); screening 82.41% (R 82.86% at threshold 0.25). 1.1.0 (2026-11-26 13:09, Java 25 / Smile 6.3.0) — deployed ensemble 88.89% (P 92.59%, R 71.43%, ROC-AUC 0.9272, threshold 0.55, CV 91.92%); screening 82.41% (R 74.29% at threshold 0.30). The headline deployed accuracy is unchanged at 88.89% across all three runs. |

## Measured test results (held-out test set, n = 108)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Train CV | TP/FP/FN/TN | 95% CI (accuracy) |
|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 88.89% | 96.00% | 68.57% | 80.00% | 0.930 | 0.886 | 0.1022 | 91.22% | 24/1/11/72 | 81.5–94.4% |
| Bagging | 87.04% | 86.21% | 71.43% | 78.13% | 0.900 | 0.847 | 0.1077 | 89.15% | 25/4/10/69 | 80.6–92.6% |
| AdaBoost | 87.04% | 92.00% | 65.71% | 76.67% | 0.919 | 0.869 | 0.1810 | 89.38% | 23/2/12/71 | 79.6–92.6% |
| Gradient Boosting | 87.96% | 82.35% | 80.00% | 81.16% | 0.943 | 0.905 | 0.0911 | 91.22% | 28/6/7/67 | 81.5–93.5% |
| Soft Voting Ensemble | 88.89% | 89.66% | 74.29% | 81.25% | 0.927 | 0.884 | 0.1006 | 91.92% | 26/3/9/70 | 82.4–94.4% |
| Gradient Boosting (Screening) | 64.81% | 47.83% | 94.29% | 63.46% | 0.879 | 0.758 | 0.1384 | 82.68% | 33/36/2/37 | 55.6–74.1% |

TP/FP/FN/TN are test-set confusion counts at each model's decision threshold
(the deployed and screening models use their tuned thresholds; the rest use 0.5).
The 95% CI is a bootstrap interval (1000 resamples, seed 42) for test accuracy.
With only 108 test rows, one patient is worth 0.93 percentage
points — a difference of 2–3 patients (e.g. 86% vs 89%) is not significant.

**Run history — what changed vs 1.1.0 (Smile 6.3.0 retrain, 2026-11-26 13:09).** Same
dataset, split, seed and tuning protocol; two deliberate changes: (1) missing
values are now filled with the training median instead of the raw mean — the
raw mean was inflated by extreme outliers (FSH max 5052, Vitamin D3 max
6014.66), so a blank FSH filled with the mean sat +3.9 SD above the clipped
mean; (2) the screening threshold is now the highest threshold reaching ≥90%
recall on out-of-fold probabilities, instead of the F1-maximising threshold.
The deployed ensemble is unchanged at 88.89% accuracy (threshold re-tuned
0.55 → 0.50, so precision fell 92.59% → 89.66% and recall rose 71.43% →
74.29%; ROC-AUC 0.9272; train CV 91.92% unchanged). Random Forest 87.96% →
88.89% (new best hyperparameters: nodeSize 1, ntrees 300, subsample 1.0);
AdaBoost 86.11% → 87.04% (maxDepth 3, 300 trees); Gradient Boosting and
Bagging unchanged at 87.96% / 87.04%. The linear SVM rose 87.96% → 88.89%
(recall 82.86%, the best single-model recall). The screening model is the
big move: threshold 0.30 → 0.08, recall 74.29% → 94.29% (it now misses only
2 of 35 test cases), but precision fell 72.22% → 47.83% and accuracy 82.41% →
64.81% — 36 of 73 healthy patients are flagged. That is the honest cost of
a 90%-recall screening rule on this data, and why the screening result is
shown as "High risk — refer for testing", not a diagnosis.

## Decision thresholds (tuned on train CV out-of-fold probabilities only, never on the test set)

| Model | Tuned threshold | OOF precision / recall / F1 | Test at tuned threshold (acc / P / R / F1) | Test at 0.5 (acc / P / R / F1) |
|---|---|---|---|---|
| Soft Voting Ensemble | 0.50 | 90.77% / 83.10% / 86.76% | 88.89% / 89.66% / 74.29% / 81.25% | 88.89% / 89.66% / 74.29% / 81.25% |
| Gradient Boosting (Screening) | 0.08 | 46.21% / 90.14% / 61.10% | 64.81% / 47.83% / 94.29% / 63.46% | 80.56% / 73.33% / 62.86% / 67.69% |

The deployed threshold maximises F1 on out-of-fold train probabilities
(ties broken toward higher recall). The screening threshold is the highest
threshold that still reaches ≥90% out-of-fold recall (ties toward higher
precision) — a screening rule that misses as few true cases as possible.
At 0.08 the screening model flags 69 of 108 test patients (33 true, 36
false alarms); at 0.5 it would flag 30 (22 true, 8 false alarms) and miss
13 cases. Recall matters most for a screening tool: a missed case costs
more than a false alarm — but the false-alarm cost is real and is reported
here, not hidden.

## Screening model

The screening model is Gradient Boosting trained on 23 non-invasive features
only — age, body measurements, cycle pattern, symptoms and lifestyle. No
hormone/blood labs and no ultrasound measures. It is the model behind
`POST /api/predict?mode=screening`, for early detection before a doctor visit.

**Follicle counts are near-diagnostic.** Antral follicle counts differ
sharply between the classes (follicleL mean 9.79 with PCOS vs 4.35 without;
follicleR 10.76 vs 4.64) — they are Rotterdam-adjacent morphology features
(ovarian morphology is one of the Rotterdam criteria), so a model that sees
them is partly re-deriving the diagnosis. The screening model exists
precisely to answer that: with no follicle counts, no AMH and no other
labs, at its 0.5 operating point it still reaches 80.56% test accuracy
with 62.86% recall; at the recall-target threshold (0.08) it catches
94.29% of cases but flags about half of healthy patients too.

## Repeated cross-validation (10×5, all 541 rows, fixed tuned hyperparameters)

| Model | Accuracy (mean ± SD) | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Soft Voting Ensemble | 90.18% ± 0.68 | 88.47% ± 1.37 | 80.51% ± 1.51 | 84.29% ± 1.12 | 0.949 ± 0.002 |
| Gradient Boosting | 89.67% ± 0.68 | 86.29% ± 1.29 | 81.36% ± 1.62 | 83.74% ± 1.11 | 0.950 ± 0.003 |
| Random Forest | 88.96% ± 0.66 | 89.13% ± 1.16 | 75.48% ± 1.65 | 81.73% ± 1.17 | 0.949 ± 0.003 |
| Logistic Regression | 88.28% ± 0.61 | 83.54% ± 1.07 | 79.94% ± 1.44 | 81.69% ± 1.00 | 0.933 ± 0.004 |
| AdaBoost | 88.02% ± 0.87 | 84.38% ± 1.40 | 77.80% ± 1.81 | 80.95% ± 1.44 | 0.937 ± 0.006 |
| Bagging | 87.95% ± 0.99 | 83.36% ± 1.80 | 78.93% ± 1.31 | 81.08% ± 1.50 | 0.924 ± 0.004 |
| SVM (Linear) | 87.73% ± 1.05 | 82.50% ± 2.63 | 79.44% ± 2.03 | 80.90% ± 1.50 | 0.856 ± 0.011 |
| KNN | 85.90% ± 0.77 | 85.09% ± 1.33 | 68.98% ± 1.95 | 76.18% ± 1.45 | 0.889 ± 0.005 |
| Gradient Boosting (Screening) | 83.25% ± 0.52 | 77.93% ± 1.30 | 68.14% ± 1.14 | 72.69% ± 0.81 | 0.868 ± 0.006 |
| Decision Tree | 83.12% ± 1.21 | 76.56% ± 2.54 | 69.89% ± 2.31 | 73.04% ± 1.88 | 0.872 ± 0.015 |

The held-out 80/20 split is kept as the primary check; repeated CV uses all
541 rows with the same fixed hyperparameters (no re-tuning per repeat), so it
measures stability, not a new best score. The ensemble's lead over Gradient
Boosting (+0.5 pts accuracy) is within one standard deviation — the ranking
is stable, the gap is small.

## McNemar's test (test set, n = 108)

| Comparison | Discordant (ensemble right / other right) | χ² (Edwards) | Exact p (two-sided) | Significant? |
|---|---|---|---|---|
| Soft Voting Ensemble vs Logistic Regression | 8 / 4 | 0.75 | 0.388 | No |
| Soft Voting Ensemble vs Gradient Boosting | 3 / 2 | 0.00 | 1.000 | No |

On 108 test rows the ensemble is not significantly better than Logistic
Regression or Gradient Boosting. The ensemble is deployed because it wins on
train CV and on repeated CV, not because the test-set gap is proven.

## Feature-group ablation (same split, same protocol)

| Variant | Features | Voting accuracy / recall | Gradient Boosting accuracy / recall |
|---|---|---|---|
| Full | 41 | 88.89% / 74.29% | 87.96% / 80.00% |
| Without ultrasound (no follicleL, follicleR, avgFollicleSizeL, avgFollicleSizeR, endometrium) | 36 | 77.78% / 51.43% | 79.63% / 57.14% |
| Without hormones (no fsh, lh, fshLh, amh, tsh, prl) | 35 | 89.81% / 77.14% | 89.81% / 77.14% |
| Non-invasive only (23 screening features) | 23 | 80.56% / 57.14% | 80.56% / 62.86% |

Ultrasound (follicle counts, follicle size, endometrium) carries most of the
signal: removing it costs ~11 points of accuracy and ~23 points of recall.
Hormone labs add nothing on top of the other features here — without them
the ensemble is slightly better (89.81%), so the hormones' information is
already carried by the remaining features. The non-invasive set alone still
reaches 80.56%.

## Calibration (Soft Voting Ensemble)

Binned predicted probabilities vs observed frequency (10 equal-width bins).
Full bin data is in `metrics.json` (`calibration.testBins` and
`calibration.oofBins`); the reliability-curve plot is Figure 7.4 in the
project report.

| Predicted bin | Test: mean predicted | Test: observed | Test n | OOF: mean predicted | OOF: observed | OOF n |
|---|---|---|---|---|---|---|
| 0.00–0.10 | 0.098 | 0.000 | 7 | 0.094 | 0.000 | 22 |
| 0.10–0.20 | 0.139 | 0.075 | 40 | 0.144 | 0.018 | 167 |
| 0.20–0.30 | 0.241 | 0.095 | 21 | 0.243 | 0.143 | 63 |
| 0.30–0.40 | 0.385 | 0.500 | 2 | 0.348 | 0.161 | 31 |
| 0.40–0.50 | 0.441 | 0.333 | 9 | 0.454 | 0.350 | 20 |
| 0.50–0.60 | 0.540 | 0.667 | 3 | 0.540 | 0.556 | 18 |
| 0.60–0.70 | 0.637 | 0.800 | 5 | 0.638 | 0.900 | 20 |
| 0.70–0.80 | 0.746 | 1.000 | 11 | 0.758 | 0.981 | 54 |
| 0.80–0.90 | 0.829 | 0.900 | 10 | 0.833 | 0.974 | 38 |
| 0.90–1.00 | — | — | 0 | — | — | 0 |

The ensemble is conservative in the middle: it under-estimates risk between
0.6 and 0.8 (predicted 0.64–0.75, observed 0.80–1.00) and slightly
over-estimates it below 0.3. A reported probability of 0.85 does not mean
"about 85% of such patients have PCOS" — at predicted 0.83 the observed
frequency is 0.90. The probabilities are risk scores, not calibrated
clinical probabilities; Brier score (0.101) measures overall error, not
calibration. The out-of-fold bins (n = 433) show the same shape with less
noise.

## Serving & retraining

- **Confidence is the probability of the predicted class.** The deployed
  ensemble uses threshold 0.50, the screening model 0.08 — so a screening
  prediction can be PCOS with a probability below 0.5 (for example, a
  screening probability of 0.09 is a PCOS prediction at the 0.08 threshold),
  and the response reports that same probability as its confidence, not
  `max(p, 1-p)`.
- **`modelAccuracy` in the predict response is the model's test accuracy**
  on the 108-row held-out set (with its 95% CI and a note saying so) — it
  describes the model, not the individual prediction. When 21 or more of
  the 41 fields were median-imputed, the response also carries a `warning`
  that the result rests mostly on training medians.
- **Retraining is asynchronous and admin-only.** `POST /api/retrain`
  returns 202 with a job id; `GET /api/retrain/status` (or
  `/api/retrain/{jobId}`) reports RUNNING / DONE / FAILED. A second
  retrain while one runs is rejected with 409. Retrained artefacts are
  written to the external resources directory and loaded from it first at
  startup (Docker: the `pcod-resources` volume), so a restart serves the
  new models; if that directory is empty or corrupt, the app falls back
  to the models baked into the jar.
- **Admin key is fail-closed.** `GET /api/predictions`,
  `GET /api/predictions/{id}`, `PATCH /api/predictions/{id}/outcome` and
  all retrain endpoints require `X-Admin-Key`; without the header or with
  the wrong key they return 401, and if `ADMIN_KEY` is unset they return
  503 (disabled) instead of running unprotected. The key is compared in
  constant time (`MessageDigest.isEqual`).
- **Only verified, complete rows retrain.** Retraining uses only
  doctor-verified outcomes, and a verified row is used only if none of its
  stored fields were median-imputed at prediction time (tracked per record
  as `imputedFields`); rows with any imputed field — or written before
  that field was tracked — are excluded, so the model never learns from
  training medians presented as real values.
- **Rate limit.** `POST /api/predict` is limited to 120 requests/minute
  per IP (fixed window, in-memory); over the limit returns 429.
- **Range validation.** A value outside a feature's plausible range is
  rejected with 400, not silently clipped. The ranges are defined once in
  `FeatureSpec` and also returned by `GET /api/features`.
- **Model version.** Every prediction record carries the model version
  that made it (`modelVersion`, from `metrics.json`), so a retrain is
  auditable from the history table.

## Baselines (same split, same train-only CV protocol)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Train CV | TP/FP/FN/TN | 95% CI (accuracy) |
|---|---|---|---|---|---|---|---|---|---|---|
| Majority Class | 67.59% | 0.00% | 0.00% | 0.00% | 0.500 | 0.324 | 0.3241 | 67.21% | 0/0/35/73 | 58.3–75.9% |
| Logistic Regression | 85.19% | 82.76% | 68.57% | 75.00% | 0.880 | 0.860 | 0.1155 | 87.07% | 24/5/11/68 | 77.8–91.7% |
| Decision Tree | 82.41% | 80.77% | 60.00% | 68.85% | 0.819 | 0.696 | 0.1428 | 84.99% | 21/5/14/68 | 75.0–89.8% |
| SVM (Linear) | 88.89% | 82.86% | 82.86% | 82.86% | 0.873 | 0.742 | 0.1111 | 89.61% | 29/6/6/67 | 82.4–94.4% |
| KNN | 84.26% | 82.14% | 65.71% | 73.02% | 0.814 | 0.716 | 0.1430 | 85.91% | 23/5/12/68 | 77.8–90.7% |

The majority-class baseline (always predict "no PCOS") scores 67.59% on
the test set by predicting the majority class every time — any useful model
must beat it.
The linear SVM outputs hard decisions (decision-function sign only, no
calibrated probabilities), so its ROC-AUC/Brier reflect 0/1 scores.

## Top features (train-only importance, mean decrease in impurity)

Random Forest (all 41): follicleR 154.1, follicleL 115.7, amh 101.5, tsh 94.1, prl 86.8, fshLh 86.5, fsh 86.1, vitD3 85.1, bmi 84.6, age 83.0

Gradient Boosting (all 41): follicleR 1199.2, follicleL 285.8, hairGrowth 241.8, skinDarkening 205.8, weightGain 128.6, cycle 105.5, fshLh 103.7, amh 84.6, lh 69.0, cycleLength 51.8

Screening model (23 non-invasive): skinDarkening 262.7, hairGrowth 148.7, cycleLength 135.7, weightGain 134.8, cycle 128.9, fastFood 61.9, waistHipRatio 54.1, age 52.4, marriageYears 46.5, pimples 43.6

## Intended use

Screening aid for PCOS risk, not a diagnosis. A prediction is a probability
from a model trained on 541 patients from 10 hospitals in Kerala; a
clinician's judgement is required for any real decision. The screening model
is for early, pre-visit risk indication without any lab or ultrasound; the
full model uses all 41 features including labs and ultrasound.

## Limitations

- Small data: 541 patients, 108 in the test set. Accuracy carries roughly
  ±3 percentage points of sampling variation; small differences between
  models are not significant — the McNemar tests above show the ensemble's
  test-set lead over Logistic Regression (p = 0.388) and Gradient Boosting
  (p = 1.000) is not statistically significant.
- **No refit on all 541 rows — by design.** The shipped models were
  trained on the 433-row train split only, so the reported test metrics
  describe the exact artefact being served. Refitting on all 541 rows
  after evaluation would use ~25% more data but then the test metrics
  would no longer describe the shipped model.
- Single-centre region (Kerala hospitals) and a single dataset — no external
  validation cohort yet. Performance on other populations is unknown.
- Recall is the weaker side: the deployed model misses some true cases
  (see the confusion matrix). It must not be used to rule PCOS out.
- Probabilities are not perfectly calibrated (see the calibration section);
  they are risk scores, not clinical probabilities.
- Missing values are filled with training medians; a prediction based mostly
  on imputed values is less trustworthy — the API reports `imputedCount`
  and `imputedFields` (how many of the 41 fields were imputed, and which)
  and requires 11 key fields, and adds a `warning` when 21 or more fields
  were imputed. Verified rows with any imputed field are excluded from
  retraining.
- Only doctor-verified outcomes (PATCH /api/predictions/&#123;id&#125;/outcome)
  are ever used for retraining. Unverified model predictions are never
  trained on.

## Privacy

Prediction records are stored in the project's own MySQL database
(`prediction_records` table) and nowhere else. The form collects an optional
patient name and ID — they are stored only with the history record, never
sent to the model and never used in training; leave them blank and nothing
identifying is stored. Each record also carries the model version that made
the prediction. Patient data must be anonymised before entry, and consent
must be obtained for storing it — see the README privacy note (India's
DPDP Act, 2023).
