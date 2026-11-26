# Model Card — PCOD/PCOS Prediction

| | |
|---|---|
| Model version | 1.1.0 (trainer run 2026-11-26, Java 25 / Smile 6.3.0) |
| Training date | 2026-11-26T13:09:15.671909 |
| Dataset | Kaggle, Prasoon Kottarathil, "Polycystic Ovary Syndrome (PCOS)" — 541 patients from 10 hospitals in Kerala, 45 columns. 41 input features after dropping identifiers and the junk column. Target `PCOS (Y/N)`: 0 = 364 (67.3%), 1 = 177 (32.7%). |
| Split | Stratified 80/20 train/test, seed 42. Train 433 rows, test 108 rows. Test split evaluated exactly once, after all tuning. |
| Tuning | 5-fold stratified cross-validation on the train split only (seed 42): hyperparameter grids, feature subsets from Random Forest importance rankings re-computed inside each CV fold (ranked on that fold's training part only), clipping on/off, minority oversampling on/off. Preprocessing re-fitted inside every fold. |
| Deployed model | Soft Voting Ensemble — chosen by best train CV, not by test accuracy. |
| Verified history rows used | 0 (only doctor-verified outcomes are ever used for training) |
| Library / stack | Java 25 (LTS), Spring Boot 4.1.2, Smile 6.3.0 (Java ML). The metrics table below is the Smile 6.3.0 re-train (2026-11-26) on the same split, seed and protocol as run 8. Previous run: run 8 (2026-11-25, Java 17 / Spring Boot 3.2.5 / Smile 3.1.1) — deployed ensemble 88.89% (P 89.66%, R 74.29%, ROC-AUC 0.9252, threshold 0.50, CV 93.07%); Gradient Boosting 88.89%, Random Forest 87.04%, Bagging 87.04%, AdaBoost 86.11%; screening 82.41% (R 82.86% at threshold 0.25); baselines LR 85.19%, DT 82.41%, SVM 83.33%, KNN 84.26%, Majority 67.59%. The headline deployed accuracy is unchanged at 88.89%; per-model numbers moved slightly (see the run-history note under the results table). |

## Measured test results (held-out test set, n = 108)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Train CV | TP/FP/FN/TN | 95% CI (accuracy) |
|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 87.96% | 95.83% | 65.71% | 78.02% | 0.930 | 0.883 | 0.1017 | 90.99% | 23/1/12/72 | 80.6–93.5% |
| Bagging | 87.04% | 86.21% | 71.43% | 78.13% | 0.900 | 0.848 | 0.1075 | 89.15% | 25/4/10/69 | 80.6–92.6% |
| AdaBoost | 86.11% | 81.25% | 74.29% | 77.61% | 0.928 | 0.888 | 0.1771 | 89.15% | 26/6/9/67 | 78.7–91.7% |
| Gradient Boosting | 87.96% | 84.38% | 77.14% | 80.60% | 0.945 | 0.906 | 0.0908 | 91.22% | 27/5/8/68 | 81.5–93.5% |
| Soft Voting Ensemble | 88.89% | 92.59% | 71.43% | 80.65% | 0.927 | 0.882 | 0.0994 | 91.92% | 25/2/10/71 | 82.4–94.4% |
| Gradient Boosting (Screening) | 82.41% | 72.22% | 74.29% | 73.24% | 0.880 | 0.758 | 0.1384 | 82.68% | 26/10/9/63 | 75.0–88.9% |

TP/FP/FN/TN are test-set confusion counts at each model's decision threshold
(the deployed and screening models use their tuned thresholds; the rest use 0.5).
The 95% CI is a bootstrap interval (1000 resamples, seed 42) for test accuracy.
With only 108 test rows, one patient is worth 0.93 percentage
points — a difference of 2–3 patients (e.g. 86% vs 89%) is not significant.

**Run history — what changed vs run 8 (Smile 3.1.1, 2026-11-25).** Same
dataset, split, seed and tuning protocol; only the library changed
(Smile 3.1.1 → 6.3.0). The deployed ensemble's accuracy is unchanged at
88.89% (threshold re-tuned 0.50 → 0.55 on train CV, so precision rose
89.66% → 92.59% and recall fell 74.29% → 71.43%; ROC-AUC 0.9252 → 0.9272;
train CV 93.07% → 91.92%). Gradient Boosting 88.89% → 87.96% (one more
false positive); Random Forest 87.04% → 87.96% (one fewer false positive);
Bagging, Logistic Regression, Decision Tree, KNN and Majority are identical.
AdaBoost accuracy is unchanged at 86.11% with a different confusion mix
(recall 68.57% → 74.29%). The linear SVM jumped 83.33% → 87.96% — Smile 6
solves the SVM differently, same C=0.1. The screening model keeps 82.41%
accuracy but its tuned threshold moved 0.25 → 0.30, so its recall fell
82.86% → 74.29% (precision 69.05% → 72.22%): the high-recall screening
operating point is weaker in this run. The IEEE paper (sent to the guide)
carries the run-8 numbers, so its per-model figures no longer match this
code exactly — only the headline 88.89% deployed accuracy still does.

## Decision thresholds (tuned on train CV out-of-fold probabilities only, never on the test set)

| Model | Tuned threshold | OOF precision / recall / F1 | Test at tuned threshold (acc / P / R / F1) | Test at 0.5 (acc / P / R / F1) |
|---|---|---|---|---|
| Soft Voting Ensemble | 0.55 | 94.21% / 80.28% / 86.69% | 88.89% / 92.59% / 71.43% / 80.65% | 88.89% / 89.66% / 74.29% / 81.25% |
| Gradient Boosting (Screening) | 0.30 | 67.26% / 79.58% / 72.90% | 82.41% / 72.22% / 74.29% / 73.24% | 80.56% / 73.33% / 62.86% / 67.69% |

Thresholds maximise F1 on out-of-fold train probabilities; ties are broken
toward the lower threshold (higher recall). Recall matters most for a
screening tool: a missed case costs more than a false alarm.

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
labs, it still reaches 82.41% test accuracy with 74.29% recall.

## Serving & retraining

- **Confidence is the probability of the predicted class.** The deployed
  ensemble uses threshold 0.55, the screening model 0.30 — so a screening
  prediction can be PCOS with a probability below 0.5 (for example, a
  screening probability of 0.31 is a PCOS prediction at the 0.30 threshold),
  and the response reports that same probability as its confidence, not
  `max(p, 1-p)`.
- **`modelAccuracy` in the predict response is the model's test accuracy**
  on the 108-row held-out set (with its 95% CI and a note saying so) — it
  describes the model, not the individual prediction. When 21 or more of
  the 41 fields were mean-imputed, the response also carries a `warning`
  that the result rests mostly on training averages.
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
  503 (disabled) instead of running unprotected.
- **Only verified, complete rows retrain.** Retraining uses only
  doctor-verified outcomes, and a verified row is used only if none of its
  stored fields were mean-imputed at prediction time (tracked per record
  as `imputedFields`); rows with any imputed field — or written before
  that field was tracked — are excluded, so the model never learns from
  training averages presented as real values.

## Baselines (same split, same train-only CV protocol)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Train CV | TP/FP/FN/TN | 95% CI (accuracy) |
|---|---|---|---|---|---|---|---|---|---|---|
| Majority Class | 67.59% | 0.00% | 0.00% | 0.00% | 0.500 | 0.324 | 0.3241 | 67.21% | 0/0/35/73 | 58.3–75.9% |
| Logistic Regression | 85.19% | 82.76% | 68.57% | 75.00% | 0.880 | 0.860 | 0.1151 | 87.07% | 24/5/11/68 | 77.8–91.7% |
| Decision Tree | 82.41% | 80.77% | 60.00% | 68.85% | 0.819 | 0.696 | 0.1428 | 84.76% | 21/5/14/68 | 75.0–89.8% |
| SVM (Linear) | 87.96% | 80.56% | 82.86% | 81.69% | 0.866 | 0.723 | 0.1204 | 89.38% | 29/7/6/66 | 82.4–93.5% |
| KNN | 84.26% | 82.14% | 65.71% | 73.02% | 0.814 | 0.716 | 0.1430 | 85.91% | 23/5/12/68 | 77.8–90.7% |

The majority-class baseline (always predict "no PCOS") scores 67.59% on
the test set by predicting the majority class every time — any useful model
must beat it.
The linear SVM outputs hard decisions (decision-function sign only, no
calibrated probabilities), so its ROC-AUC/Brier reflect 0/1 scores.

## Top features (train-only importance, mean decrease in impurity)

Random Forest (all 41): follicleR 286.1, follicleL 172.4, amh 109.2, fshLh 99.3, vitD3 98.4, waistHipRatio 97.0, lh 95.1, bmi 94.0, prl 93.4, tsh 91.6

Gradient Boosting (all 41): follicleR 1199.5, follicleL 285.9, hairGrowth 241.6, skinDarkening 206.0, weightGain 128.6, cycle 105.7, fshLh 103.9, amh 84.9, lh 68.3, cycleLength 51.1

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
  models are not significant.
- **No refit on all 541 rows — by design.** The shipped models were
  trained on the 433-row train split only, so the reported test metrics
  describe the exact artefact being served. Refitting on all 541 rows
  after evaluation would use ~25% more data but then the test metrics
  would no longer describe the shipped model.
- Single-centre region (Kerala hospitals) and a single dataset — no external
  validation cohort yet. Performance on other populations is unknown.
- Recall is the weaker side: the deployed model misses some true cases
  (see the confusion matrix). It must not be used to rule PCOS out.
- Probabilities are not perfectly calibrated (see Brier scores); they are
  risk scores, not clinical probabilities.
- Missing values are filled with training means; a prediction based mostly
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
(`prediction_records` table) and nowhere else. Patient data must be
anonymised before entry, and consent must be obtained for storing it —
see the README privacy note (India's DPDP Act, 2023).
