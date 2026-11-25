# Larger PCOS Dataset Search — Provenance Note

**Date:** 2026-11-25
**Project:** Cognitive Analytics in Healthcare — PCOD/PCOS prediction (Java + Spring Boot)
**Baseline already held:** `pcos_dataset.csv` — **541 rows × 45 cols**, Kaggle Prasoon Kottarathil
PCOS dataset, target `PCOS (Y/N)` (0 = 364, 1 = 177, 32.7% positive). Verified on disk;
see `SCHEMA.md`. (The synopsis says "UCI ML Repository" — no PCOS dataset was found on
UCI; the downloadable original is the Kaggle copy. The existing `PCOS_data_without_infertility.xlsx`
still hashes to md5 `89427c930767c314a0a69387c51672ba`, exactly as recorded in `SCHEMA.md`, and
`pcos_dataset.csv` is unchanged at 99,138 bytes.)

## Verdict (read first)

**No verified genuine larger real-patient dataset was found. Training stays on the verified
541-row dataset.**

- There is **no public 50k / 1 lakh-patient labelled tabular PCOS dataset**. Nothing at that
  scale exists in any source searched. Do not claim such numbers anywhere in the report.
- The two apparently larger same-schema files (2,000 and 2,539 rows) are **not new patients**:
  forensic checks prove both are jittered/resampled copies of the same 541 Kerala patients
  (details §2). Merging either with the 541 would leak near-duplicates across train/test and
  dishonestly inflate accuracy. **Do not merge them. Do not train on them as if they were
  new patients.**
- The largest genuinely larger tabular candidate is **`dataset066/pcos-clinical-dataset` —
  10,000 rows × 16 cols**, target `PCOS` (0 = 6,039, 1 = 3,961), verified on disk for shape,
  target distribution and its first 10 columns. BUT its author states **no collection source,
  no hospital, no paper**; it appeared on Kaggle on 2026-11-21 with 4 downloads at time of
  checking, and its remaining 6 column names were not verified. Its provenance (real vs
  synthetic/generated) is **unverified**, so it must **not** be presented as 10,000 real
  patients and was **not** copied into this folder as a training file. It is recorded here
  (§3) with its source URL and checksum so the decision on it can be made explicitly.
- The largest verified *independent real* sample found is a Bangladeshi survey dataset,
  667 rows — but with a different 30-column schema (no FSH/LH/AMH/follicle counts), so it
  cannot be merged with the 541-row schema (§4).

## How the search was done

- **Kaggle search API** (`api/v1/datasets/list?search=pcos`, pages 1–3, and `search=polycystic`,
  page 1): 60+ datasets enumerated. Every tabular clinical candidate was downloaded via the
  public Kaggle download API (no login) and verified on disk with pandas before any number
  below was recorded.
- **Web search:** queries for "PCOS dataset 10000 / 5000 / 2000 patients", "UCI Machine
  Learning Repository PCOS", "GitHub PCOS dataset CSV", "IEEE DataPort PCOS", "PhysioNet /
  Zenodo / Dataverse PCOS tabular dataset", "Kaggle Soares PCOS 3000", "PCOS-External Zafar
  320 instances".
- **UCI, PhysioNet, Zenodo, Harvard Dataverse, IEEE DataPort, GitHub, papers:** checked as
  reported in §5–§6.
- **QC re-check (this note):** every URL cited below was re-fetched with HTTP HEAD on
  2026-11-25; all Kaggle/IEEE/Zenodo URLs returned **HTTP 200**. Two secondary sources did
  NOT re-resolve and are cited as search-index findings only (DMPTool plan PDF: HTTP 400;
  Frontiers multicentre paper: HTTP 503). Every row/column count labelled "verified on disk"
  was computed from the downloaded file, not copied from a web page. Counts were
  re-added by hand (target classes sum to row count in every case).

## 1. Candidates verified on disk

| # | Dataset (source) | Rows × cols (verified) | Target column & distribution (verified) | Schema vs 541-row set | Verdict |
|---|---|---|---|---|---|
| 0 | **Baseline** — Prasoon Kottarathil, Kaggle `prasoonkottarathil/polycystic-ovary-syndrome-pcos` | 541 × 45 | `PCOS (Y/N)`: 0 = 364, 1 = 177 | — (reference) | **Use this** |
| 1 | Divya, Kaggle `cm037divya/pcos-dataset` — file `PCOS_extended_dataset.csv` | 2,000 × 44 | `PCOS (Y/N)`: 0 = 1,392, 1 = 608 (30.4% +ve) | Identical 44 headers, verbatim incl. `Marraige Status (Yrs)` typo; only spelling fix `No. of abortions` vs orig `No. of aborptions`; no junk col | **REJECT — augmented copy of the 541 (§2)** |
| 2 | Bhargava Prashant, Kaggle `bhargavaprashant/pcos-final-dataset` — file `pcos final dataset.csv` | 2,539 × 44 | `PCOS (Y/N)`: 0 = 1,755, 1 = 784 (30.9% +ve) | Same 44 headers as #1 | **REJECT — augmented copy of the 541 (§2)** |
| 3 | Kaggle `dataset066/pcos-clinical-dataset` — file `pcos_dataset.csv` | **10,000 × 16** | `PCOS`: 0 = 6,039, 1 = 3,961 (39.6% +ve) | 16 cols; first 10 verified on disk: `Age, BMI, Waist_Hip_Ratio, Cycle_Length_Days, Follicle_Count_Left, Follicle_Count_Right, LH_mIU_mL, FSH_mIU_mL, LH_FSH_Ratio, AMH_ng_mL` — all overlap 541-row attributes; remaining 6 col names **not verified**; author description lists testosterone, fasting insulin, blood glucose, HOMA-IR, hirsutism score | **CANDIDATE — provenance unverified (§3); NOT saved as training data** |
| 4 | Samiksha Dalvi, Kaggle `samikshadalvi/pcos-diagnosis-dataset` — `pcos_dataset.csv` | 1,000 × 6 | `PCOS_Diagnosis`: 0 = 801, 1 = 199 (19.9% +ve) | Only 5 features: `Age, BMI, Menstrual_Irregularity, Testosterone_Level(ng/dL), Antral_Follicle_Count` — no FSH/LH/AMH/TSH, no follicle L/R, no cycle length. Values look generated (clean ranges, e.g. BMI 18–35) | Larger than 541 but **not schema-compatible**; cannot merge. Usable at most as an external check on the 2 shared features (Age, BMI) — not recommended as training data |
| 5 | Bangladeshi survey, Kaggle `asif13/pcos-bangladeshi-dataset` — `PCOS_Clean_Categorical.csv` / `PCOS_Clean_Encoded.csv` | 667 × 30 | `PCOS`: 1 = 510, 0 = 157 (76.5% +ve — note reversed imbalance) | Different schema: `Weight_kg, Height_m, BMI, Self_reported_overweight, Menstrual_regularity, Abdominal_pain, Family_history_PCOS, Hirsutism_face, Hirsutism_body, …` (first 10 cols verified; remaining 20 not verified). No hormones, no follicle counts | Genuinely independent sample, but **not schema-compatible**; cannot merge. 667 rows only |
| 6 | Aya Moheddine, Kaggle `ayamoheddine/pcos-dataset` — `PCOS_data.csv` | 541 × 44 | `PCOS (Y/N)`: 0 = 364, 1 = 177 | Same 44 cols as baseline minus junk col | Same 541 patients — duplicate, no gain |
| 7 | Prasad Bobby, Kaggle `prasadbobby/pcosdata` — `data without infertility _final.csv` | 541 × 43 | `PCOS (Y/N)`: 0 = 364, 1 = 177 | Same as baseline minus both beta-HCG columns and junk col | Same 541 patients — duplicate, no gain |
| 8 | Shreyas Vedpathak, Kaggle `shreyasvedpathak/pcos-dataset` — `PCOS_data.csv` | 541 × 45 | `PCOS (Y/N)`: 0 = 364, 1 = 177 | Same headers as baseline | Same 541 patients — duplicate, no gain |
| 9 | Monamehrun, Kaggle `monamehrun/pcos-cleaned-dataset` — `pcos_cleaned.csv` | 541 × 49 | `PCOS`: 0 = 364, 1 = 177 | Cleaned/renamed version of the 541 (extra engineered cols) | Same 541 patients — duplicate, no gain |
| 10 | Skylar Lorenah, Kaggle `skylarlorenah/pcos-analysis-datasets` | `PCOS_infertility.csv` 541 × 6; `pcos_dataset.csv` 1,000 × 6 | Both: 0 = 364/801, 1 = 177/199 resp. | Copies of baseline infertility subset + Dalvi #4 | Duplicates of #0 and #4 — no gain |

Checksums (downloaded zips, `/tmp/pcos_large/`, sha256):
- Divya: `d19d89560d1efbe3eab2346babc79f6235046212ab1ff1cc75859557b7fe2865`
- Bhargava: `cfeea9edc87f475fdf7fb9ae56fe6fb0a139002126322e47456a3a655819375d`
- dataset066: `ee796dddee2249e798668b71e56daa37203f3f522088b05445498b0f4974a550`
- Dalvi: `cb59e83ad4540439457ddb5dca7e0772e9c87af38b44d7ad0cabd5b7c4709d0b`
- Bangladeshi: `16dca386fa83eed66e16203393079ce29fd2516c95e04833c9423abf502aea4e`

## 2. Why the 2,000- and 2,539-row files are NOT new patients (forensic evidence)

Both files claim more rows, but every test points to the same 541 source patients,
resampled with small jitter (i.e. synthetic augmentation, not new enrolment):

- **Patient File No. gives it away.** Divya: only **530 distinct** `Patient File No.` values,
  all in the range 1–541, each repeated 1–10× (56 IDs once, 85 twice, 120 three times, 94 four
  times, 78 five times, 55 six times, 25 seven times, 11 eight times, 5 nine times, 1 ten
  times; sums to 530 IDs / 2,000 rows). 11 of the original 541 patients are missing entirely.
  Bhargava: `Sl. No` runs 1–2,539 (unique) but `Patient File No.` has only **541 distinct**
  values — all 541 original IDs, each reused ~4.7× on average.
- **Near-duplicate distances.** Standardised Euclidean nearest-neighbour distance from each
  extended row to its closest original 541 row: Divya median **0.68**, 96.4% within 1.0,
  max 1.19; Bhargava median **0.61**, 97.1% within 1.0, max 1.19. Genuine distinct patients
  in the original data sit median **4.65** from their nearest neighbour. Extended rows hug
  original rows far too tightly to be independent patients.
- **The original typos propagate.** The 541-row file contains two data-entry typos:
  `1.99.` in `II beta-HCG(mIU/mL)` (1 cell) and `a` in `AMH(ng/mL)` (1 cell). Divya carries
  5× `1.99.` and 3× `a`; Bhargava carries 6× `1.99.` and 4× `a`. New real data would not
  reproduce those exact typos multiple times.
- **Labels never flip for a reused ID.** Every reused `Patient File No.` keeps the original
  patient's target label in both files (0 mismatches) — exactly what resampling around a
  source row produces.
- **Corroboration in print.** The Divya file's counts match its citing papers exactly:
  "44 feature columns and 2000 rows… positive 608 and negative 1392" (Taher, *Multilayer
  stacking for polycystic ovary syndrome diagnosis*, IJ-AI 14(3), June 2025, pp. 1968–1975;
  DOI 10.11591/ijai.v14.i3.pp1968-1975 — PDF re-fetched HTTP 200). No paper states these are
  2,000 distinct enrolled patients, and none gives a hospital/source for the extra ~1,459 rows.

**Consequence:** merging either file with the 541 would put jittered copies of the same
patients in both train and test — a textbook leakage that inflates accuracy. The files were
therefore **not saved** into this folder. (If augmentation is ever wanted, do it explicitly
inside the training pipeline, e.g. SMOTE on the training split only, and say so in the report —
do not present these files as new patients.)

## 3. The 10,000-row candidate — dataset066 (provenance unverified, NOT saved)

- **Source:** https://www.kaggle.com/datasets/dataset066/pcos-clinical-dataset (HTTP 200 on
  re-check 2026-11-25). Title "PCOS Clinical Dataset", subtitle "PCOS Prediction Using
  Clinical, Hormonal and Metabolic Health Indicators". Licence CC BY-NC-SA 4.0.
  Uploaded/updated **2026-11-21** (4 days before this search); **4 downloads** at time of
  checking.
- **Author description (Kaggle API metadata, verbatim substance):** "a structured tabular
  dataset containing 10,000 records and 16 columns, comprising 15 input features and one
  binary target variable… demographic, anthropometric, hormonal, metabolic, menstrual, and
  ovarian health indicators, such as age, body mass index (BMI), waist-to-hip ratio,
  menstrual cycle length, ovarian follicle counts, luteinizing hormone (LH),
  follicle-stimulating hormone (FSH), anti-Müllerian hormone (AMH), testosterone, fasting
  insulin, blood glucose, HOMA-IR, and hirsutism score."
- **Verified on disk (before the hold in §7):** file `pcos_dataset.csv`, **10,000 rows × 16
  cols**, target `PCOS` 0 = 6,039 / 1 = 3,961 (sums to 10,000; 39.6% positive — a much
  higher positive rate than the 32.7% in the Kerala data). First 10 columns verified:
  `Age, BMI, Waist_Hip_Ratio, Cycle_Length_Days, Follicle_Count_Left, Follicle_Count_Right,
  LH_mIU_mL, FSH_mIU_mL, LH_FSH_Ratio, AMH_ng_mL`. The remaining 6 column names were **not**
  verified against the file (see §7) — do not quote them as verified; the description's
  feature list above is the author's claim, not a verified column list.
- **Why it is not adopted:** the author gives **no data-collection source** — no hospital,
  no country, no enrolment dates, no diagnostic criteria, no linked paper. A 10,000-row
  clinical dataset with no provenance, uploaded days ago with 4 downloads, cannot be
  presented as 10,000 real patients in a final-year report. It may be synthetic/generated;
  that is **not proven** either way — it is simply **unverified**.
- **Where it is:** downloaded copy at `/tmp/pcos_large/dataset066_pcos-clinical-dataset.zip`
  (sha256 `ee796dddee2249e798668b71e56daa37203f3f522088b05445498b0f4974a550`). It can be
  re-downloaded any time from the Kaggle URL above. **Decision needed from the parent/user
  before any use:** either (a) accept it only as a clearly-labelled secondary/synthetic
  experiment after a full forensic check (distributions, duplicates, plausibility), or
  (b) leave training on the 541. It must never be merged with the 541 and described as
  "10,541 patients".

## 4. Other datasets found (smaller / incompatible / not downloadable)

- **IEEE DataPort — "PCOS data", Vemali & Pothala, DOI 10.21227/k3d8-tt94**
  (https://ieee-dataport.org/documents/pcos-data, HTTP 200): **267 entries, 22 attributes**
  (page claim) — a survey (1,038 women surveyed, 267 used) plus Kaggle data, lifestyle
  features only, label-encoded. Smaller than 541, different schema, and IEEE DataPort
  download requires an account/login. Not a larger compatible dataset.
- **Zenodo — IVF cohort, DOI 10.5281/zenodo.24338929** (HTTP 200): 93 women, **all with
  PCOS** (Rotterdam criteria), undergoing IVF; target is clinical pregnancy, not PCOS
  diagnosis. No non-PCOS controls — incompatible as a classification dataset.
- **Sci Rep 2025 external-validation set** (PubMed PMID 41022847; abstract via search index):
  "a publicly available dataset containing 320 instances and 18 diagnostic features" used
  only for external validation. The underlying file could not be traced to a downloadable
  link; Kaggle `galibbhai/pcos-external-zafar` ("PCOS-External_Zafar", xlsx, 2 sheets) was
  downloaded but its row/column counts were **not verified** (see §7) and no link between it
  and the 320-row set was found. Do not cite it as the 320-row set.
- **Multicentre hospital study, Frontiers in Endocrinology 2025/2026** (search-index finding;
  direct fetch returned HTTP 503 on re-check): 1,600 training + 283 external cases, 23 lab
  indicators — hospital data with **no public download**. Not usable.
- **"Soares dataset, approximately 3,000 rows"** (DMPTool plan 151779; direct fetch returned
  HTTP 400 on re-check, cited from search index only): no Kaggle slug, link or file was
  given, and no dataset by that name appears in the Kaggle search-API listing (60+ results
  for "pcos"/"polycystic"). **Untraceable — do not count it as an available dataset.**
- **119-patient survey dataset** (IJRTE, Nov 2019): smaller than 541, no verified public file.
- **UCI ML Repository:** no PCOS dataset page exists there (searched). The synopsis's "UCI"
  attribution is incorrect; cite Kaggle (Kottarathil) instead.
- **PhysioNet:** no PCOS tabular dataset found. **GitHub:** mirrors found are copies of the
  same 541 (incl. the corrupted CSV mirror with `#NAME?` errors in BMI / FSH/LH /
  Waist:Hip Ratio documented in `SCHEMA.md` — never use that mirror; the saved CSV here
  comes from the Kaggle xlsx, whose md5 still matches).
- **Ultrasound image datasets** (e.g. 3,856 images; 11,784 images; PCOSGen): image
  classification, not tabular clinical rows — out of scope for this Java tabular ensemble.
- **Unverified xlsx candidates** (downloaded, sheet names seen, row counts **not** verified —
  see §7, so no counts are claimed for them): `michaelmendiolasy/pcos-clinical-dataset`
  (author claims "more than 400 anonymized records" — page claim only),
  `mdabdulkaium/pcos-dataset` (description: "combines… publicly available PCOS research
  data and newly coll…" — truncated; likely derived from the 541),
  `mohamedyahya72/pcos-dataset-clinical` (file is literally named
  `PCOS_data_without_infertility.xlsx`, the 541-row original's filename),
  `galibbhai/pcos-external-zafar` (see above).

## 5. Merging safety

| Pair | Safe to merge? |
|---|---|
| 541 + Divya 2,000 | **No** — Divya rows are jittered copies of the same 541 patients (§2). Leakage. |
| 541 + Bhargava 2,539 | **No** — same reason (§2). |
| 541 + dataset066 10,000 | **Not without a provenance decision** (§3). Schemas overlap on ~10 features, but the 6 unverified columns, the 39.6% positive rate (vs 32.7%), and the absence of any source make silent merging indefensible. If ever used, keep it a separate, labelled experiment. |
| 541 + Dalvi 1,000 | **No** — only Age/BMI overlap; no hormones/follicles. Different (likely generated) data. |
| 541 + Bangladeshi 667 | **No** — different schema, different population/survey design, reversed class balance. |

## 6. Conclusion for the report

- Train and evaluate on the **verified 541-row dataset** (`pcos_dataset.csv`), as the synopsis
  specifies. Report its size honestly (541 patients, 41 input features after dropping IDs
  and the junk column).
- State in the report's data section that a search for larger public labelled PCOS data was
  carried out (Kaggle, UCI, IEEE DataPort, Zenodo, PhysioNet, GitHub, papers): the only
  larger same-schema files are resampled copies of the same 541 patients, and no public
  50k/1-lakh-patient labelled tabular PCOS dataset exists. This is a genuine, citable
  limitation — not a failure of the project.
- The 99% accuracy target (user, 2026-11-25) must be pursued by honest modelling on the 541
  (proper train/test split, no leakage from the augmented copies), not by inflating the
  dataset. Any accuracy reported from training on Divya/Bhargava/dataset066 would be
  misleading and must not appear in the report as a result on real patients.

## 7. QC notes & open items (accuracy over speed)

- **URL re-check 2026-11-25:** all Kaggle dataset URLs cited in §1, plus
  https://ieee-dataport.org/documents/pcos-data, https://doi.org/10.21227/k3d8-tt94,
  https://zenodo.org/records/24338929 and the Divya-paper PDF returned **HTTP 200**.
  The DMPTool plan PDF returned **HTTP 400** and the Frontiers multicentre article
  **HTTP 503** on re-check — both are therefore cited from the search index only, with
  that flagged, and no number from either is treated as verified.
- **Numbers re-checked:** every "verified on disk" count above was recomputed from the
  downloaded file and hand-checked (target classes sum to the row count in each case).
  The Divya/Bhargava per-ID count distributions were re-added (530 IDs / 2,000 rows;
  541 IDs / 2,539 rows).
- **Hold on further file handling:** one downloaded archive in this batch
  (`skylarlorenah/pcos-analysis-datasets`) contains a `pcos_screening_model.pkl` file.
  Pickle files can execute code on load, so it was never opened, and after that flag no
  further archive was extracted or loaded. That is why (a) the xlsx candidates in §4 have
  no verified row counts, (b) dataset066's last 6 column names are unverified, and
  (c) dataset066 was **not copied into this folder**. Nothing in this folder was changed
  except this note file. If the parent/user authorises it, the remaining checks are:
  verify dataset066's 6 remaining columns + run the §2 forensic tests on it, and verify
  the four xlsx candidates' shapes — then update this note.
- **No file was saved as a "large dataset"** because no candidate met the bar of
  *verified + compatible + genuine new patients*. No dataset was invented.
