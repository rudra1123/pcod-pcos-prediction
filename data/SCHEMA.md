# PCOS Dataset — Schema & Provenance

Project: **Cognitive Analytics in Healthcare — PCOD/PCOS prediction** (Java + Spring Boot final-year project).
Synopsis claim: 541 patient instances, 45 attributes, UCI ML Repository, "PCOS data without infertility".

## Verdict: EXACT MATCH on size — 541 rows × 45 columns

Verified on disk 2026-11-25. No difference from the synopsis on row/column counts.

## Sources tried

| # | Source | Result |
|---|--------|--------|
| 1 | Kaggle dataset page — https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos (Prasoon Kottarathil, "Polycystic Ovary Syndrome (PCOS)") | Page reachable (HTTP 200). This is the canonical source. Note: the synopsis says "UCI ML Repository", but the dataset as circulated is the Kaggle copy of the same Kerala-hospitals data (10 hospitals, Kerala, India); the UCI listing mirrors this Kaggle dataset. |
| 2 | Kaggle API download (no login) — https://www.kaggle.com/api/v1/datasets/download/prasoonkottarathil/polycystic-ovary-syndrome-pcos | **SUCCESS — final source used.** Returned a signed Google Cloud Storage URL, HTTP 200, `archive.zip` 122,546 bytes. Contents: `PCOS_data_without_infertility.xlsx` (125,274 bytes, dated 2020-07-11) and `PCOS_infertility.csv` (15,378 bytes). |
| 3 | GitHub mirror CSV — https://raw.githubusercontent.com/letisalba/DATA_608/master/Final%20Project/csv/PCOSData_without_infertility.csv | HTTP 200, 84,880 bytes, 541×45 — **BUT CORRUPTED: do not use.** 1,393 cells contain Excel `#NAME?` formula errors (BMI 299, FSH/LH 532, Waist:Hip Ratio 532). Only the 3 derived columns differ from the Kaggle original; all other cells match. |
| 4 | GitHub mirror — https://raw.githubusercontent.com/letisalba/DATA_608/master/Final%20Project/csv/PCOS_infertility.csv | HTTP 200, 15,378 bytes, md5 identical to the file inside the Kaggle archive (`0d3d1baff02529c2ffdb9260710f783a`). Confirms the mirror lineage. |
| 5 | GitHub — https://raw.githubusercontent.com/ashna-verma/PCOS-Detection-System/main/PCOS_data_without_infertility.xlsx | HTTP 200, 125,274 bytes, md5 identical to the Kaggle archive's xlsx (`89427c930767c314a0a69387c51672ba`). Its first sheet is an "Instructions" sheet, which is why a naive read shows 20×4 — the data is on the second sheet, `Full_new`. |
| 6 | GitHub — https://raw.githubusercontent.com/ashna-verma/PCOS-Detection-System/main/pcos_data.csv | HTTP 404 — processed/cleaned file not present at that path. Not needed. |

## Final source used

**Kaggle archive** (URL 2 above), file `PCOS_data_without_infertility.xlsx`, sheet **`Full_new`**, converted to CSV.
The saved `pcos_dataset.csv` is that sheet exported verbatim (header + 541 rows), so the derived columns keep their real numeric values instead of the `#NAME?` errors in the GitHub CSV mirror.

File checksums (md5):
- `pcos_dataset.csv` (saved): derived from xlsx below; 99,138 bytes
- `PCOS_data_without_infertility.xlsx` (saved original): `89427c930767c314a0a69387c51672ba`
- `PCOS_infertility.csv` (saved, 6-column infertility subset): `0d3d1baff02529c2ffdb9260710f783a`

## Verified shape

- **Rows: 541** (Sl. No 1–541, all unique; Patient File No. all unique; 0 duplicated rows)
- **Columns: 45** (see list below)
- Age range 20–48 years; Weight 31–108 kg; Height 137–180 cm.

## Target column

- **`PCOS (Y/N)`** — binary, 0 = no PCOS, 1 = PCOS diagnosed. No missing values.
- Distribution: **0 → 364 (67.3%), 1 → 177 (32.7%)**. Class imbalance ~2:1; use stratified train/test split.

## Column list (verbatim headers, in file order — note irregular spaces/casing, e.g. ` Age (yrs)`, `Height(Cm) `, `Marraige Status (Yrs)` [sic])

1. Sl. No
2. Patient File No.
3. PCOS (Y/N) — **target**
4. ` Age (yrs)`
5. Weight (Kg)
6. `Height(Cm) `
7. BMI
8. Blood Group
9. `Pulse rate(bpm) `
10. RR (breaths/min)
11. Hb(g/dl)
12. Cycle(R/I)
13. Cycle length(days)
14. Marraige Status (Yrs)
15. Pregnant(Y/N)
16. No. of aborptions
17. `  I   beta-HCG(mIU/mL)`
18. `II    beta-HCG(mIU/mL)`
19. FSH(mIU/mL)
20. LH(mIU/mL)
21. FSH/LH
22. Hip(inch)
23. Waist(inch)
24. Waist:Hip Ratio
25. TSH (mIU/L)
26. AMH(ng/mL)
27. PRL(ng/mL)
28. Vit D3 (ng/mL)
29. PRG(ng/mL)
30. RBS(mg/dl)
31. Weight gain(Y/N)
32. hair growth(Y/N)
33. Skin darkening (Y/N)
34. Hair loss(Y/N)
35. Pimples(Y/N)
36. Fast food (Y/N)
37. Reg.Exercise(Y/N)
38. BP _Systolic (mmHg)
39. BP _Diastolic (mmHg)
40. Follicle No. (L)
41. Follicle No. (R)
42. Avg. F size (L) (mm)
43. Avg. F size (R) (mm)
44. Endometrium (mm)
45. Unnamed: 44 — **junk column, see below**

## Missing values (in saved CSV)

| Column | Missing |
|---|---|
| Marraige Status (Yrs) | 1 of 541 |
| Fast food (Y/N) | 1 of 541 |
| Unnamed: 44 | 539 of 541 (only 2 junk values: `.` and `7`) |
| All other 42 columns | 0 |

Total missing cells: 541, of which 539 are the junk column. Real missing data is just 2 cells — impute with the mean/mode per the synopsis methodology (mean imputation).

## Differences / caveats vs the synopsis

1. **No size difference: exactly 541 × 45 as claimed.**
2. **Column 45 (`Unnamed: 44`) is not a real attribute** — it is a trailing empty column from the Excel export (539 empty, values `.` and `7` in 2 rows). Real attributes are 44 (columns 1–44), of which 2 are identifiers (Sl. No, Patient File No.) and 1 is the target, leaving 41 input features. Papers citing this dataset variously say "45 columns", "44 attributes" or "41 features" — all refer to this same file with different counting conventions.
3. **Source naming:** the synopsis says "UCI ML Repository"; the downloadable original is the Kaggle dataset by Prasoon Kottarathil (same data, 10 Kerala hospitals). Cite Kaggle as the source in the report.
4. **Derived columns are real values here:** BMI, FSH/LH and Waist:Hip Ratio are fully numeric in the saved CSV (541/541 each). Do NOT use the GitHub CSV mirror — its copies of these 3 columns are `#NAME?` errors.
5. **Spelling/headers preserved verbatim** (`Marraige`, `aborptions`, leading/trailing spaces). The Java loader should trim headers and map them to clean field names; do not rename the file itself.
6. **Outliers present** (e.g. FSH up to 5052, Vit D3 up to 6014.66, beta-HCG up to 32460.97) — the synopsis methodology (StandardScaler, feature selection) should handle these; consider clipping/winsorising and documenting it.

## Recommended Java preprocessing (for the Spring Boot app)

- Drop: `Sl. No`, `Patient File No.`, `Unnamed: 44` (identifiers/junk — never model inputs).
- Target: `PCOS (Y/N)` (0/1).
- Impute the 2 genuinely missing cells (mean for Marraige Status, mode for Fast food) — or drop those 2 rows; document the choice.
- 41 remaining input features; apply the synopsis feature-importance selection, then StandardScaler, then the ensemble (Random Forest / Bagging / AdaBoost / Gradient Boosting).

## Files in this folder

- `pcos_dataset.csv` — verified dataset (541×45), from the Kaggle original.
- `PCOS_data_without_infertility.xlsx` — original Kaggle file (sheets: Instructions, Full_new).
- `PCOS_infertility.csv` — 6-column infertility subset from the same Kaggle archive (541×6; not used by the synopsis).
- `sample_patients.json` — 3 real de-identified sample rows for predict-endpoint testing (target removed from input, true label stored separately).
- `SCHEMA.md` — this file.
