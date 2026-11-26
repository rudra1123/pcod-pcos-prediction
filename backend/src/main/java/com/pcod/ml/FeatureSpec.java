package com.pcod.ml;

import java.util.List;

/**
 * The 41 input features, in the exact order used everywhere (training,
 * metrics, /api/features, /api/predict). Keys are clean snake_case.
 *
 * <p>Each feature maps to one verbatim CSV header (headers are matched after
 * trimming, because the source file has irregular leading/trailing spaces and
 * the spellings "Marraige" / "aborptions"). Dropped columns: "Sl. No",
 * "Patient File No." (identifiers), "Unnamed: 44" (junk column, 539/541
 * empty) and the target "PCOS (Y/N)". See data/SCHEMA.md.</p>
 */
public final class FeatureSpec {

    public record Feature(String key, String label, String csvHeader, String type,
                            double min, double max) {}

    /**
     * Plausible range for server-side validation: a value outside it is
     * rejected with HTTP 400 instead of being silently clipped (age 500 or
     * FSH 9000 must never reach the model). These are the same bounds the
     * frontend already uses (frontend/src/features.js): the observed
     * dataset min/max, except pulse rate (raw min 13 bpm) and blood
     * pressure (raw min 12/8 mmHg), which were clipped to plausible
     * clinical bounds. They are data ranges, not diagnosis cut-offs.
     */

    public static final List<Feature> FEATURES = List.of(
            new Feature("age", "Age (years)", "Age (yrs)", "number", 20, 48),
            new Feature("weightKg", "Weight (kg)", "Weight (Kg)", "number", 31, 108),
            new Feature("heightCm", "Height (cm)", "Height(Cm)", "number", 137, 180),
            new Feature("bmi", "BMI", "BMI", "number", 12.42, 38.9),
            new Feature("bloodGroup", "Blood group (code)", "Blood Group", "number", 11, 18),
            new Feature("pulseRate", "Pulse rate (bpm)", "Pulse rate(bpm)", "number", 40, 120),
            new Feature("rr", "Respiratory rate (breaths/min)", "RR (breaths/min)", "number", 16, 28),
            new Feature("hb", "Haemoglobin (g/dl)", "Hb(g/dl)", "number", 8.5, 14.8),
            new Feature("cycle", "Cycle (R/I)", "Cycle(R/I)", "number", 2, 5),
            new Feature("cycleLength", "Cycle length (days)", "Cycle length(days)", "number", 0, 12),
            new Feature("marriageYears", "Marriage duration (years)", "Marraige Status (Yrs)", "number", 0, 30),
            new Feature("pregnant", "Pregnant (0 = No, 1 = Yes)", "Pregnant(Y/N)", "number", 0, 1),
            new Feature("abortions", "No. of abortions", "No. of aborptions", "number", 0, 5),
            new Feature("betaHcg1", "Beta-HCG I (mIU/mL)", "I   beta-HCG(mIU/mL)", "number", 1.3, 32461),
            new Feature("betaHcg2", "Beta-HCG II (mIU/mL)", "II    beta-HCG(mIU/mL)", "number", 0.99, 25000),
            new Feature("fsh", "FSH (mIU/mL)", "FSH(mIU/mL)", "number", 0.21, 5052),
            new Feature("lh", "LH (mIU/mL)", "LH(mIU/mL)", "number", 0.02, 2018),
            new Feature("fshLh", "FSH/LH ratio", "FSH/LH", "number", 0, 1372.83),
            new Feature("hip", "Hip (inch)", "Hip(inch)", "number", 26, 48),
            new Feature("waist", "Waist (inch)", "Waist(inch)", "number", 24, 47),
            new Feature("waistHipRatio", "Waist:Hip ratio", "Waist:Hip Ratio", "number", 0.76, 0.98),
            new Feature("tsh", "TSH (mIU/L)", "TSH (mIU/L)", "number", 0.04, 65),
            new Feature("amh", "AMH (ng/mL)", "AMH(ng/mL)", "number", 0.1, 66),
            new Feature("prl", "PRL (ng/mL)", "PRL(ng/mL)", "number", 0.4, 128.24),
            new Feature("vitD3", "Vitamin D3 (ng/mL)", "Vit D3 (ng/mL)", "number", 0, 6014.66),
            new Feature("prg", "PRG (ng/mL)", "PRG(ng/mL)", "number", 0.05, 85),
            new Feature("rbs", "Random blood sugar (mg/dl)", "RBS(mg/dl)", "number", 60, 350),
            new Feature("weightGain", "Weight gain (0 = No, 1 = Yes)", "Weight gain(Y/N)", "number", 0, 1),
            new Feature("hairGrowth", "Hair growth (0 = No, 1 = Yes)", "hair growth(Y/N)", "number", 0, 1),
            new Feature("skinDarkening", "Skin darkening (0 = No, 1 = Yes)", "Skin darkening (Y/N)", "number", 0, 1),
            new Feature("hairLoss", "Hair loss (0 = No, 1 = Yes)", "Hair loss(Y/N)", "number", 0, 1),
            new Feature("pimples", "Pimples (0 = No, 1 = Yes)", "Pimples(Y/N)", "number", 0, 1),
            new Feature("fastFood", "Fast food (0 = No, 1 = Yes)", "Fast food (Y/N)", "number", 0, 1),
            new Feature("regularExercise", "Regular exercise (0 = No, 1 = Yes)", "Reg.Exercise(Y/N)", "number", 0, 1),
            new Feature("bpSystolic", "BP systolic (mmHg)", "BP _Systolic (mmHg)", "number", 80, 180),
            new Feature("bpDiastolic", "BP diastolic (mmHg)", "BP _Diastolic (mmHg)", "number", 40, 120),
            new Feature("follicleL", "Follicle count (left)", "Follicle No. (L)", "number", 0, 22),
            new Feature("follicleR", "Follicle count (right)", "Follicle No. (R)", "number", 0, 20),
            new Feature("avgFollicleSizeL", "Avg. follicle size (left, mm)", "Avg. F size (L) (mm)", "number", 0, 24),
            new Feature("avgFollicleSizeR", "Avg. follicle size (right, mm)", "Avg. F size (R) (mm)", "number", 0, 24),
            new Feature("endometrium", "Endometrium (mm)", "Endometrium (mm)", "number", 0, 18)
    );

    /** Number of input features. */
    public static final int N_FEATURES = FEATURES.size(); // 41

    /**
     * Screening-mode features: obtainable WITHOUT any laboratory test or
     * ultrasound — age, body measurements, cycle pattern, symptoms and
     * lifestyle only. Excluded on purpose: every hormone / blood lab
     * (beta-HCG, FSH, LH, FSH/LH, TSH, AMH, PRL, Vit D3, PRG, RBS, Hb,
     * blood group — all need a blood sample) and every ultrasound measure
     * (follicle counts, follicle sizes, endometrium). A model trained on
     * these 23 features alone is what makes "early detection before a
     * doctor visit" true: it needs no clinic visit at all.
     */
    public static final java.util.Set<String> SCREENING_FEATURES = java.util.Set.of(
            "age", "weightKg", "heightCm", "bmi",
            "pulseRate", "rr",
            "cycle", "cycleLength", "marriageYears", "pregnant", "abortions",
            "hip", "waist", "waistHipRatio",
            "weightGain", "hairGrowth", "skinDarkening", "hairLoss", "pimples",
            "fastFood", "regularExercise",
            "bpSystolic", "bpDiastolic");

    /** Indices of the screening features, in FeatureSpec order. */
    public static int[] screeningIndices() {
        return FEATURES.stream()
                .filter(f -> SCREENING_FEATURES.contains(f.key()))
                .mapToInt(f -> FEATURES.indexOf(f))
                .toArray();
    }

    public static boolean isScreening(String key) {
        return SCREENING_FEATURES.contains(key);
    }

    /**
     * Minimum fields POST /api/predict requires. All 11 are non-invasive
     * (questionnaire / measurement only) and all are screening features:
     * a basic identifier (age, BMI), the cycle pattern, and the visible
     * symptoms. A request missing any of them is a 400 listing the missing
     * fields — a prediction from almost no data would be meaningless, and
     * an empty request must never return a confident prediction.
     */
    public static final List<String> REQUIRED_FEATURES = List.of(
            "age", "bmi", "cycle", "cycleLength",
            "weightGain", "hairGrowth", "skinDarkening", "hairLoss", "pimples",
            "fastFood", "regularExercise");

    public static boolean isRequired(String key) {
        return REQUIRED_FEATURES.contains(key);
    }

    /** Target column name used inside Smile DataFrames. */
    public static final String TARGET = "target";

    /** All DataFrame column names: 41 feature keys + target. */
    public static final String[] COLUMN_NAMES;

    static {
        if (N_FEATURES != 41) {
            throw new IllegalStateException("Expected 41 features, found " + N_FEATURES);
        }
        COLUMN_NAMES = new String[N_FEATURES + 1];
        for (int i = 0; i < N_FEATURES; i++) {
            COLUMN_NAMES[i] = FEATURES.get(i).key();
        }
        COLUMN_NAMES[N_FEATURES] = TARGET;
    }

    public static int indexOf(String key) {
        for (int i = 0; i < FEATURES.size(); i++) {
            if (FEATURES.get(i).key().equals(key)) {
                return i;
            }
        }
        return -1;
    }

    private FeatureSpec() {}
}
