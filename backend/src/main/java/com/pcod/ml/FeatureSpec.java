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

    public record Feature(String key, String label, String csvHeader, String type) {}

    public static final List<Feature> FEATURES = List.of(
            new Feature("age", "Age (years)", "Age (yrs)", "number"),
            new Feature("weightKg", "Weight (kg)", "Weight (Kg)", "number"),
            new Feature("heightCm", "Height (cm)", "Height(Cm)", "number"),
            new Feature("bmi", "BMI", "BMI", "number"),
            new Feature("bloodGroup", "Blood group (code)", "Blood Group", "number"),
            new Feature("pulseRate", "Pulse rate (bpm)", "Pulse rate(bpm)", "number"),
            new Feature("rr", "Respiratory rate (breaths/min)", "RR (breaths/min)", "number"),
            new Feature("hb", "Haemoglobin (g/dl)", "Hb(g/dl)", "number"),
            new Feature("cycle", "Cycle (R/I)", "Cycle(R/I)", "number"),
            new Feature("cycleLength", "Cycle length (days)", "Cycle length(days)", "number"),
            new Feature("marriageYears", "Marriage duration (years)", "Marraige Status (Yrs)", "number"),
            new Feature("pregnant", "Pregnant (0 = No, 1 = Yes)", "Pregnant(Y/N)", "number"),
            new Feature("abortions", "No. of abortions", "No. of aborptions", "number"),
            new Feature("betaHcg1", "Beta-HCG I (mIU/mL)", "I   beta-HCG(mIU/mL)", "number"),
            new Feature("betaHcg2", "Beta-HCG II (mIU/mL)", "II    beta-HCG(mIU/mL)", "number"),
            new Feature("fsh", "FSH (mIU/mL)", "FSH(mIU/mL)", "number"),
            new Feature("lh", "LH (mIU/mL)", "LH(mIU/mL)", "number"),
            new Feature("fshLh", "FSH/LH ratio", "FSH/LH", "number"),
            new Feature("hip", "Hip (inch)", "Hip(inch)", "number"),
            new Feature("waist", "Waist (inch)", "Waist(inch)", "number"),
            new Feature("waistHipRatio", "Waist:Hip ratio", "Waist:Hip Ratio", "number"),
            new Feature("tsh", "TSH (mIU/L)", "TSH (mIU/L)", "number"),
            new Feature("amh", "AMH (ng/mL)", "AMH(ng/mL)", "number"),
            new Feature("prl", "PRL (ng/mL)", "PRL(ng/mL)", "number"),
            new Feature("vitD3", "Vitamin D3 (ng/mL)", "Vit D3 (ng/mL)", "number"),
            new Feature("prg", "PRG (ng/mL)", "PRG(ng/mL)", "number"),
            new Feature("rbs", "Random blood sugar (mg/dl)", "RBS(mg/dl)", "number"),
            new Feature("weightGain", "Weight gain (0 = No, 1 = Yes)", "Weight gain(Y/N)", "number"),
            new Feature("hairGrowth", "Hair growth (0 = No, 1 = Yes)", "hair growth(Y/N)", "number"),
            new Feature("skinDarkening", "Skin darkening (0 = No, 1 = Yes)", "Skin darkening (Y/N)", "number"),
            new Feature("hairLoss", "Hair loss (0 = No, 1 = Yes)", "Hair loss(Y/N)", "number"),
            new Feature("pimples", "Pimples (0 = No, 1 = Yes)", "Pimples(Y/N)", "number"),
            new Feature("fastFood", "Fast food (0 = No, 1 = Yes)", "Fast food (Y/N)", "number"),
            new Feature("regularExercise", "Regular exercise (0 = No, 1 = Yes)", "Reg.Exercise(Y/N)", "number"),
            new Feature("bpSystolic", "BP systolic (mmHg)", "BP _Systolic (mmHg)", "number"),
            new Feature("bpDiastolic", "BP diastolic (mmHg)", "BP _Diastolic (mmHg)", "number"),
            new Feature("follicleL", "Follicle count (left)", "Follicle No. (L)", "number"),
            new Feature("follicleR", "Follicle count (right)", "Follicle No. (R)", "number"),
            new Feature("avgFollicleSizeL", "Avg. follicle size (left, mm)", "Avg. F size (L) (mm)", "number"),
            new Feature("avgFollicleSizeR", "Avg. follicle size (right, mm)", "Avg. F size (R) (mm)", "number"),
            new Feature("endometrium", "Endometrium (mm)", "Endometrium (mm)", "number")
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
