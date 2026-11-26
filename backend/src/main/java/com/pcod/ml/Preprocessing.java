package com.pcod.ml;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.List;

/**
 * Fitted preprocessing parameters, persisted as {@code preprocessing.json}
 * and loaded by the Spring Boot app at startup so inference uses exactly the
 * same transformation as training.
 *
 * <p>Pipeline (project methodology, fitted on the TRAIN split only — no
 * leakage from the test split):</p>
 * <ol>
 *   <li><b>Median imputation</b> — a missing value is replaced by the training
 *       median of that feature. The median, not the mean: a few extreme
 *       outliers (FSH max 5052, Vitamin D3 max 6014.66) inflate the raw mean
 *       far above the typical value, so a blank FSH filled with the mean
 *       would sit +3.9 SD above the clipped mean and push the prediction
 *       toward PCOS. Only 2 cells in the whole dataset are genuinely
 *       missing (Marriage duration: 1, Fast food: 1); the 539 empty cells
 *       belong to the dropped junk column {@code Unnamed: 44}.</li>
 *   <li><b>Outlier clipping (winsorising)</b> — each feature is clipped to its
 *       1st / 99th training percentile. Needed because FSH (max 5052),
 *       Vitamin D3 (max 6014.66) and beta-HCG (max 32460.97) contain extreme
 *       outliers that would otherwise dominate standardisation.</li>
 *   <li><b>Standardisation (StandardScaler-style z-score)</b> —
 *       {@code z = (x - mean) / std}, with mean/std computed on the
 *       imputed + clipped training data. This is the same arithmetic as
 *       Smile's {@code smile.feature.transform.Standardizer} (which also
 *       computes per-column mean and standard deviation); it is implemented
 *       explicitly here because the Smile transform object is not
 *       serialisable to a single-row predict call, whereas the fitted
 *       mean/std must be stored and re-applied verbatim at inference time.</li>
 * </ol>
 */
public class Preprocessing {

    /** One fitted feature: raw-scale stats + standardisation parameters. */
    public static class FeatureStats {
        public String key;
        public String label;
        public String type;
        /** Observed min / max on the full dataset (raw scale, before clipping) — descriptive statistics only; the ranges /api/features returns (and /api/predict enforces) come from FeatureSpec. */
        public double min;
        public double max;
        /** Training median on the raw scale — also the /api/features default and the median-imputation value. */
        @JsonProperty("default")
        public double defaultValue;
        /** Mean / std of the imputed + clipped TRAIN data (standardisation parameters). */
        public double mean;
        public double std;
        /** 1st / 99th training percentiles (clipping bounds). */
        public double clipLow;
        public double clipHigh;

        public FeatureStats() {}

        public FeatureStats(String key, String label, String type) {
            this.key = key;
            this.label = label;
            this.type = type;
        }
    }

    public List<FeatureStats> features = new ArrayList<>();
    public int trainSize;
    public int testSize;
    public int seed = 42;
    public int class0Count;
    public int class1Count;
    /** Feature keys the deployed model uses (subset chosen by train CV; all 41 if no selection won). */
    public List<String> selectedFeatures = new ArrayList<>();

    @JsonIgnore
    public FeatureStats stats(String key) {
        return features.stream().filter(f -> f.key.equals(key)).findFirst()
                .orElseThrow(() -> new IllegalArgumentException("Unknown feature: " + key));
    }

    /** Impute (training median) -> clip -> standardise one raw feature value. */
    public double transform(int featureIndex, double rawValue) {
        FeatureStats s = features.get(featureIndex);
        double v = Double.isNaN(rawValue) ? s.defaultValue : rawValue;
        v = Math.min(Math.max(v, s.clipLow), s.clipHigh);
        double std = s.std == 0.0 ? 1.0 : s.std;
        return (v - s.mean) / std;
    }

    /** Standardise a full 41-value raw feature vector (NaN = missing). */
    public double[] transform(double[] raw) {
        double[] out = new double[FeatureSpec.N_FEATURES];
        for (int j = 0; j < FeatureSpec.N_FEATURES; j++) {
            out[j] = transform(j, raw[j]);
        }
        return out;
    }
}
