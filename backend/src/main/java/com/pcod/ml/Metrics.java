package com.pcod.ml;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Test-set metrics for one model (positive class = 1, PCOS diagnosed).
 *
 * <p>Every entry carries the full honest picture, not just accuracy:
 * confusion-matrix counts, precision/recall/F1, ROC-AUC and PR-AUC,
 * Brier score (calibration), the ROC curve points, and a 95% bootstrap
 * confidence interval for accuracy — because with a 108-row test set, a
 * 2–3 patient difference (86% vs 89%) is not a significant difference.</p>
 */
public class Metrics {

    /** One confusion-matrix cell set (test set, at the model's decision threshold). */
    public static class ConfusionMatrix {
        public int tp;
        public int fp;
        public int fn;
        public int tn;

        public ConfusionMatrix() {}

        public ConfusionMatrix(int tp, int fp, int fn, int tn) {
            this.tp = tp;
            this.fp = fp;
            this.fn = fn;
            this.tn = tn;
        }
    }

    /** One ROC-curve point. */
    public static class RocPoint {
        public double fpr;
        public double tpr;

        public RocPoint() {}

        public RocPoint(double fpr, double tpr) {
            this.fpr = fpr;
            this.tpr = tpr;
        }
    }

    public static class ModelMetrics {
        public String name;
        public double accuracy;
        public double precision;
        public double recall;
        public double f1;
        /** 5-fold CV accuracy on the TRAIN split (the tuning criterion). */
        public double cvAccuracy;
        /** Hyperparameters + feature count + clipping of the tuned config. */
        public String hyperparameters;
        public int nFeatures;
        public boolean clipApplied;
        public boolean oversample;
        /** Feature keys this model was trained on (in training order). */
        public List<String> featureSubset = new ArrayList<>();
        /** "full" (all 41 features) or "screening" (23 non-invasive features). */
        public String mode = "full";
        /** True for the screening model (same signal as mode = "screening"). */
        public boolean screening = false;
        /** Decision threshold these test metrics were computed at (0.5 unless tuned). */
        public double threshold = 0.5;
        /** Confusion matrix on the test set, at {@link #threshold}. */
        public ConfusionMatrix confusionMatrix;
        /** Area under the ROC curve on the test set (threshold-independent). */
        public double rocAuc;
        /** Area under the precision-recall curve on the test set. */
        public double prAuc;
        /** Brier score on the test set: mean (p - y)^2; lower is better calibrated. */
        public double brierScore;
        /** ROC curve points on the test set, as {fpr, tpr}. */
        public List<RocPoint> rocCurve = new ArrayList<>();
        /** 95% bootstrap confidence interval for test accuracy (1000 resamples, seed 42). */
        public double accuracyCiLow;
        public double accuracyCiHigh;

        public ModelMetrics() {}

        public ModelMetrics(String name, double accuracy, double precision, double recall, double f1) {
            this.name = name;
            this.accuracy = accuracy;
            this.precision = precision;
            this.recall = recall;
            this.f1 = f1;
        }
    }

    /**
     * Decision-threshold report for one deployed model. The threshold is
     * chosen on TRAIN cross-validation out-of-fold probabilities ONLY,
     * never on the test set. The deployed (full) model maximises F1 (ties
     * broken toward the lower threshold / higher recall); the screening
     * model instead takes the highest threshold that still reaches at
     * least 90% recall on the out-of-fold probabilities — for a screening
     * tool a missed case costs more than a false alarm, so recall is
     * constrained first and precision is maximised under that constraint.
     * The test numbers below are then measured once, at that fixed
     * threshold, alongside the same test numbers at the default 0.5 so the
     * trade-off is visible.
     */
    public static class ThresholdReport {
        public String model;
        public double threshold;
        public String chosenOn = "train 5-fold CV out-of-fold probabilities, maximising F1 "
                + "(ties broken toward higher recall); never tuned on the test set";
        public double oofPrecision;
        public double oofRecall;
        public double oofF1;
        public double testAccuracy;
        public double testPrecision;
        public double testRecall;
        public double testF1;
        public double defaultThresholdTestAccuracy;
        public double defaultThresholdTestPrecision;
        public double defaultThresholdTestRecall;
        public double defaultThresholdTestF1;

        public ThresholdReport() {}
    }

    public List<ModelMetrics> models = new ArrayList<>();
    /**
     * Baselines, trained and evaluated under the same protocol as the
     * ensembles (same split, same train-only CV tuning, one test
     * evaluation): majority-class, Logistic Regression, single Decision
     * Tree, linear SVM, KNN. They are what "89% accuracy" has to beat to
     * mean anything.
     */
    public List<ModelMetrics> baselines = new ArrayList<>();
    /**
     * Model version of this training run (e.g. "1.2.0"). Recorded on every
     * prediction row (PredictionRecord.modelVersion) so a retrain is
     * auditable: any stored prediction can be traced to the model that
     * made it. Bumped when the pipeline changes (imputation rule,
     * threshold rule, library versions), not on every retrain.
     */
    public String modelVersion;
    /** One model in the repeated stratified CV (10 repeats x 5 folds, all 541 rows). */
    public static class RepeatedCvEntry {
        public String model;
        public int repeats = 10;
        public int folds = 5;
        public double accuracyMean;
        public double accuracySd;
        public double precisionMean;
        public double precisionSd;
        public double recallMean;
        public double recallSd;
        public double f1Mean;
        public double f1Sd;
        public double rocAucMean;
        public double rocAucSd;

        public RepeatedCvEntry() {}
    }

    /**
     * Repeated stratified cross-validation (10 repeats x 5 folds) on ALL
     * 541 rows, with the tuned hyperparameters held fixed (no re-tuning
     * per repeat): the mean +/- SD of each metric across the 10 repeats.
     * With a 108-row test set the held-out CI is 12 points wide; this is
     * the tighter, more stable estimate. The held-out 80/20 split is kept
     * as a secondary check (see the per-model test metrics).
     */
    public List<RepeatedCvEntry> repeatedCv = new ArrayList<>();

    /** One McNemar paired comparison on the held-out test set. */
    public static class McNemarEntry {
        public String comparison;
        /** Voting correct, other wrong. */
        public int b;
        /** Voting wrong, other correct. */
        public int c;
        /** Chi-square with Edwards continuity correction (1 df). */
        public double chiSquare;
        /** Exact two-sided binomial p-value on the discordant pairs. */
        public double pValue;
        public String method = "McNemar exact test (two-sided binomial on discordant pairs)";

        public McNemarEntry() {}
    }

    /** McNemar paired tests: voting ensemble vs each baseline / best single model. */
    public List<McNemarEntry> mcnemar = new ArrayList<>();

    /** One feature-group ablation variant (same split, fixed tuned hyperparameters). */
    public static class AblationEntry {
        public String variant;
        public int nFeatures;
        /** Voting-ensemble test accuracy / recall at threshold 0.5. */
        public double votingAccuracy;
        public double votingRecall;
        /** Gradient Boosting test accuracy / recall at threshold 0.5. */
        public double gbmAccuracy;
        public double gbmRecall;

        public AblationEntry() {}
    }

    /**
     * Feature-group ablation: the same protocol on feature subsets, so the
     * "follicle counts are near-diagnostic" claim becomes a measured
     * result instead of an impression.
     */
    public List<AblationEntry> ablation = new ArrayList<>();

    /** One reliability-curve bin: mean predicted probability vs observed frequency. */
    public static class CalibrationBin {
        public double binLow;
        public double binHigh;
        public int count;
        public double meanPredicted;
        public double observedFrequency;

        public CalibrationBin() {}
    }

    /** Reliability-curve data for one model (test set and out-of-fold). */
    public static class CalibrationReport {
        public String model;
        public List<CalibrationBin> testBins = new ArrayList<>();
        public List<CalibrationBin> oofBins = new ArrayList<>();

        public CalibrationReport() {}
    }

    /**
     * Calibration (reliability curve) for the voting ensemble: binned
     * predicted probabilities against the observed PCOS frequency. A Brier
     * score alone does not show calibration — this does. The plot is built
     * from these bins.
     */
    public CalibrationReport calibration;
    /** Deployed model: best TRAIN-CV single model, or "Soft Voting Ensemble". */
    public String bestModel;
    /** Test accuracy of the deployed model, at its tuned decision threshold. */
    public double bestModelTestAccuracy;
    public int trainSize;
    public int testSize;
    public int seed = 42;
    public int cvFolds = 5;
    public String generatedAt;
    /** How tuning was done (for the report / viva). */
    public String tuningMethod;
    /** Random-Forest feature importance on the TRAIN split only (key -> score). */
    public Map<String, Double> featureImportance = new LinkedHashMap<>();
    /** Gradient-Boosting feature importance on the TRAIN split only (key -> score). */
    public Map<String, Double> gbmImportance = new LinkedHashMap<>();
    /** Screening-model (Gradient Boosting on 23 non-invasive features) importance. */
    public Map<String, Double> screeningImportance = new LinkedHashMap<>();
    /** Full CV log per model: one line per candidate config tried. */
    public Map<String, List<String>> cvLog = new LinkedHashMap<>();
    /** Per-model fitted preprocessing (train-only), so inference can reproduce each model's exact transform. */
    public Map<String, Preprocessing> modelPreprocessing = new LinkedHashMap<>();
    public double votingCvAccuracy;
    public double votingTestAccuracy;
    public boolean votingUsed;
    /** Soft-voting ensemble test metrics (same single test evaluation as the rest). */
    public ConfusionMatrix votingConfusionMatrix;
    public double votingRocAuc;
    public double votingPrAuc;
    public double votingBrierScore;
    public List<RocPoint> votingRocCurve = new ArrayList<>();
    public double votingAccuracyCiLow;
    public double votingAccuracyCiHigh;
    /** Threshold report for the deployed (full) model, and for the screening model. */
    public ThresholdReport deployedThreshold;
    public ThresholdReport screeningThreshold;
    /**
     * Number of doctor-verified prediction-history rows that were appended
     * to the TRAIN split for this training run (0 for a plain CSV run).
     * Only rows with a filled verified outcome are ever used for training —
     * unverified model predictions are never trained on.
     */
    public int verifiedRowsUsed;

    public ModelMetrics best() {
        return models.stream().filter(m -> m.name.equals(bestModel)).findFirst()
                .orElseThrow(() -> new IllegalStateException("Best model not in metrics: " + bestModel));
    }

    public ModelMetrics byName(String name) {
        return models.stream().filter(m -> m.name.equals(name)).findFirst().orElse(null);
    }
}
