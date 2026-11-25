package com.pcod.ml;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import smile.base.cart.SplitRule;
import smile.classification.AdaBoost;
import smile.classification.Classifier;
import smile.classification.DecisionTree;
import smile.classification.GradientTreeBoost;
import smile.classification.KNN;
import smile.classification.LogisticRegression;
import smile.classification.RandomForest;
import smile.classification.SVM;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.vector.BaseVector;
import smile.data.vector.DoubleVector;
import smile.data.vector.IntVector;
import smile.math.MathEx;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.LongStream;

/**
 * Shared training pipeline, used by the {@code ModelTrainer} main class and
 * by POST /api/retrain. Trains all four ensemble models on the real dataset:
 * Random Forest, Bagging (see {@link BaggingClassifier}), AdaBoost and
 * Gradient Boosting.
 *
 * <h2>Honest evaluation protocol (test set never touched before final evaluation)</h2>
 * <ol>
 *   <li>Stratified 80/20 train/test split, seed 42. The test split is touched
 *       exactly ONCE, at the very end, to measure the final models.</li>
 *   <li>All tuning uses 5-fold stratified cross-validation <b>on the train
 *       split only</b>: hyperparameter candidates, feature-subset candidates
 *       and the clipping on/off choice are all compared by CV accuracy.</li>
 *   <li>Preprocessing (mean imputation, 1st/99th-percentile clipping,
 *       standardisation) is fitted on training data only — inside CV, it is
 *       re-fitted on each fold's training part, never on the validation fold.
 *       Feature ranking (Random Forest importance) is likewise re-computed
 *       inside every CV fold, on that fold's training part only, when
 *       feature subsets are compared — so subset selection never sees a
 *       validation fold. The full-train ranking is used only to build the
 *       final model's feature list once a subset size has won. (An earlier
 *       version ranked once on the whole train split and reused that ranking
 *       for every fold; that made the CV scores of the smaller subsets
 *       slightly optimistic, because the ranking had seen the validation
 *       rows. The test split was never involved in either version.)</li>
 *   <li>The deployed model is the one with the best <b>train CV</b> accuracy
 *       (single model or soft-voting ensemble, whichever wins on CV). Test
 *       metrics are reported for every model, honestly, whatever they are.
 *       No number is ever adjusted to hit a target.</li>
 *   <li>Baselines (majority class, Logistic Regression, single Decision
 *       Tree, linear SVM, KNN) and the screening model (Gradient Boosting
 *       on the 23 non-invasive features) are trained and evaluated under
 *       the same split and the same train-only CV protocol.</li>
 *   <li>Decision thresholds for the deployed model and the screening model
 *       are tuned on train cross-validation out-of-fold probabilities
 *       only, maximising F1 (ties toward higher recall); the test set is
 *       then measured once at the tuned threshold and once at 0.5.</li>
 *   <li>Retraining uses doctor-verified history rows only: prediction
 *       history rows are appended to the TRAIN split only when their
 *       verified outcome is filled in. Unverified model predictions are
 *       never trained on.</li>
 * </ol>
 */
public class TrainingService {

    public static final int SEED = 42;
    public static final double TRAIN_FRACTION = 0.8;
    public static final int CV_FOLDS = 5;

    /** Model display name -> serialised model file name under models/. */
    public static final Map<String, String> MODEL_FILES = Map.of(
            "Random Forest", "random-forest.ser",
            "Bagging", "bagging.ser",
            "AdaBoost", "adaboost.ser",
            "Gradient Boosting", "gradient-boosting.ser"
    );

    /** Screening model: Gradient Boosting on the 23 non-invasive features only. */
    public static final String SCREENING_MODEL = "Gradient Boosting (Screening)";
    public static final String SCREENING_FILE = "gradient-boosting-screening.ser";

    /** Baselines, trained and evaluated under the same protocol as the ensembles. */
    public static final List<String> BASELINE_NAMES = List.of(
            "Majority Class", "Logistic Regression", "Decision Tree", "SVM (Linear)", "KNN");

    /** One dataset row: 41 raw feature values (NaN = missing) + label. */
    public record Row(double[] x, int y) {}

    public record TrainingResult(Metrics metrics, Preprocessing preprocessing) {}

    /** Hyperparameters + feature subset for one model. */
    public record ModelConfig(String model, Map<String, Object> params, int[] features, boolean clip,
                              boolean oversample) {
        public ModelConfig(String model, Map<String, Object> params, int[] features, boolean clip) {
            this(model, params, features, clip, false);
        }

        public String describe() {
            // Sort keys: Map.of() iteration order is unspecified and varies per
            // JVM run, so without sorting metrics.json would differ cosmetically
            // between identical runs even though every number is the same.
            List<String> parts = new ArrayList<>();
            params.entrySet().stream()
                    .sorted(Map.Entry.comparingByKey())
                    .forEach(e -> parts.add(e.getKey() + "=" + e.getValue()));
            return model + " [" + String.join(", ", parts)
                    + ", features=" + features.length + ", clip=" + clip
                    + ", oversample=" + oversample + "]";
        }
    }

    /** Outcome of the CV tuning for one model. */
    public record TunedModel(ModelConfig config, double cvAccuracy) {}

    private final ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    // ------------------------------------------------------------------ CSV

    /** Loads the dataset. Drops Sl. No / Patient File No. / Unnamed: 44, keeps the 41 features + target. */
    public List<Row> loadCsv(Path csvPath) throws IOException {
        malformedCellsThisLoad = 0;
        List<Row> rows = new ArrayList<>();
        try (BufferedReader br = Files.newBufferedReader(csvPath)) {
            String headerLine = br.readLine();
            if (headerLine == null) {
                throw new IOException("Empty CSV: " + csvPath);
            }
            List<String> rawHeaders = splitCsvLine(headerLine);
            Map<String, Integer> col = new LinkedHashMap<>();
            for (int i = 0; i < rawHeaders.size(); i++) {
                col.put(rawHeaders.get(i).trim(), i);
            }
            int[] featureCols = new int[FeatureSpec.N_FEATURES];
            for (int j = 0; j < FeatureSpec.N_FEATURES; j++) {
                Integer idx = col.get(FeatureSpec.FEATURES.get(j).csvHeader());
                if (idx == null) {
                    throw new IOException("CSV header not found for feature "
                            + FeatureSpec.FEATURES.get(j).key() + " (expected header '"
                            + FeatureSpec.FEATURES.get(j).csvHeader() + "')");
                }
                featureCols[j] = idx;
            }
            Integer targetCol = col.get("PCOS (Y/N)");
            if (targetCol == null) {
                throw new IOException("CSV header not found: 'PCOS (Y/N)'");
            }
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                List<String> cells = splitCsvLine(line);
                double[] x = new double[FeatureSpec.N_FEATURES];
                for (int j = 0; j < FeatureSpec.N_FEATURES; j++) {
                    String v = featureCols[j] < cells.size() ? cells.get(featureCols[j]).trim() : "";
                    x[j] = parseCell(v, featureCols[j]);
                }
                int y = Integer.parseInt(cells.get(targetCol).trim());
                rows.add(new Row(x, y));
            }
        }
        return rows;
    }

    /**
     * Splits one CSV line into fields, honouring RFC 4180 double-quoted
     * fields: a comma inside quotes does not split the field, and a
     * doubled quote ("") inside a quoted field is one literal quote.
     * A quote at the start of a field opens a quoted field; a quote in the
     * middle of a field is kept literally (lenient, like most parsers).
     * Written inline — no new CSV dependency. (The old code used
     * {@code String.split(",")}, which broke on quoted commas.)
     */
    public static List<String> splitCsvLine(String line) {
        List<String> fields = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (inQuotes) {
                if (c == '"') {
                    if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
                        cur.append('"');
                        i++;
                    } else {
                        inQuotes = false;
                    }
                } else {
                    cur.append(c);
                }
            } else if (c == '"') {
                if (cur.length() == 0) {
                    inQuotes = true;
                } else {
                    cur.append(c);
                }
            } else if (c == ',') {
                fields.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(c);
            }
        }
        fields.add(cur.toString());
        return fields;
    }

    /**
     * Parses one numeric cell. The source file has two malformed numeric cells
     * (found 2026-11-25, in addition to the 2 genuinely missing cells in
     * SCHEMA.md): beta-HCG II = "1.99." (trailing dot) and AMH = "a" (stray
     * letter). A trailing dot is stripped ("1.99." -&gt; 1.99); any other
     * unparseable value is treated as missing (NaN) and mean-imputed, and
     * counted in the log — never silently dropped, never invented.
     *
     * <p>The malformed-cell count is per {@link #loadCsv(Path)} call
     * ({@link #malformedCellsThisLoad}, reset at the start of each load) —
     * never a static accumulator across retrains.</p>
     */
    private int malformedCellsThisLoad = 0;

    /** Malformed numeric cells found by the most recent {@link #loadCsv(Path)} call. */
    public int malformedCells() {
        return malformedCellsThisLoad;
    }

    private double parseCell(String v, int col) {
        if (v == null || v.isEmpty()) {
            return Double.NaN;
        }
        try {
            return Double.parseDouble(v);
        } catch (NumberFormatException e) {
            String cleaned = v.replaceAll("\\.+$", "");
            try {
                double d = Double.parseDouble(cleaned);
                System.out.printf("Cleaned malformed cell (col %d): %s -> %s%n", col, v, cleaned);
                return d;
            } catch (NumberFormatException e2) {
                malformedCellsThisLoad++;
                System.out.printf("Malformed numeric cell treated as missing (col %d): '%s'%n", col, v);
                return Double.NaN;
            }
        }
    }

    // ------------------------------------------------------- split + scaling

    /** Stratified 80/20 split with seed 42: shuffles within each class, then takes 80% per class. */
    public static List<List<Row>> stratifiedSplit(List<Row> rows, double trainFraction, long seed) {
        List<Row> class0 = new ArrayList<>();
        List<Row> class1 = new ArrayList<>();
        for (Row r : rows) {
            (r.y() == 1 ? class1 : class0).add(r);
        }
        Random rng = new Random(seed);
        Collections.shuffle(class0, rng);
        Collections.shuffle(class1, rng);
        int n0Train = (int) Math.round(class0.size() * trainFraction);
        int n1Train = (int) Math.round(class1.size() * trainFraction);
        List<Row> train = new ArrayList<>();
        train.addAll(class0.subList(0, n0Train));
        train.addAll(class1.subList(0, n1Train));
        List<Row> test = new ArrayList<>();
        test.addAll(class0.subList(n0Train, class0.size()));
        test.addAll(class1.subList(n1Train, class1.size()));
        Collections.shuffle(train, new Random(seed + 1));
        Collections.shuffle(test, new Random(seed + 2));
        return List.of(train, test);
    }

    /** Stratified k-fold partition of the given rows (seed 42). Each fold keeps the ~2:1 class ratio. */
    public static List<List<Row>> stratifiedFolds(List<Row> rows, int k, long seed) {
        List<Row> class0 = new ArrayList<>();
        List<Row> class1 = new ArrayList<>();
        for (Row r : rows) {
            (r.y() == 1 ? class1 : class0).add(r);
        }
        Random rng = new Random(seed);
        Collections.shuffle(class0, rng);
        Collections.shuffle(class1, rng);
        List<List<Row>> folds = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            folds.add(new ArrayList<>());
        }
        for (int i = 0; i < class0.size(); i++) {
            folds.get(i % k).add(class0.get(i));
        }
        for (int i = 0; i < class1.size(); i++) {
            folds.get(i % k).add(class1.get(i));
        }
        return folds;
    }

    private static double percentile(double[] sorted, double p) {
        if (sorted.length == 0) {
            return Double.NaN;
        }
        double rank = p / 100.0 * (sorted.length - 1);
        int lo = (int) Math.floor(rank);
        int hi = (int) Math.ceil(rank);
        if (lo == hi) {
            return sorted[lo];
        }
        return sorted[lo] + (rank - lo) * (sorted[hi] - sorted[lo]);
    }

    /** Averages P(PCOS) across several models, row by row (soft voting). */
    public static ProbPredictions averageProbs(List<ProbPredictions> parts) {
        int n = parts.get(0).actual().length;
        int[] actual = parts.get(0).actual();
        double[] prob = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (ProbPredictions p : parts) {
                sum += p.prob()[i];
            }
            prob[i] = sum / parts.size();
        }
        return new ProbPredictions(actual, prob);
    }

    /**
     * Feature importance (mean decrease in impurity) from one tree model
     * fitted on TRAIN only, keyed by feature key (in the config's feature
     * order). Random Forest and Gradient Boosting both expose it in Smile.
     */
    private Map<String, Double> importanceMap(List<Row> train, ModelConfig cfg) {
        Preprocessing p = fitPreprocessing(train, train, cfg.clip());
        DataFrame df = toDataFrame(train, p, cfg.features());
        Object model = trainOne(cfg, df, SEED);
        double[] imp;
        if (model instanceof RandomForest rf) {
            imp = rf.importance();
        } else if (model instanceof GradientTreeBoost gtb) {
            imp = gtb.importance();
        } else {
            throw new IllegalArgumentException("No importance available for: " + cfg.model());
        }
        Map<String, Double> out = new LinkedHashMap<>();
        for (int k = 0; k < cfg.features().length; k++) {
            out.put(FeatureSpec.FEATURES.get(cfg.features()[k]).key(), imp[k]);
        }
        return out;
    }

    /**
     * Fits imputation means, clip bounds and standardisation params on
     * {@code train} only; min/max are computed on {@code all} (used only for
     * the /api/features display ranges). Pass {@code clip=false} to disable
     * outlier clipping (bounds become infinite).
     */
    public Preprocessing fitPreprocessing(List<Row> train, List<Row> all, boolean clip) {
        Preprocessing prep = new Preprocessing();
        prep.trainSize = train.size();
        prep.testSize = all.size() - train.size();
        prep.seed = SEED;
        prep.class0Count = (int) all.stream().filter(r -> r.y() == 0).count();
        prep.class1Count = (int) all.stream().filter(r -> r.y() == 1).count();

        for (int j = 0; j < FeatureSpec.N_FEATURES; j++) {
            FeatureSpec.Feature f = FeatureSpec.FEATURES.get(j);
            Preprocessing.FeatureStats s = new Preprocessing.FeatureStats(f.key(), f.label(), f.type());

            double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
            for (Row r : all) {
                double v = r.x()[j];
                if (!Double.isNaN(v)) {
                    min = Math.min(min, v);
                    max = Math.max(max, v);
                }
            }
            s.min = min;
            s.max = max;

            // Training mean (raw scale) — the mean-imputation value and the /api/features default.
            double sum = 0;
            int n = 0;
            for (Row r : train) {
                double v = r.x()[j];
                if (!Double.isNaN(v)) {
                    sum += v;
                    n++;
                }
            }
            double rawMean = n == 0 ? 0.0 : sum / n;
            s.defaultValue = rawMean;

            // Clip bounds: 1st / 99th percentile of the (imputed) training values.
            double[] trainVals = new double[train.size()];
            for (int i = 0; i < train.size(); i++) {
                double v = train.get(i).x()[j];
                trainVals[i] = Double.isNaN(v) ? rawMean : v;
            }
            double[] sorted = trainVals.clone();
            Arrays.sort(sorted);
            s.clipLow = clip ? percentile(sorted, 1.0) : Double.NEGATIVE_INFINITY;
            s.clipHigh = clip ? percentile(sorted, 99.0) : Double.POSITIVE_INFINITY;

            // Standardisation mean/std on imputed + clipped training values.
            double[] clipped = new double[trainVals.length];
            double cSum = 0;
            for (int i = 0; i < trainVals.length; i++) {
                clipped[i] = Math.min(Math.max(trainVals[i], s.clipLow), s.clipHigh);
                cSum += clipped[i];
            }
            double mean = cSum / clipped.length;
            double var = 0;
            for (double v : clipped) {
                var += (v - mean) * (v - mean);
            }
            s.mean = mean;
            s.std = Math.sqrt(var / clipped.length);
            if (s.std == 0.0) {
                s.std = 1.0;
            }
            prep.features.add(s);
        }
        return prep;
    }

    /** All 41 feature indices, in FeatureSpec order. */
    public static int[] allFeatures() {
        int[] idx = new int[FeatureSpec.N_FEATURES];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        return idx;
    }

    /**
     * Builds a Smile DataFrame from a row matrix whose LAST column is the
     * target (0/1). Features become {@link DoubleVector}s and the target an
     * {@link IntVector}: Smile's {@code ClassLabels.fit} calls
     * {@code toIntArray()} on the response vector, which {@code DoubleVector}
     * does NOT support (it throws UnsupportedOperationException), so a
     * double-valued target column crashes training. This is the one place
     * DataFrames are built from matrices — all callers go through here.
     */
    public static DataFrame buildDataFrame(double[][] data, String[] names) {
        BaseVector[] vectors = new BaseVector[names.length];
        for (int j = 0; j < names.length - 1; j++) {
            double[] col = new double[data.length];
            for (int i = 0; i < data.length; i++) {
                col[i] = data[i][j];
            }
            vectors[j] = DoubleVector.of(names[j], col);
        }
        int[] y = new int[data.length];
        for (int i = 0; i < data.length; i++) {
            y[i] = (int) Math.round(data[i][names.length - 1]);
        }
        vectors[names.length - 1] = IntVector.of(names[names.length - 1], y);
        return DataFrame.of(vectors);
    }

    /** Standardised DataFrame (selected features + target) for Smile. */
    public DataFrame toDataFrame(List<Row> rows, Preprocessing prep, int[] featureIdx) {
        String[] names = new String[featureIdx.length + 1];
        for (int k = 0; k < featureIdx.length; k++) {
            names[k] = FeatureSpec.FEATURES.get(featureIdx[k]).key();
        }
        names[featureIdx.length] = FeatureSpec.TARGET;
        double[][] data = new double[rows.size()][featureIdx.length + 1];
        for (int i = 0; i < rows.size(); i++) {
            double[] z = prep.transform(rows.get(i).x());
            for (int k = 0; k < featureIdx.length; k++) {
                data[i][k] = z[featureIdx[k]];
            }
            data[i][featureIdx.length] = rows.get(i).y();
        }
        return buildDataFrame(data, names);
    }

    // ------------------------------------------------------------- evaluation

    @FunctionalInterface
    public interface Predictor {
        int predict(Tuple x, double[] posteriori);
    }

    /** Accuracy / precision / recall / F1 of a predictor on a DataFrame (positive class = 1). */
    public Metrics.ModelMetrics evaluate(String name, Predictor predictor, DataFrame test) {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        int targetCol = test.ncol() - 1;
        for (int i = 0; i < test.size(); i++) {
            Tuple row = test.get(i);
            int actual = (int) Math.round(row.getDouble(targetCol));
            double[] post = new double[2];
            int pred = predictor.predict(row, post);
            // Use the probability consistently: label = P(class 1) >= 0.5.
            if (post[0] + post[1] > 0) {
                pred = post[1] >= 0.5 ? 1 : 0;
            }
            if (pred == 1 && actual == 1) tp++;
            else if (pred == 0 && actual == 0) tn++;
            else if (pred == 1 && actual == 0) fp++;
            else fn++;
        }
        int total = tp + tn + fp + fn;
        double accuracy = (double) (tp + tn) / total;
        double precision = (tp + fp) == 0 ? 0.0 : (double) tp / (tp + fp);
        double recall = (tp + fn) == 0 ? 0.0 : (double) tp / (tp + fn);
        double f1 = (precision + recall) == 0 ? 0.0 : 2 * precision * recall / (precision + recall);
        return new Metrics.ModelMetrics(name, accuracy, precision, recall, f1);
    }

    private static Predictor smilePredictor(Classifier<Tuple> model) {
        return (x, post) -> model.predict(x, post);
    }

    public static Predictor predictorFor(Object model) {
        if (model instanceof BaggingClassifier bagging) {
            return bagging::predict;
        }
        if (model instanceof Classifier<?> classifier) {
            @SuppressWarnings("unchecked")
            Classifier<Tuple> clf = (Classifier<Tuple>) classifier;
            return smilePredictor(clf);
        }
        throw new IllegalArgumentException("Unknown model type: " + model.getClass());
    }

    // ------------------------------------------------------------ model fits

    /**
     * Trains one model from a config on the given (already standardised)
     * training DataFrame. All randomness is seeded from SEED so training is
     * reproducible.
     *
     * <p>Determinism note: Smile's RNG is thread-local
     * ({@code MathEx.setSeed} seeds only the calling thread), and Random
     * Forest builds its trees on a parallel stream — the calling thread
     * sometimes runs a tree task itself, which re-seeds its RNG to that
     * tree's seed. A single {@code setSeed} at the start of training is
     * therefore NOT enough: after a Random Forest fit, the calling
     * thread's RNG state depends on work-stealing, and the next
     * Gradient Boosting / AdaBoost / Bagging fit (single-threaded, same
     * thread) would start from a different RNG state on every run. So
     * every fit is re-seeded here, immediately before training, which
     * makes repeated runs bit-identical (verified: two consecutive full
     * trainer runs produce identical metrics.json apart from the
     * timestamp).</p>
     */
    public Object trainOne(ModelConfig config, DataFrame trainDf, long seed) {
        MathEx.setSeed(seed);
        Formula formula = Formula.lhs(FeatureSpec.TARGET);
        Map<String, Object> p = config.params();
        // The screening model IS Gradient Boosting — only its feature subset differs.
        String kind = SCREENING_MODEL.equals(config.model()) ? "Gradient Boosting" : config.model();
        return switch (kind) {
            case "Random Forest" -> {
                int ntrees = (int) p.getOrDefault("ntrees", 500);
                Random seedRng = new Random(seed);
                LongStream rfSeeds = LongStream.generate(() -> {
                    long s;
                    do {
                        s = seedRng.nextLong();
                    } while (s <= 1);
                    return s;
                }).limit(ntrees);
                yield RandomForest.fit(formula, trainDf, ntrees,
                        (int) p.getOrDefault("mtry", 0), SplitRule.GINI,
                        (int) p.getOrDefault("maxDepth", 20),
                        Math.max(10, trainDf.size() / 5),
                        (int) p.getOrDefault("nodeSize", 5),
                        (double) p.getOrDefault("subsample", 1.0), null, rfSeeds);
            }
            case "Bagging" -> {
                int nTrees = (int) p.getOrDefault("ntrees", 200);
                DecisionTree[] trees = new DecisionTree[nTrees];
                Random bagRng = new Random(seed);
                double[][] matrix = dataframeToMatrix(trainDf);
                for (int t = 0; t < nTrees; t++) {
                    double[][] boot = new double[trainDf.size()][trainDf.ncol()];
                    for (int i = 0; i < trainDf.size(); i++) {
                        boot[i] = matrix[bagRng.nextInt(trainDf.size())];
                    }
                    DataFrame bootDf = buildDataFrame(boot, columnNames(trainDf));
                    trees[t] = DecisionTree.fit(formula, bootDf, SplitRule.GINI,
                            (int) p.getOrDefault("maxDepth", 20),
                            Math.max(10, trainDf.size() / 5),
                            (int) p.getOrDefault("nodeSize", 5));
                }
                yield new BaggingClassifier(trees);
            }
            case "AdaBoost" -> AdaBoost.fit(formula, trainDf,
                    (int) p.getOrDefault("ntrees", 300),
                    (int) p.getOrDefault("maxDepth", 2),
                    (int) p.getOrDefault("maxNodes", 4),
                    (int) p.getOrDefault("nodeSize", 1));
            case "Gradient Boosting" -> GradientTreeBoost.fit(formula, trainDf,
                    (int) p.getOrDefault("ntrees", 500),
                    (int) p.getOrDefault("maxDepth", 3),
                    (int) p.getOrDefault("maxNodes", 8),
                    (int) p.getOrDefault("nodeSize", 5),
                    (double) p.getOrDefault("shrinkage", 0.05),
                    (double) p.getOrDefault("subsample", 0.7));
            case "Decision Tree" -> DecisionTree.fit(formula, trainDf, SplitRule.GINI,
                    (int) p.getOrDefault("maxDepth", 5),
                    (int) p.getOrDefault("maxNodes", 31),
                    (int) p.getOrDefault("nodeSize", 1));
            default -> throw new IllegalArgumentException("Unknown model: " + config.model());
        };
    }

    // ------------------------------------------- probability-based evaluation

    /** One test/train row's true label and a model's P(PCOS) for it. */
    public record ProbPredictions(int[] actual, double[] prob) {}

    /** Collects true labels + P(PCOS) for a Tuple-based predictor on a DataFrame. */
    public ProbPredictions collectProbs(Predictor predictor, DataFrame df) {
        int n = df.size();
        int targetCol = df.ncol() - 1;
        int[] actual = new int[n];
        double[] prob = new double[n];
        for (int i = 0; i < n; i++) {
            Tuple row = df.get(i);
            actual[i] = (int) Math.round(row.getDouble(targetCol));
            double[] post = new double[2];
            predictor.predict(row, post);
            prob[i] = post[0] + post[1] > 0 ? post[1] / (post[0] + post[1]) : 0.5;
        }
        return new ProbPredictions(actual, prob);
    }

    /** Standardised feature matrix + labels for the array-based baselines (LR / linear SVM / KNN). */
    public record Matrix(double[][] x, int[] y) {}

    public Matrix toMatrix(List<Row> rows, Preprocessing prep, int[] featureIdx) {
        double[][] x = new double[rows.size()][featureIdx.length];
        int[] y = new int[rows.size()];
        for (int i = 0; i < rows.size(); i++) {
            double[] z = prep.transform(rows.get(i).x());
            for (int k = 0; k < featureIdx.length; k++) {
                x[i][k] = z[featureIdx[k]];
            }
            y[i] = rows.get(i).y();
        }
        return new Matrix(x, y);
    }

    /**
     * Fits one array-based baseline. Logistic Regression (L2, lambda),
     * linear SVM (LASVM, C) and KNN (k neighbours) all take double[][]
     * input in Smile 3.1.1; their predict(x, posteriori) fills the
     * posterior from the fitted model (KNN: neighbour vote share).
     */
    public Classifier<double[]> fitBaseline(String kind, Map<String, Object> params, Matrix m, long seed) {
        MathEx.setSeed(seed);
        return switch (kind) {
            case "Logistic Regression" -> LogisticRegression.fit(m.x(), m.y(),
                    (double) params.getOrDefault("lambda", 0.1), 1e-4, 100);
            case "SVM (Linear)" -> {
                // Smile's linear SVM (LASVM) takes labels {-1, +1}, not {0, 1},
                // and its predict(x, posteriori) throws (hard classifier).
                // The wrapper maps +1/-1 to 1/0 and fills a one-hot posterior,
                // so the rest of the pipeline can treat it like any other
                // binary model. Its ROC-AUC/Brier therefore reflect hard
                // decisions, not calibrated probabilities (see the model card).
                int[] ypm = Arrays.stream(m.y()).map(y -> y == 1 ? 1 : -1).toArray();
                yield new HardBinaryClassifier(
                        SVM.fit(m.x(), ypm, (double) params.getOrDefault("C", 1.0), 1e-3));
            }
            case "KNN" -> KNN.fit(m.x(), m.y(), (int) params.getOrDefault("k", 5));
            default -> throw new IllegalArgumentException("Unknown baseline: " + kind);
        };
    }

    /**
     * Adapts a hard ±1 classifier (Smile's linear SVM) to the 0/1
     * probability interface the evaluation code expects: predict(x)
     * returns 1/0, and predict(x, posteriori) fills a one-hot posterior
     * [P(0), P(1)].
     */
    static final class HardBinaryClassifier implements Classifier<double[]> {
        private final Classifier<double[]> inner;

        HardBinaryClassifier(Classifier<double[]> inner) {
            this.inner = inner;
        }

        @Override
        public int numClasses() {
            return 2;
        }

        @Override
        public int[] classes() {
            return new int[]{0, 1};
        }

        @Override
        public int predict(double[] x) {
            return inner.predict(x) == 1 ? 1 : 0;
        }

        @Override
        public int predict(double[] x, double[] posteriori) {
            int label = predict(x);
            posteriori[0] = label == 1 ? 0.0 : 1.0;
            posteriori[1] = label == 1 ? 1.0 : 0.0;
            return label;
        }
    }

    /** 5-fold CV accuracy of one array-based baseline config, on TRAIN rows only. */
    public double cvAccuracyBaseline(String kind, Map<String, Object> params,
                                     List<Row> trainRows, int[] features) {
        List<List<Row>> folds = stratifiedFolds(trainRows, CV_FOLDS, SEED);
        double correct = 0;
        int total = 0;
        for (int f = 0; f < CV_FOLDS; f++) {
            List<Row> val = folds.get(f);
            List<Row> foldTrain = new ArrayList<>();
            for (int g = 0; g < CV_FOLDS; g++) {
                if (g != f) {
                    foldTrain.addAll(folds.get(g));
                }
            }
            Preprocessing prep = fitPreprocessing(foldTrain, foldTrain, true);
            Matrix trainM = toMatrix(foldTrain, prep, features);
            Matrix valM = toMatrix(val, prep, features);
            Classifier<double[]> model = fitBaseline(kind, params, trainM, SEED + f);
            for (int i = 0; i < valM.x().length; i++) {
                double[] post = new double[2];
                int pred = model.predict(valM.x()[i], post) == 1 ? 1 : 0;
                if (post[0] + post[1] > 0) {
                    pred = post[1] >= 0.5 ? 1 : 0;
                }
                if (pred == valM.y()[i]) {
                    correct++;
                }
                total++;
            }
        }
        return correct / total;
    }

    /**
     * Full test metrics from true labels + P(PCOS), at one decision
     * threshold: accuracy / precision / recall / F1, confusion matrix,
     * ROC-AUC, PR-AUC (average precision), Brier score, ROC curve points
     * and a 95% bootstrap CI for accuracy (1000 resamples, seed 42).
     */
    public Metrics.ModelMetrics metricsFromProbs(String name, int[] actual, double[] prob, double threshold) {
        int tp = 0, fp = 0, fn = 0, tn = 0;
        for (int i = 0; i < actual.length; i++) {
            int pred = prob[i] >= threshold ? 1 : 0;
            if (pred == 1 && actual[i] == 1) tp++;
            else if (pred == 0 && actual[i] == 0) tn++;
            else if (pred == 1 && actual[i] == 0) fp++;
            else fn++;
        }
        int total = tp + tn + fp + fn;
        double accuracy = (double) (tp + tn) / total;
        double precision = (tp + fp) == 0 ? 0.0 : (double) tp / (tp + fp);
        double recall = (tp + fn) == 0 ? 0.0 : (double) tp / (tp + fn);
        double f1 = (precision + recall) == 0 ? 0.0 : 2 * precision * recall / (precision + recall);
        Metrics.ModelMetrics m = new Metrics.ModelMetrics(name, accuracy, precision, recall, f1);
        m.threshold = threshold;
        m.confusionMatrix = new Metrics.ConfusionMatrix(tp, fp, fn, tn);
        m.rocAuc = rocAuc(actual, prob);
        m.prAuc = prAuc(actual, prob);
        m.brierScore = brierScore(actual, prob);
        m.rocCurve = rocCurve(actual, prob);
        double[] ci = bootstrapAccuracyCi(actual, prob, threshold);
        m.accuracyCiLow = ci[0];
        m.accuracyCiHigh = ci[1];
        return m;
    }

    /** ROC-AUC via the rank (Mann-Whitney) formula, with average ranks for ties. */
    public static double rocAuc(int[] actual, double[] prob) {
        int n = actual.length;
        int nPos = 0, nNeg = 0;
        for (int v : actual) {
            if (v == 1) nPos++; else nNeg++;
        }
        if (nPos == 0 || nNeg == 0) {
            return 0.5;
        }
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, Comparator.comparingDouble(i -> prob[i]));
        double[] rank = new double[n];
        int i = 0;
        while (i < n) {
            int j = i;
            while (j + 1 < n && prob[order[j + 1]] == prob[order[i]]) {
                j++;
            }
            double avgRank = (i + j) / 2.0 + 1.0; // 1-based average rank
            for (int k = i; k <= j; k++) {
                rank[order[k]] = avgRank;
            }
            i = j + 1;
        }
        double rankSumPos = 0;
        for (int k = 0; k < n; k++) {
            if (actual[k] == 1) {
                rankSumPos += rank[k];
            }
        }
        return (rankSumPos - (double) nPos * (nPos + 1) / 2) / ((double) nPos * nNeg);
    }

    /**
     * Average precision (area under the precision-recall curve), computed
     * by grouping on distinct probability values so tied scores (including
     * a constant predictor) are handled deterministically. For a constant
     * predictor this equals the positive-class prevalence.
     */
    public static double prAuc(int[] actual, double[] prob) {
        int n = actual.length;
        int nPos = 0;
        for (int v : actual) {
            if (v == 1) nPos++;
        }
        if (nPos == 0) {
            return 0.0;
        }
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(prob[b], prob[a]));
        double ap = 0;
        int tp = 0, fp = 0;
        int i = 0;
        while (i < n) {
            int j = i;
            while (j + 1 < n && prob[order[j + 1]] == prob[order[i]]) {
                j++;
            }
            for (int k = i; k <= j; k++) {
                if (actual[order[k]] == 1) tp++; else fp++;
            }
            double precision = (double) tp / (tp + fp);
            int tpBefore = tp;
            for (int k = i; k <= j; k++) {
                if (actual[order[k]] == 1) {
                    tpBefore--;
                }
            }
            int newPos = tp - tpBefore;
            ap += precision * newPos / nPos;
            i = j + 1;
        }
        return ap;
    }

    /** Brier score: mean (p - y)^2. Lower = better calibrated probabilities. */
    public static double brierScore(int[] actual, double[] prob) {
        double sum = 0;
        for (int i = 0; i < actual.length; i++) {
            double d = prob[i] - actual[i];
            sum += d * d;
        }
        return sum / actual.length;
    }

    /** ROC curve points, sweeping every distinct predicted probability as a threshold. */
    public static List<Metrics.RocPoint> rocCurve(int[] actual, double[] prob) {
        int n = actual.length;
        int nPos = 0, nNeg = 0;
        for (int v : actual) {
            if (v == 1) nPos++; else nNeg++;
        }
        List<Metrics.RocPoint> pts = new ArrayList<>();
        pts.add(new Metrics.RocPoint(0.0, 0.0));
        if (nPos == 0 || nNeg == 0) {
            pts.add(new Metrics.RocPoint(1.0, 1.0));
            return pts;
        }
        Double[] thresholds = Arrays.stream(prob).boxed().distinct()
                .sorted(Comparator.reverseOrder()).toArray(Double[]::new);
        for (double t : thresholds) {
            int tp = 0, fp = 0;
            for (int i = 0; i < n; i++) {
                if (prob[i] >= t) {
                    if (actual[i] == 1) tp++; else fp++;
                }
            }
            pts.add(new Metrics.RocPoint((double) fp / nNeg, (double) tp / nPos));
        }
        pts.add(new Metrics.RocPoint(1.0, 1.0));
        return pts;
    }

    /** 95% bootstrap CI for accuracy at one threshold (1000 resamples, seed 42). */
    public static double[] bootstrapAccuracyCi(int[] actual, double[] prob, double threshold) {
        int n = actual.length;
        int boot = 1000;
        double[] accs = new double[boot];
        Random rng = new Random(SEED);
        for (int b = 0; b < boot; b++) {
            int correct = 0;
            for (int i = 0; i < n; i++) {
                int idx = rng.nextInt(n);
                int pred = prob[idx] >= threshold ? 1 : 0;
                if (pred == actual[idx]) {
                    correct++;
                }
            }
            accs[b] = (double) correct / n;
        }
        Arrays.sort(accs);
        return new double[]{accs[(int) Math.round(0.025 * (boot - 1))], accs[(int) Math.round(0.975 * (boot - 1))]};
    }

    // --------------------------------------------------------- threshold tuning

    /**
     * Out-of-fold P(PCOS) for every TRAIN row, for one tuned config:
     * 5-fold stratified CV; each fold's model is trained on the other four
     * folds (preprocessing re-fitted inside the fold) and predicts its own
     * validation fold. These probabilities are TRAIN-only, so a threshold
     * chosen from them has never seen the test set.
     */
    public ProbPredictions oofProbsSingle(ModelConfig config, List<Row> trainRows) {
        List<List<Row>> folds = stratifiedFolds(trainRows, CV_FOLDS, SEED);
        int[] actual = new int[trainRows.size()];
        double[] prob = new double[trainRows.size()];
        Map<Row, Integer> pos = new java.util.IdentityHashMap<>();
        for (int i = 0; i < trainRows.size(); i++) {
            pos.put(trainRows.get(i), i);
        }
        for (int f = 0; f < CV_FOLDS; f++) {
            List<Row> val = folds.get(f);
            List<Row> foldTrain = new ArrayList<>();
            for (int g = 0; g < CV_FOLDS; g++) {
                if (g != f) {
                    foldTrain.addAll(folds.get(g));
                }
            }
            if (config.oversample()) {
                foldTrain = oversample(foldTrain, SEED + f);
            }
            Preprocessing prep = fitPreprocessing(foldTrain, foldTrain, config.clip());
            DataFrame trainDf = toDataFrame(foldTrain, prep, config.features());
            DataFrame valDf = toDataFrame(val, prep, config.features());
            Object model = trainOne(config, trainDf, SEED + f);
            Predictor predictor = predictorFor(model);
            ProbPredictions pp = collectProbs(predictor, valDf);
            for (int i = 0; i < val.size(); i++) {
                int idx = pos.get(val.get(i));
                actual[idx] = pp.actual()[i];
                prob[idx] = pp.prob()[i];
            }
        }
        return new ProbPredictions(actual, prob);
    }

    /** Out-of-fold P(PCOS) for the soft-voting ensemble (average of the four tuned models). */
    public ProbPredictions oofProbsVoting(List<TunedModel> tuned, List<Row> trainRows) {
        List<List<Row>> folds = stratifiedFolds(trainRows, CV_FOLDS, SEED);
        int[] actual = new int[trainRows.size()];
        double[] prob = new double[trainRows.size()];
        Map<Row, Integer> pos = new java.util.IdentityHashMap<>();
        for (int i = 0; i < trainRows.size(); i++) {
            pos.put(trainRows.get(i), i);
        }
        for (int f = 0; f < CV_FOLDS; f++) {
            List<Row> val = folds.get(f);
            List<Row> foldTrain = new ArrayList<>();
            for (int g = 0; g < CV_FOLDS; g++) {
                if (g != f) {
                    foldTrain.addAll(folds.get(g));
                }
            }
            List<Predictor> predictors = new ArrayList<>();
            List<DataFrame> valDfs = new ArrayList<>();
            for (TunedModel tm : tuned) {
                List<Row> modelTrain = tm.config().oversample()
                        ? oversample(foldTrain, SEED + f) : foldTrain;
                Preprocessing prep = fitPreprocessing(modelTrain, modelTrain, tm.config().clip());
                DataFrame trainDf = toDataFrame(modelTrain, prep, tm.config().features());
                DataFrame valDf = toDataFrame(val, prep, tm.config().features());
                Object model = trainOne(tm.config(), trainDf, SEED + f);
                predictors.add(predictorFor(model));
                valDfs.add(valDf);
            }
            for (int i = 0; i < val.size(); i++) {
                double pSum = 0;
                for (int m = 0; m < predictors.size(); m++) {
                    double[] post = new double[2];
                    predictors.get(m).predict(valDfs.get(m).get(i), post);
                    pSum += post[1];
                }
                int idx = pos.get(val.get(i));
                actual[idx] = val.get(i).y();
                prob[idx] = pSum / predictors.size();
            }
        }
        return new ProbPredictions(actual, prob);
    }

    /**
     * Fills a {@link Metrics.ThresholdReport}: tunes the threshold on the
     * out-of-fold probabilities, then measures the SAME model on the test
     * set once at the tuned threshold and once at the default 0.5.
     */
    private void fillThresholdReport(Metrics.ThresholdReport report,
                                     ProbPredictions oof, ProbPredictions testPp) {
        double[] tuned = tuneThreshold(oof);
        report.threshold = tuned[0];
        report.oofPrecision = tuned[1];
        report.oofRecall = tuned[2];
        report.oofF1 = tuned[3];
        Metrics.ModelMetrics atTuned = metricsFromProbs(report.model,
                testPp.actual(), testPp.prob(), tuned[0]);
        Metrics.ModelMetrics atDefault = metricsFromProbs(report.model,
                testPp.actual(), testPp.prob(), 0.5);
        report.testAccuracy = atTuned.accuracy;
        report.testPrecision = atTuned.precision;
        report.testRecall = atTuned.recall;
        report.testF1 = atTuned.f1;
        report.defaultThresholdTestAccuracy = atDefault.accuracy;
        report.defaultThresholdTestPrecision = atDefault.precision;
        report.defaultThresholdTestRecall = atDefault.recall;
        report.defaultThresholdTestF1 = atDefault.f1;
    }

    /** Fills the tuning bookkeeping (CV, hyperparameters, subset) of one model entry. */
    private void fillModelEntry(Metrics.ModelMetrics m, TunedModel tm) {
        ModelConfig cfg = tm.config();
        m.cvAccuracy = tm.cvAccuracy();
        m.hyperparameters = cfg.describe();
        m.nFeatures = cfg.features().length;
        m.clipApplied = cfg.clip();
        m.oversample = cfg.oversample();
        m.featureSubset = Arrays.stream(cfg.features())
                .mapToObj(j -> FeatureSpec.FEATURES.get(j).key()).toList();
    }

    /**
     * Chooses the decision threshold from out-of-fold probabilities:
     * scans 0.05–0.95, takes the threshold with the highest F1; ties are
     * broken toward the LOWER threshold (higher recall — for a screening
     * tool a missed case costs more than a false alarm). Returns the
     * threshold plus its out-of-fold precision/recall/F1.
     */
    public double[] tuneThreshold(ProbPredictions oof) {
        double bestT = 0.5;
        double bestF1 = -1;
        double bestP = 0, bestR = 0;
        for (int k = 1; k <= 19; k++) {
            double t = k * 0.05;
            int tp = 0, fp = 0, fn = 0;
            for (int i = 0; i < oof.actual().length; i++) {
                int pred = oof.prob()[i] >= t ? 1 : 0;
                if (pred == 1 && oof.actual()[i] == 1) tp++;
                else if (pred == 1) fp++;
                else if (oof.actual()[i] == 1) fn++;
            }
            double p = (tp + fp) == 0 ? 0.0 : (double) tp / (tp + fp);
            double r = (tp + fn) == 0 ? 0.0 : (double) tp / (tp + fn);
            double f1 = (p + r) == 0 ? 0.0 : 2 * p * r / (p + r);
            if (f1 > bestF1 + 1e-12 || (Math.abs(f1 - bestF1) <= 1e-12 && t < bestT)) {
                bestT = t;
                bestF1 = f1;
                bestP = p;
                bestR = r;
            }
        }
        return new double[]{bestT, bestP, bestR, bestF1};
    }

    private static String[] columnNames(DataFrame df) {
        String[] names = new String[df.ncol()];
        for (int j = 0; j < df.ncol(); j++) {
            names[j] = df.schema().field(j).name;
        }
        return names;
    }

    /**
     * Random oversampling of the minority class (duplicates drawn with
     * replacement, seeded): a simple, honest augmentation for the ~2:1 class
     * imbalance. Applied ONLY to training rows (never to validation/test).
     * Compared against no-oversampling by train CV; used only if it wins.
     */
    public static List<Row> oversample(List<Row> rows, long seed) {
        List<Row> minority = new ArrayList<>();
        List<Row> majority = new ArrayList<>();
        for (Row r : rows) {
            (r.y() == 1 ? minority : majority).add(r);
        }
        if (minority.isEmpty() || minority.size() >= majority.size()) {
            return new ArrayList<>(rows);
        }
        List<Row> out = new ArrayList<>(rows);
        Random rng = new Random(seed);
        while (out.stream().filter(r -> r.y() == 1).count() < majority.size()) {
            out.add(minority.get(rng.nextInt(minority.size())));
        }
        Collections.shuffle(out, new Random(seed + 1));
        return out;
    }

    /**
     * 5-fold CV accuracy of one config, on the TRAIN rows only. Preprocessing
     * is re-fitted inside every fold (train part only) — no leakage from the
     * validation fold into imputation / clipping / standardisation.
     */
    public double cvAccuracy(ModelConfig config, List<Row> trainRows) {
        List<List<Row>> folds = stratifiedFolds(trainRows, CV_FOLDS, SEED);
        double correct = 0;
        int total = 0;
        for (int f = 0; f < CV_FOLDS; f++) {
            List<Row> val = folds.get(f);
            List<Row> foldTrain = new ArrayList<>();
            for (int g = 0; g < CV_FOLDS; g++) {
                if (g != f) {
                    foldTrain.addAll(folds.get(g));
                }
            }
            if (config.oversample()) {
                foldTrain = oversample(foldTrain, SEED + f);
            }
            Preprocessing prep = fitPreprocessing(foldTrain, foldTrain, config.clip());
            DataFrame trainDf = toDataFrame(foldTrain, prep, config.features());
            DataFrame valDf = toDataFrame(val, prep, config.features());
            Object model = trainOne(config, trainDf, SEED + f);
            Metrics.ModelMetrics m = evaluate(config.model(), predictorFor(model), valDf);
            correct += m.accuracy * valDf.size();
            total += valDf.size();
        }
        return correct / total;
    }

    /** Soft-voting CV accuracy: average P(PCOS) of the four tuned models. */
    public double cvAccuracyVoting(List<TunedModel> tuned, List<Row> trainRows) {
        List<List<Row>> folds = stratifiedFolds(trainRows, CV_FOLDS, SEED);
        double correct = 0;
        int total = 0;
        for (int f = 0; f < CV_FOLDS; f++) {
            List<Row> val = folds.get(f);
            List<Row> foldTrain = new ArrayList<>();
            for (int g = 0; g < CV_FOLDS; g++) {
                if (g != f) {
                    foldTrain.addAll(folds.get(g));
                }
            }
            List<Predictor> predictors = new ArrayList<>();
            List<DataFrame> valDfs = new ArrayList<>();
            for (TunedModel tm : tuned) {
                List<Row> modelTrain = tm.config().oversample()
                        ? oversample(foldTrain, SEED + f) : foldTrain;
                Preprocessing prep = fitPreprocessing(modelTrain, modelTrain, tm.config().clip());
                DataFrame trainDf = toDataFrame(modelTrain, prep, tm.config().features());
                DataFrame valDf = toDataFrame(val, prep, tm.config().features());
                Object model = trainOne(tm.config(), trainDf, SEED + f);
                predictors.add(predictorFor(model));
                valDfs.add(valDf);
            }
            for (int i = 0; i < val.size(); i++) {
                double pSum = 0;
                for (int m = 0; m < predictors.size(); m++) {
                    double[] post = new double[2];
                    predictors.get(m).predict(valDfs.get(m).get(i), post);
                    pSum += post[1];
                }
                int pred = (pSum / predictors.size()) >= 0.5 ? 1 : 0;
                if (pred == val.get(i).y()) {
                    correct++;
                }
                total++;
            }
        }
        return correct / total;
    }

    // -------------------------------------------------------- candidate grids

    private static List<Map<String, Object>> rfGrid() {
        return List.of(
                Map.of("ntrees", 500, "mtry", 0, "maxDepth", 20, "nodeSize", 5, "subsample", 1.0),
                Map.of("ntrees", 500, "mtry", 6, "maxDepth", 20, "nodeSize", 1, "subsample", 1.0),
                Map.of("ntrees", 500, "mtry", 0, "maxDepth", 10, "nodeSize", 5, "subsample", 1.0),
                Map.of("ntrees", 300, "mtry", 0, "maxDepth", 20, "nodeSize", 1, "subsample", 1.0),
                Map.of("ntrees", 800, "mtry", 0, "maxDepth", 20, "nodeSize", 5, "subsample", 1.0),
                Map.of("ntrees", 500, "mtry", 0, "maxDepth", 20, "nodeSize", 2, "subsample", 0.8));
    }

    private static List<Map<String, Object>> baggingGrid() {
        return List.of(
                Map.of("ntrees", 200, "maxDepth", 20, "nodeSize", 5),
                Map.of("ntrees", 200, "maxDepth", 10, "nodeSize", 5),
                Map.of("ntrees", 100, "maxDepth", 20, "nodeSize", 1),
                Map.of("ntrees", 300, "maxDepth", 20, "nodeSize", 5));
    }

    private static List<Map<String, Object>> adaGrid() {
        return List.of(
                Map.of("ntrees", 300, "maxDepth", 2, "maxNodes", 4, "nodeSize", 1),
                Map.of("ntrees", 100, "maxDepth", 1, "maxNodes", 2, "nodeSize", 1),
                Map.of("ntrees", 500, "maxDepth", 2, "maxNodes", 6, "nodeSize", 1),
                Map.of("ntrees", 300, "maxDepth", 3, "maxNodes", 8, "nodeSize", 1),
                Map.of("ntrees", 200, "maxDepth", 2, "maxNodes", 4, "nodeSize", 1));
    }

    private static List<Map<String, Object>> gtbGrid() {
        return List.of(
                Map.of("ntrees", 500, "maxDepth", 3, "maxNodes", 8, "nodeSize", 5, "shrinkage", 0.05, "subsample", 0.7),
                Map.of("ntrees", 300, "maxDepth", 2, "maxNodes", 6, "nodeSize", 5, "shrinkage", 0.1, "subsample", 0.8),
                Map.of("ntrees", 500, "maxDepth", 5, "maxNodes", 31, "nodeSize", 5, "shrinkage", 0.05, "subsample", 0.7),
                Map.of("ntrees", 200, "maxDepth", 3, "maxNodes", 8, "nodeSize", 5, "shrinkage", 0.1, "subsample", 1.0),
                Map.of("ntrees", 800, "maxDepth", 2, "maxNodes", 6, "nodeSize", 5, "shrinkage", 0.03, "subsample", 0.7),
                Map.of("ntrees", 500, "maxDepth", 3, "maxNodes", 15, "nodeSize", 3, "shrinkage", 0.05, "subsample", 0.7));
    }

    private List<Map<String, Object>> gridFor(String model) {
        return switch (model) {
            case "Random Forest" -> rfGrid();
            case "Bagging" -> baggingGrid();
            case "AdaBoost" -> adaGrid();
            case "Gradient Boosting" -> gtbGrid();
            default -> throw new IllegalArgumentException("Unknown model: " + model);
        };
    }

    /**
     * Tunes one baseline by train-only 5-fold CV over its candidate
     * parameters. Decision Tree goes through the Tuple path
     * ({@link #cvAccuracy}); Logistic Regression, linear SVM and KNN
     * through the array path ({@link #cvAccuracyBaseline}). The winner and
     * its CV accuracy are recorded in the given maps and the CV log.
     */
    private void tuneBaseline(String kind, List<Map<String, Object>> candidates, List<Row> train,
                              Map<String, List<String>> cvLog,
                              Map<String, Map<String, Object>> bestParams,
                              Map<String, Double> bestCv) {
        List<String> log = new ArrayList<>();
        Map<String, Object> best = null;
        double bestScore = -1;
        for (Map<String, Object> params : candidates) {
            double cv;
            String label;
            if (kind.equals("Decision Tree")) {
                ModelConfig cfg = new ModelConfig(kind, params, allFeatures(), true);
                cv = cvAccuracy(cfg, train);
                label = cfg.describe();
            } else {
                cv = cvAccuracyBaseline(kind, params, train, allFeatures());
                label = kind + " " + params;
            }
            log.add(label + " -> CV accuracy " + String.format("%.4f", cv));
            System.out.printf("[CV] %-18s %-70s %.4f%n", kind, label, cv);
            if (cv > bestScore) {
                bestScore = cv;
                best = params;
            }
        }
        cvLog.put(kind, log);
        bestParams.put(kind, best);
        bestCv.put(kind, bestScore);
        System.out.printf("BEST (CV) %-18s %.4f :: %s%n", kind, bestScore, best);
    }

    /** Feature ranking from a Random Forest trained on the TRAIN split only (mean decrease in impurity). */
    public List<Integer> rankFeaturesByImportance(List<Row> trainRows) {
        Preprocessing prep = fitPreprocessing(trainRows, trainRows, true);
        DataFrame trainDf = toDataFrame(trainRows, prep, allFeatures());
        ModelConfig base = new ModelConfig("Random Forest",
                Map.of("ntrees", 500, "mtry", 0, "maxDepth", 20, "nodeSize", 5, "subsample", 1.0),
                allFeatures(), true);
        RandomForest rf = (RandomForest) trainOne(base, trainDf, SEED);
        double[] imp = rf.importance();
        List<Integer> order = new ArrayList<>();
        for (int j = 0; j < imp.length; j++) {
            order.add(j);
        }
        order.sort(Comparator.comparingDouble((Integer j) -> imp[j]).reversed());
        System.out.println("Feature importance (train RF, top 10):");
        for (int k = 0; k < Math.min(10, order.size()); k++) {
            System.out.printf("  %-18s %.4f%n", FeatureSpec.FEATURES.get(order.get(k)).key(), imp[order.get(k)]);
        }
        return order;
    }

    /**
     * Feature rankings, one per CV fold, each computed on that fold's
     * TRAINING part only (Random Forest importance). Subset candidates are
     * compared with these per-fold rankings, so the ranking for a fold
     * never sees that fold's validation rows — subset selection cannot leak
     * validation information into a CV score. (Ranking once on the whole
     * train split and reusing it for every fold would make the CV scores
     * of the smaller subsets slightly optimistic.)
     */
    public List<List<Integer>> foldRankings(List<Row> trainRows) {
        List<List<Row>> folds = stratifiedFolds(trainRows, CV_FOLDS, SEED);
        List<List<Integer>> out = new ArrayList<>();
        for (int f = 0; f < CV_FOLDS; f++) {
            List<Row> foldTrain = new ArrayList<>();
            for (int g = 0; g < CV_FOLDS; g++) {
                if (g != f) {
                    foldTrain.addAll(folds.get(g));
                }
            }
            out.add(rankFeaturesByImportance(foldTrain));
        }
        return out;
    }

    /**
     * 5-fold CV accuracy of one config whose feature subset is the top-k of
     * the PER-FOLD ranking (see {@link #foldRankings(List)}): in each fold
     * the subset is built from that fold's training rows only, then the
     * model is trained on the fold-train subset and evaluated on the
     * validation fold. Preprocessing is re-fitted inside every fold, as in
     * {@link #cvAccuracy(ModelConfig, List)}.
     */
    public double cvAccuracyTopK(ModelConfig base, List<Row> trainRows, int topK,
                                 List<List<Integer>> foldRankings) {
        List<List<Row>> folds = stratifiedFolds(trainRows, CV_FOLDS, SEED);
        double correct = 0;
        int total = 0;
        for (int f = 0; f < CV_FOLDS; f++) {
            List<Row> val = folds.get(f);
            List<Row> foldTrain = new ArrayList<>();
            for (int g = 0; g < CV_FOLDS; g++) {
                if (g != f) {
                    foldTrain.addAll(folds.get(g));
                }
            }
            if (base.oversample()) {
                foldTrain = oversample(foldTrain, SEED + f);
            }
            List<Integer> ranking = foldRankings.get(f);
            int[] subset = ranking.subList(0, Math.min(topK, ranking.size()))
                    .stream().mapToInt(Integer::intValue).toArray();
            ModelConfig cfg = new ModelConfig(base.model(), base.params(), subset,
                    base.clip(), base.oversample());
            Preprocessing prep = fitPreprocessing(foldTrain, foldTrain, cfg.clip());
            DataFrame trainDf = toDataFrame(foldTrain, prep, cfg.features());
            DataFrame valDf = toDataFrame(val, prep, cfg.features());
            Object model = trainOne(cfg, trainDf, SEED + f);
            Metrics.ModelMetrics m = evaluate(cfg.model(), predictorFor(model), valDf);
            correct += m.accuracy * valDf.size();
            total += valDf.size();
        }
        return correct / total;
    }

    // ---------------------------------------------------------------- training

    /** Plain CSV run (no verified history rows). */
    public TrainingResult train(Path csvPath, Path resourcesDir) throws IOException {
        return train(csvPath, resourcesDir, List.of());
    }

    /**
     * Full pipeline: load CSV, split, tune by train-only CV, retrain the best
     * config of each model on the full train split, evaluate ONCE on the
     * held-out test split, write metrics.json / preprocessing.json /
     * models/*.ser into {@code resourcesDir}. Also trains the baselines and
     * the screening model, and tunes the deployed decision thresholds on
     * train cross-validation out-of-fold probabilities only.
     *
     * @param extraVerifiedRows doctor-verified prediction-history rows
     *                          (features + verified outcome). They are
     *                          appended to the TRAIN split only — never to
     *                          the test split — and only rows with a filled
     *                          verified outcome are ever passed in.
     *                          Unverified model predictions are never
     *                          trained on. Empty for a plain CSV run.
     */
    public TrainingResult train(Path csvPath, Path resourcesDir, List<Row> extraVerifiedRows) throws IOException {
        MathEx.setSeed(SEED); // Smile's global RNG — reproducible training.
        List<Row> all = loadCsv(csvPath);
        if (malformedCells() > 0) {
            System.out.println("Malformed numeric cells treated as missing (mean-imputed): " + malformedCells());
        }
        if (all.size() != 541) {
            System.out.println("WARNING: expected 541 rows, found " + all.size());
        }
        List<List<Row>> split = stratifiedSplit(all, TRAIN_FRACTION, SEED);
        List<Row> train = split.get(0);
        List<Row> test = split.get(1);
        if (extraVerifiedRows != null && !extraVerifiedRows.isEmpty()) {
            train = new ArrayList<>(train);
            train.addAll(extraVerifiedRows);
            System.out.printf("Added %d doctor-verified history rows to the TRAIN split "
                    + "(test split unchanged, still pure held-out data)%n", extraVerifiedRows.size());
        }
        System.out.printf("Dataset: %d rows (class 0: %d, class 1: %d); train %d, test %d (stratified 80/20, seed %d)%n",
                all.size(), (int) all.stream().filter(r -> r.y() == 0).count(),
                (int) all.stream().filter(r -> r.y() == 1).count(), train.size(), test.size(), SEED);

        // ---- Stage 0: feature rankings.
        // The FULL-train ranking is used ONLY to build the final model's
        // feature list once a subset size has won on CV. Subset CANDIDATES
        // are compared by CV with per-fold rankings (perFoldRankings):
        // the ranking used to compare subsets never sees a validation fold,
        // so subset selection cannot leak validation information into the
        // CV score. The rankings are computed once here (5 Random Forest
        // fits on fold-train data) and reused for every model — that is
        // memoisation of the same fold's ranking, not leakage: each fold's
        // ranking still uses only that fold's training rows.
        List<Integer> ranked = rankFeaturesByImportance(train);
        List<List<Integer>> perFoldRankings = foldRankings(train);
        int[] topK = {25, 20, 15, 10};

        // ---- Stage A+B: per model — best hyperparameters (CV, all features),
        // then best feature subset + clipping (CV, best hyperparameters).
        List<String> modelNames = List.of("Random Forest", "Bagging", "AdaBoost", "Gradient Boosting");
        Map<String, TunedModel> tuned = new LinkedHashMap<>();
        Map<String, List<String>> cvLog = new LinkedHashMap<>();
        for (String name : modelNames) {
            List<String> log = new ArrayList<>();
            TunedModel best = null;
            for (Map<String, Object> params : gridFor(name)) {
                ModelConfig cfg = new ModelConfig(name, params, allFeatures(), true);
                double cv = cvAccuracy(cfg, train);
                log.add(cfg.describe() + " -> CV accuracy " + String.format("%.4f", cv));
                System.out.printf("[CV] %-18s %-70s %.4f%n", name, cfg.describe(), cv);
                if (best == null || cv > best.cvAccuracy()) {
                    best = new TunedModel(cfg, cv);
                }
            }
            // Feature subsets + clipping, at the best hyperparameters.
            // Each top-k is scored by CV with the ranking re-computed inside
            // each fold (cvAccuracyTopK); the stored winner uses the
            // full-train ranking for its final feature list.
            for (int k : topK) {
                for (boolean clip : new boolean[]{true, false}) {
                    ModelConfig probe = new ModelConfig(name, best.config().params(), allFeatures(), clip);
                    double cv = cvAccuracyTopK(probe, train, k, perFoldRankings);
                    int[] subset = ranked.subList(0, k).stream().mapToInt(Integer::intValue).toArray();
                    ModelConfig cfg = new ModelConfig(name, best.config().params(), subset, clip);
                    log.add(cfg.describe() + " (top-" + k + " per-fold ranking, clip=" + clip
                            + ") -> CV accuracy " + String.format("%.4f", cv));
                    System.out.printf("[CV] %-18s top-%d (per-fold) clip=%-5s %.4f%n", name, k, clip, cv);
                    if (cv > best.cvAccuracy()) {
                        best = new TunedModel(cfg, cv);
                    }
                }
            }
            // Clipping on/off at the best subset too (subset all-41 case).
            ModelConfig noClip = new ModelConfig(name, best.config().params(), best.config().features(), false);
            double cvNoClip = cvAccuracy(noClip, train);
            log.add(noClip.describe() + " -> CV accuracy " + String.format("%.4f", cvNoClip));
            if (cvNoClip > best.cvAccuracy()) {
                best = new TunedModel(noClip, cvNoClip);
            }
            // Minority oversampling at the best config (train folds only).
            ModelConfig over = new ModelConfig(name, best.config().params(), best.config().features(),
                    best.config().clip(), true);
            double cvOver = cvAccuracy(over, train);
            log.add(over.describe() + " -> CV accuracy " + String.format("%.4f", cvOver));
            System.out.printf("[CV] %-18s oversample=true %.4f%n", name, cvOver);
            if (cvOver > best.cvAccuracy()) {
                best = new TunedModel(over, cvOver);
            }
            tuned.put(name, best);
            cvLog.put(name, log);
            System.out.printf("BEST (CV) %-18s %.4f :: %s%n", name, best.cvAccuracy(), best.config().describe());
        }

        // ---- Stage B2: baselines — same split, same train-only CV protocol.
        // They are what "89% accuracy" has to beat to mean anything.
        // Majority-class needs no tuning (it always predicts 0).
        Map<String, Map<String, Object>> baselineBestParams = new LinkedHashMap<>();
        Map<String, Double> baselineBestCv = new LinkedHashMap<>();
        // Logistic Regression: L2 lambda.
        tuneBaseline("Logistic Regression",
                List.of(Map.of("lambda", 0.01), Map.of("lambda", 0.1), Map.of("lambda", 1.0)),
                train, cvLog, baselineBestParams, baselineBestCv);
        // Single Decision Tree: depth / node size (via the Tuple path).
        tuneBaseline("Decision Tree",
                List.of(Map.of("maxDepth", 3, "maxNodes", 31, "nodeSize", 1),
                        Map.of("maxDepth", 5, "maxNodes", 31, "nodeSize", 1),
                        Map.of("maxDepth", 5, "maxNodes", 31, "nodeSize", 5),
                        Map.of("maxDepth", 10, "maxNodes", 31, "nodeSize", 1)),
                train, cvLog, baselineBestParams, baselineBestCv);
        // Linear SVM: C.
        tuneBaseline("SVM (Linear)",
                List.of(Map.of("C", 0.1), Map.of("C", 1.0), Map.of("C", 10.0)),
                train, cvLog, baselineBestParams, baselineBestCv);
        // KNN: neighbours.
        tuneBaseline("KNN",
                List.of(Map.of("k", 3), Map.of("k", 5), Map.of("k", 9), Map.of("k", 15)),
                train, cvLog, baselineBestParams, baselineBestCv);

        // ---- Stage B3: screening model — Gradient Boosting on the 23
        // non-invasive features ONLY (no labs, no ultrasound). Same GTB
        // grid as the full model, tuned by train-only CV.
        int[] screeningFeatures = FeatureSpec.screeningIndices();
        List<String> screeningLog = new ArrayList<>();
        TunedModel screeningBest = null;
        for (Map<String, Object> params : gtbGrid()) {
            ModelConfig cfg = new ModelConfig(SCREENING_MODEL, params, screeningFeatures, true);
            double cv = cvAccuracy(cfg, train);
            screeningLog.add(cfg.describe() + " -> CV accuracy " + String.format("%.4f", cv));
            System.out.printf("[CV] %-28s %-70s %.4f%n", "Screening-GBM", cfg.describe(), cv);
            if (screeningBest == null || cv > screeningBest.cvAccuracy()) {
                screeningBest = new TunedModel(cfg, cv);
            }
        }
        cvLog.put(SCREENING_MODEL, screeningLog);
        System.out.printf("BEST (CV) %-28s %.4f :: %s%n", "Screening-GBM",
                screeningBest.cvAccuracy(), screeningBest.config().describe());

        // ---- Stage C: soft-voting ensemble of the four tuned models (CV comparison on train).
        double votingCv = cvAccuracyVoting(new ArrayList<>(tuned.values()), train);
        System.out.printf("Soft-voting ensemble CV accuracy: %.4f%n", votingCv);
        double bestSingleCv = tuned.values().stream().mapToDouble(TunedModel::cvAccuracy).max().orElse(0);
        boolean useVoting = votingCv > bestSingleCv;
        System.out.println(useVoting
                ? "Voting ensemble wins on train CV -> deployed model is the soft-voting ensemble."
                : "Voting ensemble does NOT beat the best single model on train CV -> deploying best single model.");

        // ---- Final fit on the FULL train split + ONE evaluation on held-out test.
        // Shared preprocessing.json: fitted on train; its min/max + class counts
        // are descriptive full-dataset statistics for the /api/features form
        // ranges only — they are NEVER model inputs (each model's transform
        // uses its own train-fitted mean/std/clip, stored in metrics.json).
        Preprocessing prep = fitPreprocessing(train, all, true);
        // The deployed model's preprocessing (clip setting + selected features)
        // is what the API must reproduce at inference time.
        ModelConfig deployedConfig;
        String deployedName;
        if (useVoting) {
            deployedName = "Soft Voting Ensemble";
            deployedConfig = null; // voting uses each model's own config
        } else {
            deployedName = tuned.entrySet().stream()
                    .max(Comparator.comparingDouble(e -> e.getValue().cvAccuracy()))
                    .orElseThrow().getKey();
            deployedConfig = tuned.get(deployedName).config();
        }
        // Selected features for the API = deployed model's subset (voting: union of the four subsets).
        int[] apiFeatures;
        if (useVoting) {
            apiFeatures = tuned.values().stream()
                    .flatMapToInt(tm -> Arrays.stream(tm.config().features()))
                    .distinct().sorted().toArray();
            // Voting needs each model standardised with ITS OWN clip setting;
            // the API therefore stores one preprocessing per model — see
            // preprocessing.json "modelPreprocessing". For the shared fields
            // (min/max/default) the full-train fit with clipping is used.
        } else {
            apiFeatures = deployedConfig.features();
        }
        prep.selectedFeatures = Arrays.stream(apiFeatures)
                .mapToObj(j -> FeatureSpec.FEATURES.get(j).key()).toList();

        Metrics metrics = new Metrics();
        metrics.trainSize = train.size();
        metrics.testSize = test.size();
        metrics.seed = SEED;
        metrics.generatedAt = LocalDateTime.now().toString();
        metrics.cvFolds = CV_FOLDS;
        metrics.tuningMethod = "5-fold stratified CV on the TRAIN split only (seed 42). "
                + "Hyperparameter grid + feature-subset (RF importance re-ranked inside each CV fold, "
                + "on the fold's training rows only) + clipping on/off compared by CV. "
                + "Test split evaluated exactly once, after all tuning. Preprocessing re-fitted inside every CV fold. "
                + "Baselines and the screening model use the same split and the same train-only CV protocol. "
                + "Decision thresholds are tuned on train CV out-of-fold probabilities only, never on the test set.";
        metrics.verifiedRowsUsed = extraVerifiedRows == null ? 0 : extraVerifiedRows.size();
        metrics.cvLog = cvLog;
        metrics.votingCvAccuracy = votingCv;
        metrics.votingUsed = useVoting;

        // Feature importances (mean decrease in impurity), all fitted on TRAIN only:
        // Random Forest + Gradient Boosting on all 41 features, screening GBM on its 23.
        metrics.featureImportance = importanceMap(train, tuned.get("Random Forest").config());
        metrics.gbmImportance = importanceMap(train, tuned.get("Gradient Boosting").config());
        metrics.screeningImportance = importanceMap(train, screeningBest.config());

        // ---- Final fit of the four ensembles + their test probabilities (ONE test evaluation).
        Map<String, Object> trained = new LinkedHashMap<>();
        Map<String, ProbPredictions> testProbs = new LinkedHashMap<>();
        for (String name : modelNames) {
            TunedModel tm = tuned.get(name);
            ModelConfig cfg = tm.config();
            // Train-only fit: min/max here also come from train only, so the
            // stored per-model transform contains zero test information.
            List<Row> finalTrain = cfg.oversample() ? oversample(train, SEED) : train;
            Preprocessing modelPrep = fitPreprocessing(finalTrain, finalTrain, cfg.clip());
            DataFrame trainDf = toDataFrame(finalTrain, modelPrep, cfg.features());
            DataFrame testDf = toDataFrame(test, modelPrep, cfg.features());
            Object model = trainOne(cfg, trainDf, SEED);
            trained.put(name, model);
            testProbs.put(name, collectProbs(predictorFor(model), testDf));
            // Per-model preprocessing (clip + means/stds) for voting inference.
            metrics.modelPreprocessing.put(name, modelPrep);
        }
        // Voting test probabilities: average P(PCOS) of the four models.
        ProbPredictions votingPp = averageProbs(new ArrayList<>(testProbs.values()));

        // ---- Baselines: final fit on the full train split + test probabilities.
        Map<String, ProbPredictions> baselineProbs = new LinkedHashMap<>();
        for (String kind : List.of("Logistic Regression", "Decision Tree", "SVM (Linear)", "KNN")) {
            Map<String, Object> params = baselineBestParams.get(kind);
            ProbPredictions pp;
            Preprocessing bp = fitPreprocessing(train, train, true);
            if (kind.equals("Decision Tree")) {
                ModelConfig cfg = new ModelConfig(kind, params, allFeatures(), true);
                DataFrame trainDf = toDataFrame(train, bp, allFeatures());
                DataFrame testDf = toDataFrame(test, bp, allFeatures());
                Object model = trainOne(cfg, trainDf, SEED);
                pp = collectProbs(predictorFor(model), testDf);
            } else {
                Matrix trainM = toMatrix(train, bp, allFeatures());
                Matrix testM = toMatrix(test, bp, allFeatures());
                Classifier<double[]> model = fitBaseline(kind, params, trainM, SEED);
                int[] actual = testM.y();
                double[] prob = new double[actual.length];
                for (int i = 0; i < actual.length; i++) {
                    double[] post = new double[2];
                    model.predict(testM.x()[i], post);
                    prob[i] = post[0] + post[1] > 0 ? post[1] / (post[0] + post[1]) : 0.5;
                }
                pp = new ProbPredictions(actual, prob);
            }
            baselineProbs.put(kind, pp);
        }
        // Majority-class baseline: P(PCOS) = 0 for every row (always predicts 0).
        ProbPredictions majorityPp = new ProbPredictions(
                test.stream().mapToInt(Row::y).toArray(), new double[test.size()]);

        // ---- Screening model: final fit on the full train split + test probabilities.
        Object screeningModel;
        ProbPredictions screeningPp;
        {
            ModelConfig cfg = screeningBest.config();
            Preprocessing sp = fitPreprocessing(train, train, cfg.clip());
            DataFrame trainDf = toDataFrame(train, sp, screeningFeatures);
            DataFrame testDf = toDataFrame(test, sp, screeningFeatures);
            screeningModel = trainOne(cfg, trainDf, SEED);
            screeningPp = collectProbs(predictorFor(screeningModel), testDf);
            metrics.modelPreprocessing.put(SCREENING_MODEL, sp);
        }

        // ---- Threshold tuning (train CV out-of-fold ONLY — never the test set).
        // Deployed model: threshold maximising OOF F1 (ties -> higher recall).
        Metrics.ThresholdReport deployedReport = new Metrics.ThresholdReport();
        ProbPredictions deployedTestPp;
        if (useVoting) {
            deployedReport.model = "Soft Voting Ensemble";
            deployedTestPp = votingPp;
            ProbPredictions oof = oofProbsVoting(new ArrayList<>(tuned.values()), train);
            fillThresholdReport(deployedReport, oof, deployedTestPp);
        } else {
            deployedReport.model = deployedName;
            deployedTestPp = testProbs.get(deployedName);
            ProbPredictions oof = oofProbsSingle(deployedConfig, train);
            fillThresholdReport(deployedReport, oof, deployedTestPp);
        }
        metrics.deployedThreshold = deployedReport;
        System.out.printf("Threshold (deployed %s): tuned=%.2f (OOF P=%.4f R=%.4f F1=%.4f) | "
                        + "test at tuned: acc=%.4f P=%.4f R=%.4f F1=%.4f | test at 0.5: acc=%.4f P=%.4f R=%.4f F1=%.4f%n",
                deployedReport.model, deployedReport.threshold,
                deployedReport.oofPrecision, deployedReport.oofRecall, deployedReport.oofF1,
                deployedReport.testAccuracy, deployedReport.testPrecision,
                deployedReport.testRecall, deployedReport.testF1,
                deployedReport.defaultThresholdTestAccuracy, deployedReport.defaultThresholdTestPrecision,
                deployedReport.defaultThresholdTestRecall, deployedReport.defaultThresholdTestF1);
        // Screening model: same procedure, its own threshold.
        Metrics.ThresholdReport screeningReport = new Metrics.ThresholdReport();
        screeningReport.model = SCREENING_MODEL;
        fillThresholdReport(screeningReport, oofProbsSingle(screeningBest.config(), train), screeningPp);
        metrics.screeningThreshold = screeningReport;
        System.out.printf("Threshold (screening): tuned=%.2f (OOF P=%.4f R=%.4f F1=%.4f) | "
                        + "test at tuned: acc=%.4f P=%.4f R=%.4f F1=%.4f | test at 0.5: acc=%.4f P=%.4f R=%.4f F1=%.4f%n",
                screeningReport.threshold,
                screeningReport.oofPrecision, screeningReport.oofRecall, screeningReport.oofF1,
                screeningReport.testAccuracy, screeningReport.testPrecision,
                screeningReport.testRecall, screeningReport.testF1,
                screeningReport.defaultThresholdTestAccuracy, screeningReport.defaultThresholdTestPrecision,
                screeningReport.defaultThresholdTestRecall, screeningReport.defaultThresholdTestF1);

        // ---- Assemble the metrics entries.
        for (String name : modelNames) {
            Metrics.ModelMetrics m;
            if (!useVoting && name.equals(deployedName)) {
                // Deployed single model: its entry describes the deployed
                // decision rule, i.e. the TUNED threshold.
                m = metricsFromProbs(name, testProbs.get(name).actual(),
                        testProbs.get(name).prob(), deployedReport.threshold);
            } else {
                m = metricsFromProbs(name, testProbs.get(name).actual(),
                        testProbs.get(name).prob(), 0.5);
            }
            fillModelEntry(m, tuned.get(name));
            m.mode = "full";
            metrics.models.add(m);
            System.out.printf("[TEST] %-18s accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f "
                            + "ROC-AUC=%.4f PR-AUC=%.4f Brier=%.4f (CV %.4f, threshold %.2f)%n",
                    m.name, m.accuracy, m.precision, m.recall, m.f1,
                    m.rocAuc, m.prAuc, m.brierScore, m.cvAccuracy, m.threshold);
        }
        // Voting ensemble entry (the deployed model when voting won on train CV).
        Metrics.ModelMetrics votingEntry = metricsFromProbs("Soft Voting Ensemble",
                votingPp.actual(), votingPp.prob(),
                useVoting ? deployedReport.threshold : 0.5);
        votingEntry.cvAccuracy = votingCv;
        votingEntry.hyperparameters = "average P(PCOS) of the four tuned models";
        votingEntry.nFeatures = 41;
        votingEntry.clipApplied = true;
        votingEntry.featureSubset = Arrays.stream(allFeatures())
                .mapToObj(j -> FeatureSpec.FEATURES.get(j).key()).toList();
        votingEntry.mode = "full";
        metrics.models.add(votingEntry);
        metrics.votingTestAccuracy = votingEntry.accuracy;
        metrics.votingConfusionMatrix = votingEntry.confusionMatrix;
        metrics.votingRocAuc = votingEntry.rocAuc;
        metrics.votingPrAuc = votingEntry.prAuc;
        metrics.votingBrierScore = votingEntry.brierScore;
        metrics.votingRocCurve = votingEntry.rocCurve;
        metrics.votingAccuracyCiLow = votingEntry.accuracyCiLow;
        metrics.votingAccuracyCiHigh = votingEntry.accuracyCiHigh;
        System.out.printf("[TEST] %-18s accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f "
                        + "ROC-AUC=%.4f PR-AUC=%.4f Brier=%.4f (CV %.4f, threshold %.2f)%n",
                votingEntry.name, votingEntry.accuracy, votingEntry.precision,
                votingEntry.recall, votingEntry.f1, votingEntry.rocAuc,
                votingEntry.prAuc, votingEntry.brierScore, votingEntry.cvAccuracy,
                votingEntry.threshold);
        // Screening entry (at its tuned threshold).
        Metrics.ModelMetrics sEntry = metricsFromProbs(SCREENING_MODEL,
                screeningPp.actual(), screeningPp.prob(), screeningReport.threshold);
        sEntry.cvAccuracy = screeningBest.cvAccuracy();
        sEntry.hyperparameters = screeningBest.config().describe();
        sEntry.nFeatures = screeningFeatures.length;
        sEntry.clipApplied = true;
        sEntry.oversample = false;
        sEntry.featureSubset = Arrays.stream(screeningFeatures)
                .mapToObj(j -> FeatureSpec.FEATURES.get(j).key()).toList();
        sEntry.mode = "screening";
        sEntry.screening = true;
        metrics.models.add(sEntry);
        System.out.printf("[TEST] %-28s accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f "
                        + "ROC-AUC=%.4f PR-AUC=%.4f Brier=%.4f (CV %.4f, threshold %.2f)%n",
                sEntry.name, sEntry.accuracy, sEntry.precision, sEntry.recall,
                sEntry.f1, sEntry.rocAuc, sEntry.prAuc, sEntry.brierScore,
                sEntry.cvAccuracy, sEntry.threshold);
        // Baselines (at the default 0.5 threshold — they are reference points, not deployed).
        for (String kind : BASELINE_NAMES) {
            Metrics.ModelMetrics b;
            if (kind.equals("Majority Class")) {
                b = metricsFromProbs(kind, majorityPp.actual(), majorityPp.prob(), 0.5);
                b.hyperparameters = "always predict 0 (majority class, 67.3% of the data)";
                b.nFeatures = 0;
                // CV accuracy of "always predict 0" on train = train share of class 0.
                b.cvAccuracy = (double) train.stream().filter(r -> r.y() == 0).count() / train.size();
            } else {
                b = metricsFromProbs(kind, baselineProbs.get(kind).actual(),
                        baselineProbs.get(kind).prob(), 0.5);
                b.hyperparameters = kind + " " + baselineBestParams.get(kind);
                b.nFeatures = 41;
                b.cvAccuracy = baselineBestCv.get(kind);
            }
            b.mode = "full";
            b.clipApplied = true;
            b.featureSubset = Arrays.stream(allFeatures())
                    .mapToObj(j -> FeatureSpec.FEATURES.get(j).key()).toList();
            metrics.baselines.add(b);
            System.out.printf("[TEST] %-18s accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f "
                            + "ROC-AUC=%.4f PR-AUC=%.4f Brier=%.4f (CV %.4f)%n",
                    b.name, b.accuracy, b.precision, b.recall, b.f1,
                    b.rocAuc, b.prAuc, b.brierScore, b.cvAccuracy);
        }

        metrics.bestModel = deployedName;
        // modelAccuracy reported by /api/predict = test accuracy of the deployed
        // model AT ITS TUNED THRESHOLD (the decision rule actually in production).
        metrics.bestModelTestAccuracy = useVoting
                ? votingEntry.accuracy
                : metrics.byName(deployedName).accuracy;
        System.out.printf("Deployed model (best TRAIN CV): %s (CV %.4f, test accuracy at tuned threshold %.4f)%n",
                deployedName, useVoting ? votingCv : tuned.get(deployedName).cvAccuracy(),
                metrics.bestModelTestAccuracy);

        // Persist artefacts.
        Path modelsDir = resourcesDir.resolve("models");
        Files.createDirectories(modelsDir);
        for (Map.Entry<String, Object> e : trained.entrySet()) {
            Path out = modelsDir.resolve(MODEL_FILES.get(e.getKey()));
            try (OutputStream os = Files.newOutputStream(out);
                 ObjectOutputStream oos = new ObjectOutputStream(os)) {
                oos.writeObject(e.getValue());
            }
            System.out.println("Saved model: " + out);
        }
        Path screeningOut = modelsDir.resolve(SCREENING_FILE);
        try (OutputStream os = Files.newOutputStream(screeningOut);
             ObjectOutputStream oos = new ObjectOutputStream(os)) {
            oos.writeObject(screeningModel);
        }
        System.out.println("Saved model: " + screeningOut);
        // The shared preprocessing.json carries the DEPLOYED model's transform
        // (or, for voting, the union-feature transform with clipping on — the
        // per-model transforms live inside metrics.json for reference and the
        // API re-derives each model's transform from metrics.json at startup).
        mapper.writeValue(resourcesDir.resolve("metrics.json").toFile(), metrics);
        mapper.writeValue(resourcesDir.resolve("preprocessing.json").toFile(), prep);
        System.out.println("Saved metrics.json + preprocessing.json to " + resourcesDir);
        return new TrainingResult(metrics, prep);
    }

    private static double[][] dataframeToMatrix(DataFrame df) {
        double[][] m = new double[df.size()][df.ncol()];
        for (int i = 0; i < df.size(); i++) {
            Tuple row = df.get(i);
            for (int j = 0; j < df.ncol(); j++) {
                m[i][j] = row.getDouble(j);
            }
        }
        return m;
    }
}
