package com.pcod.ml;

import tools.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import smile.classification.Classifier;
import smile.data.DataFrame;
import smile.data.Tuple;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Loads the trained artefacts once at startup (NOT retrained on every boot):
 * the four serialised Smile models, metrics.json and preprocessing.json.
 *
 * <p>Artefacts are loaded from the external resources directory FIRST
 * ({@code pcod.resources.dir}, i.e. {@code PCOD_RESOURCES_DIR}) when it
 * holds a metrics.json — that is where POST /api/retrain writes, so a
 * retrained model survives an application restart. When the directory
 * has no metrics.json (fresh checkout, or the jar's baked-in artefacts),
 * the classpath copies are used instead. A partial or corrupt external
 * directory falls back to the classpath rather than leaving the service
 * down.</p>
 *
 * <p>All loaded state lives in ONE immutable {@link Snapshot} held in a
 * single volatile field: metrics, preprocessing and models are swapped
 * together, atomically, so a prediction can never observe new metrics
 * with old models (or vice versa). {@link #reload(Path)} builds the whole
 * snapshot first and swaps it in one assignment; it is also synchronised
 * so two concurrent reloads cannot interleave.</p>
 *
 * <p>Inference reproduces each model's exact training transform: the
 * per-model preprocessing (train-fitted mean/std/clip) and feature subset
 * stored in metrics.json. The deployed model is the best TRAIN-CV model
 * (a single model, or the soft-voting ensemble averaging the four models'
 * probabilities when the ensemble won on train CV). A second model, the
 * screening model (Gradient Boosting on the 23 non-invasive features),
 * is served when the caller asks for mode=screening.</p>
 *
 * <p>Both deployed models use their TUNED decision threshold (chosen on
 * train CV out-of-fold probabilities, stored in metrics.json), not 0.5:
 * a prediction is PCOS when P(PCOS) &gt;= threshold. Confidence is the
 * probability of the PREDICTED class (P(PCOS) when the prediction is 1,
 * 1 - P(PCOS) when it is 0) — not max(p, 1-p), which would report the
 * wrong side's probability whenever the threshold is not 0.5.</p>
 */
@Service
public class ModelStore {

    private static final Logger log = LoggerFactory.getLogger(ModelStore.class);

    public static final String VOTING = "Soft Voting Ensemble";

    private final ObjectMapper mapper = new ObjectMapper();

    /**
     * One immutable, fully-loaded set of artefacts. Swapped atomically:
     * readers take the reference once and use it consistently, so a
     * prediction never mixes metrics from one training run with models
     * from another.
     */
    public record Snapshot(Metrics metrics, Preprocessing preprocessing,
                           Map<String, Object> models, boolean ready) {}

    private volatile Snapshot snapshot = new Snapshot(null, null, Map.of(), false);

    /** External directory retraining writes to (see application.properties). */
    @Value("${pcod.resources.dir:src/main/resources}")
    private String resourcesDir;

    @PostConstruct
    public void init() {
        Path external = Path.of(resourcesDir);
        if (Files.isRegularFile(external.resolve("metrics.json"))) {
            try {
                snapshot = loadFromDirectory(external);
                log.info("Loaded {} models from external resources dir {}; deployed model: {} (test accuracy {})",
                        snapshot.models().size(), external, snapshot.metrics().bestModel,
                        snapshot.metrics().bestModelTestAccuracy);
                return;
            } catch (Exception e) {
                log.warn("External resources dir {} is unusable ({}); falling back to the "
                        + "classpath artefacts baked into the jar.", external, e.toString());
            }
        }
        try {
            snapshot = loadFromClasspath();
            log.info("Loaded {} models from classpath; deployed model: {} (test accuracy {})",
                    snapshot.models().size(), snapshot.metrics().bestModel,
                    snapshot.metrics().bestModelTestAccuracy);
        } catch (Exception e) {
            snapshot = new Snapshot(null, null, Map.of(), false);
            log.warn("Models not loaded at startup (run the trainer first: "
                    + "mvn exec:java -Dexec.mainClass=com.pcod.trainer.ModelTrainer). "
                    + "/api/predict will return 503 until models are available. Cause: {}", e.toString());
        }
    }

    private Snapshot loadFromClasspath() throws IOException {
        Metrics m = readClasspath("metrics.json", Metrics.class);
        Preprocessing p = readClasspath("preprocessing.json", Preprocessing.class);
        Map<String, Object> loaded = new LinkedHashMap<>();
        for (Map.Entry<String, String> e : TrainingService.MODEL_FILES.entrySet()) {
            try (InputStream is = getClass().getResourceAsStream("/models/" + e.getValue());
                 ObjectInputStream ois = new ObjectInputStream(is)) {
                loaded.put(e.getKey(), ois.readObject());
            } catch (ClassNotFoundException ex) {
                throw new IOException("Cannot deserialise model " + e.getValue(), ex);
            }
        }
        loadScreening(loaded, "/models/" + TrainingService.SCREENING_FILE, null);
        return new Snapshot(m, p, Map.copyOf(loaded), true);
    }

    private Snapshot loadFromDirectory(Path dir) throws IOException {
        Metrics m = mapper.readValue(dir.resolve("metrics.json").toFile(), Metrics.class);
        Preprocessing p = mapper.readValue(dir.resolve("preprocessing.json").toFile(), Preprocessing.class);
        Map<String, Object> loaded = new LinkedHashMap<>();
        for (Map.Entry<String, String> e : TrainingService.MODEL_FILES.entrySet()) {
            Path f = dir.resolve("models").resolve(e.getValue());
            try (InputStream is = Files.newInputStream(f);
                 ObjectInputStream ois = new ObjectInputStream(is)) {
                loaded.put(e.getKey(), ois.readObject());
            } catch (ClassNotFoundException ex) {
                throw new IOException("Cannot deserialise model " + f, ex);
            }
        }
        loadScreening(loaded, null, dir.resolve("models").resolve(TrainingService.SCREENING_FILE));
        return new Snapshot(m, p, Map.copyOf(loaded), true);
    }

    /** Loads the screening model if present (artefacts written before it existed have none). */
    private void loadScreening(Map<String, Object> loaded, String classpathResource, Path file) throws IOException {
        try {
            InputStream is = file != null
                    ? Files.newInputStream(file)
                    : getClass().getResourceAsStream(classpathResource);
            if (is == null) {
                log.warn("Screening model not found ({}); mode=screening will be unavailable "
                        + "until the trainer runs again.", classpathResource);
                return;
            }
            try (ObjectInputStream ois = new ObjectInputStream(is)) {
                loaded.put(TrainingService.SCREENING_MODEL, ois.readObject());
            }
        } catch (ClassNotFoundException e) {
            throw new IOException("Cannot deserialise screening model", e);
        }
    }

    /**
     * Re-load metrics / preprocessing / models from a directory on disk
     * (after retraining). Builds the whole snapshot first, then swaps it
     * in with a single volatile assignment — in-flight predictions keep
     * using the old snapshot until the swap, and never see a half-loaded
     * one. Synchronised against concurrent reloads.
     */
    public synchronized void reload(Path resourcesDir) throws IOException {
        Snapshot next = loadFromDirectory(resourcesDir);
        snapshot = next;
        log.info("Reloaded {} models from {}; deployed model: {}",
                next.models().size(), resourcesDir, next.metrics().bestModel);
    }

    private <T> T readClasspath(String name, Class<T> type) throws IOException {
        try (InputStream is = getClass().getResourceAsStream("/" + name)) {
            if (is == null) {
                throw new IOException("Classpath resource not found: " + name);
            }
            return mapper.readValue(is, type);
        }
    }

    public boolean isReady() {
        return snapshot.ready();
    }

    public Metrics getMetrics() {
        return snapshot.metrics();
    }

    public Preprocessing getPreprocessing() {
        return snapshot.preprocessing();
    }

    /** Result of one prediction with the deployed model. */
    public record Prediction(int prediction, String label, double probability,
                             double confidence, String model, double modelAccuracy,
                             double threshold) {}

    /** P(PCOS) from one named model, using that model's own preprocessing + feature subset. */
    private double probabilityFromModel(Snapshot snap, String name, double[] raw) {
        Object model = snap.models().get(name);
        if (model == null) {
            throw new IllegalStateException("Model not loaded: " + name);
        }
        Metrics metrics = snap.metrics();
        Metrics.ModelMetrics mm = metrics.byName(name);
        Preprocessing prep = metrics.modelPreprocessing != null
                ? metrics.modelPreprocessing.get(name) : null;
        List<String> subsetKeys = mm != null && mm.featureSubset != null && !mm.featureSubset.isEmpty()
                ? mm.featureSubset : null;
        if (prep == null || subsetKeys == null) {
            // Fallback for artefacts written before per-model preprocessing existed.
            prep = snap.preprocessing();
            subsetKeys = snap.preprocessing().selectedFeatures != null
                    && !snap.preprocessing().selectedFeatures.isEmpty()
                    ? snap.preprocessing().selectedFeatures
                    : FeatureSpec.FEATURES.stream().map(FeatureSpec.Feature::key).toList();
        }
        double[] z = prep.transform(raw);
        int[] subsetIdx = subsetKeys.stream().mapToInt(FeatureSpec::indexOf).toArray();
        String[] names = new String[subsetIdx.length + 1];
        double[][] one = new double[1][subsetIdx.length + 1];
        for (int k = 0; k < subsetIdx.length; k++) {
            names[k] = subsetKeys.get(k);
            one[0][k] = z[subsetIdx[k]];
        }
        names[subsetIdx.length] = FeatureSpec.TARGET;
        one[0][subsetIdx.length] = 0.0; // dummy target; the formula ignores it for prediction
        DataFrame df = TrainingService.buildDataFrame(one, names);
        Tuple row = df.get(0);

        double[] post = new double[2];
        if (model instanceof BaggingClassifier bagging) {
            bagging.predict(row, post);
        } else if (model instanceof Classifier<?> classifier) {
            @SuppressWarnings("unchecked")
            Classifier<Tuple> clf = (Classifier<Tuple>) classifier;
            clf.predict(row, post);
        } else {
            throw new IllegalStateException("Unknown model type: " + model.getClass());
        }
        return clamp01(post[1]);
    }

    /**
     * Confidence = probability of the PREDICTED class. With a tuned
     * threshold below 0.5 (screening, e.g. 0.30), max(p, 1-p) would report the
     * probability of the class NOT predicted (e.g. p = 0.30 predicts PCOS
     * but max() returns 70%, the No-PCOS side). This is the fix for that.
     */
    private static double confidenceFor(int prediction, double p1) {
        return prediction == 1 ? p1 : 1.0 - p1;
    }

    /**
     * Predicts with the deployed (full) model. {@code raw} holds the 41 raw-scale
     * feature values (NaN = missing -> training median). Applies the same
     * clip + standardisation as training, then the saved model, and decides
     * with the model's TUNED threshold (see metrics.json).
     */
    public Prediction predict(double[] raw) {
        return predict(raw, "full");
    }

    /**
     * Predicts in one mode: "full" (the deployed 41-feature model) or
     * "screening" (Gradient Boosting on the 23 non-invasive features).
     * The screening model needs no lab or ultrasound values — any of those
     * in the request are simply ignored.
     */
    public Prediction predict(double[] raw, String mode) {
        Snapshot snap = snapshot;
        if (!snap.ready()) {
            throw new IllegalStateException("Models are not loaded yet — train first.");
        }
        Metrics metrics = snap.metrics();
        if ("screening".equalsIgnoreCase(mode)) {
            if (!snap.models().containsKey(TrainingService.SCREENING_MODEL)) {
                throw new IllegalStateException("Screening model is not loaded — "
                        + "train again to create it (see README).");
            }
            double p1 = probabilityFromModel(snap, TrainingService.SCREENING_MODEL, raw);
            double threshold = metrics.screeningThreshold != null
                    ? metrics.screeningThreshold.threshold : 0.5;
            Metrics.ModelMetrics mm = metrics.byName(TrainingService.SCREENING_MODEL);
            int prediction = p1 >= threshold ? 1 : 0;
            double confidence = confidenceFor(prediction, p1);
            return new Prediction(
                    prediction,
                    prediction == 1 ? "High risk \u2014 refer for testing" : "Low risk",
                    p1,
                    confidence,
                    TrainingService.SCREENING_MODEL,
                    mm == null ? 0.0 : mm.accuracy,
                    threshold);
        }
        double p1;
        if (VOTING.equals(metrics.bestModel)) {
            List<Double> probs = new ArrayList<>();
            for (String name : TrainingService.MODEL_FILES.keySet()) {
                probs.add(probabilityFromModel(snap, name, raw));
            }
            p1 = probs.stream().mapToDouble(Double::doubleValue).average().orElse(0.5);
        } else {
            p1 = probabilityFromModel(snap, metrics.bestModel, raw);
        }
        double threshold = metrics.deployedThreshold != null
                ? metrics.deployedThreshold.threshold : 0.5;
        int prediction = p1 >= threshold ? 1 : 0;
        double confidence = confidenceFor(prediction, p1);
        return new Prediction(
                prediction,
                prediction == 1 ? "High risk \u2014 refer for testing" : "Low risk",
                p1,
                confidence,
                metrics.bestModel,
                metrics.bestModelTestAccuracy,
                threshold);
    }

    private static double clamp01(double v) {
        if (Double.isNaN(v)) {
            return 0.5;
        }
        return Math.min(1.0, Math.max(0.0, v));
    }
}
