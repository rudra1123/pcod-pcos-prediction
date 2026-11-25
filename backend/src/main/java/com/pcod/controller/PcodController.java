package com.pcod.controller;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.pcod.db.PredictionRecord;
import com.pcod.db.PredictionRecordRepository;
import com.pcod.ml.FeatureSpec;
import com.pcod.ml.Metrics;
import com.pcod.ml.ModelStore;
import com.pcod.ml.Preprocessing;
import com.pcod.ml.TrainingService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.domain.PageRequest;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * REST API for PCOD/PCOS prediction.
 *
 * <ul>
 *   <li>GET  /api/health   — liveness check</li>
 *   <li>GET  /api/models   — the trained models with measured test metrics
 *       (4 ensembles + soft-voting ensemble + screening model)</li>
 *   <li>GET  /api/metrics  — the full metrics.json (models, baselines,
 *       feature importances, thresholds, voting metrics)</li>
 *   <li>GET  /api/models/{name}/importance — feature importance for one
 *       tree model (Random Forest / Gradient Boosting / screening)</li>
 *   <li>GET  /api/features — the 41 input features (label, range, training-mean default)</li>
 *   <li>POST /api/predict  — prediction from the deployed model (also saved
 *       to MySQL as a {@code PredictionRecord} when the database is available;
 *       a database failure never breaks the prediction response).
 *       {@code ?mode=screening} uses the 23-feature screening model.</li>
 *   <li>GET  /api/predictions      — prediction history, latest first (?limit=20)</li>
 *   <li>GET  /api/predictions/{id} — one stored prediction</li>
 *   <li>PATCH /api/predictions/{id}/outcome — record a doctor-verified outcome (0/1)</li>
 *   <li>POST /api/retrain  — retrain all models on the dataset CSV and reload
 *       (protected by X-Admin-Key when ADMIN_KEY is set; retraining uses
 *       doctor-verified history rows only, never unverified predictions)</li>
 * </ul>
 */
@RestController
@RequestMapping("/api")
@Tag(name = "PCOD Prediction API", description = "PCOS screening predictions from the trained Java ML models")
public class PcodController {

    private static final Logger log = LoggerFactory.getLogger(PcodController.class);

    private final ModelStore store;

    /**
     * Optional: absent when the database is not configured (for example in
     * unit tests, where the JPA auto-configuration is excluded), and also
     * {@code null} at runtime if the repository bean could not be created.
     * Prediction history is a best-effort feature — it must never stop a
     * prediction from being returned.
     */
    private final PredictionRecordRepository history;

    private final ObjectMapper mapper = new ObjectMapper();

    @Value("${pcod.dataset.path:../data/pcos_dataset.csv}")
    private String datasetPath;

    @Value("${pcod.resources.dir:src/main/resources}")
    private String resourcesDir;

    /**
     * Admin key protecting POST /api/retrain, from the ADMIN_KEY
     * environment variable. When it is empty (not set), retraining is
     * unprotected and a warning is logged at startup — set ADMIN_KEY in
     * any shared deployment.
     */
    @Value("${pcod.admin.key:}")
    private String adminKey;

    public PcodController(ModelStore store, ObjectProvider<PredictionRecordRepository> historyProvider) {
        this.store = store;
        this.history = historyProvider.getIfAvailable();
    }

    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "UP", "service", "PCOD Prediction API");
    }

    @GetMapping("/models")
    @Operation(summary = "Trained models with measured test metrics",
            description = "The 4 ensembles, the soft-voting ensemble and the screening model, "
                    + "each with test accuracy/precision/recall/F1, ROC-AUC, PR-AUC, Brier score, "
                    + "confusion matrix, ROC curve and bootstrap CI.")
    public ResponseEntity<?> models() {
        if (!store.isReady()) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("error", "Models are not loaded yet — train first (see README)."));
        }
        return ResponseEntity.ok(store.getMetrics().models);
    }

    /** The full metrics.json: models, baselines, importances, thresholds, voting metrics. */
    @GetMapping("/metrics")
    @Operation(summary = "Full metrics (models, baselines, importances, thresholds)")
    public ResponseEntity<?> metrics() {
        if (!store.isReady()) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("error", "Models are not loaded yet — train first (see README)."));
        }
        return ResponseEntity.ok(store.getMetrics());
    }

    /**
     * Feature importance for one tree model, highest first:
     * Random Forest, Gradient Boosting, or the screening model. The name
     * accepts the display name ("Random Forest", "Gradient Boosting",
     * "Gradient Boosting (Screening)") or a slug ("random-forest",
     * "gradient-boosting", "gradient-boosting-screening", "screening").
     * Bagging, AdaBoost, the voting ensemble and the baselines expose no
     * per-feature importance, so they answer 404.
     */
    @GetMapping("/models/{name}/importance")
    @Operation(summary = "Feature importance for one tree model")
    public ResponseEntity<?> importance(@PathVariable String name) {
        if (!store.isReady()) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("error", "Models are not loaded yet — train first (see README)."));
        }
        Metrics metrics = store.getMetrics();
        String key = name.trim().toLowerCase();
        Map<String, Double> source;
        String displayName;
        if (key.equals("random forest") || key.equals("random-forest") || key.equals("rf")) {
            source = metrics.featureImportance;
            displayName = "Random Forest";
        } else if (key.equals("gradient boosting") || key.equals("gradient-boosting")
                || key.equals("gbm") || key.equals("gradient-boosting-full")) {
            source = metrics.gbmImportance;
            displayName = "Gradient Boosting";
        } else if (key.equals("gradient boosting (screening)") || key.equals("gradient-boosting-screening")
                || key.equals("screening")) {
            source = metrics.screeningImportance;
            displayName = TrainingService.SCREENING_MODEL;
        } else {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(Map.of("error", "No feature importance for model '" + name + "'. "
                            + "Available: Random Forest, Gradient Boosting, Gradient Boosting (Screening)."));
        }
        if (source == null || source.isEmpty()) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(Map.of("error", "Feature importance for '" + displayName + "' is not available "
                            + "in this metrics file — retrain to regenerate it."));
        }
        List<Map<String, Object>> out = new ArrayList<>();
        source.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .forEach(e -> {
                    Map<String, Object> row = new LinkedHashMap<>();
                    row.put("feature", e.getKey());
                    row.put("importance", e.getValue());
                    out.add(row);
                });
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", displayName);
        body.put("importance", out);
        return ResponseEntity.ok(body);
    }

    @GetMapping("/features")
    public ResponseEntity<?> features() {
        if (!store.isReady()) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("error", "Models are not loaded yet — train first (see README)."));
        }
        List<Map<String, Object>> out = new ArrayList<>();
        for (Preprocessing.FeatureStats f : store.getPreprocessing().features) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("key", f.key);
            m.put("label", f.label);
            m.put("min", f.min);
            m.put("max", f.max);
            m.put("default", f.defaultValue);
            m.put("type", f.type);
            // Screening-mode membership + whether /api/predict requires this field.
            m.put("screening", FeatureSpec.isScreening(f.key));
            m.put("required", FeatureSpec.isRequired(f.key));
            out.add(m);
        }
        return ResponseEntity.ok(out);
    }

    /**
     * Accepts either a flat JSON object with the 41 feature keys, or
     * {@code {"features": {...}}}. A minimum set of 11 non-invasive fields
     * is REQUIRED (see {@link FeatureSpec#REQUIRED_FEATURES}) — a request
     * missing any of them is a 400 listing the missing fields, because a
     * prediction from almost no data would be meaningless and an empty
     * request must never return a confident prediction. Other missing
     * features default to the training mean (mean imputation); the response
     * reports how many of the 41 fields were imputed
     * ({@code imputedCount}). A present-but-non-numeric value is a 400,
     * never a 500. {@code ?mode=screening} uses the 23-feature screening
     * model instead of the deployed full model.
     */
    @PostMapping("/predict")
    @Operation(summary = "Predict PCOS for one patient",
            description = "11 non-invasive fields are required; the other 30 default to training means "
                    + "(reported as imputedCount). mode=screening uses the 23-feature screening model.")
    public ResponseEntity<?> predict(
            @RequestBody(required = false) JsonNode body,
            @RequestParam(defaultValue = "full") @Parameter(description = "full or screening") String mode) {
        if (!store.isReady()) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("error", "Models are not loaded yet — train first (see README)."));
        }
        if (!"full".equalsIgnoreCase(mode) && !"screening".equalsIgnoreCase(mode)) {
            return badRequest("Query parameter 'mode' must be 'full' or 'screening', got: " + mode);
        }
        if (body == null || !body.isObject()) {
            return badRequest("Request body must be a JSON object with the 41 feature keys, "
                    + "or {\"features\": {...}}.");
        }
        JsonNode featuresNode = body.has("features") ? body.get("features") : body;
        if (!featuresNode.isObject()) {
            return badRequest("\"features\" must be a JSON object with the 41 feature keys.");
        }
        List<String> missingRequired = new ArrayList<>();
        for (String key : FeatureSpec.REQUIRED_FEATURES) {
            JsonNode v = featuresNode.get(key);
            if (v == null || v.isNull()) {
                missingRequired.add(key);
            }
        }
        if (!missingRequired.isEmpty()) {
            return badRequest("Missing required fields: " + String.join(", ", missingRequired)
                    + ". These 11 non-invasive fields are required for a meaningful prediction.");
        }
        double[] raw = new double[FeatureSpec.N_FEATURES];
        int imputedCount = 0;
        for (int j = 0; j < FeatureSpec.N_FEATURES; j++) {
            String key = FeatureSpec.FEATURES.get(j).key();
            JsonNode v = featuresNode.get(key);
            if (v == null || v.isNull()) {
                raw[j] = Double.NaN; // missing -> training mean (mean imputation)
                imputedCount++;
            } else if (!v.isNumber() || !Double.isFinite(v.asDouble())) {
                return badRequest("Feature '" + key + "' must be a number, got: " + v);
            } else {
                raw[j] = v.asDouble();
            }
        }
        ModelStore.Prediction p = store.predict(raw, mode);
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("prediction", p.prediction());
        out.put("label", p.label());
        out.put("probability", round(p.probability()));
        out.put("confidence", round(p.confidence()));
        out.put("model", p.model());
        out.put("modelAccuracy", round(p.modelAccuracy()));
        out.put("mode", "screening".equalsIgnoreCase(mode) ? "screening" : "full");
        out.put("threshold", round(p.threshold()));
        out.put("imputedCount", imputedCount);
        out.put("imputedOf", FeatureSpec.N_FEATURES);
        saveHistory(raw, p);
        return ResponseEntity.ok(out);
    }

    /**
     * Prediction history, latest first. {@code ?limit=N} caps the number of
     * records (default 20, max 100). Returns 503 when the database is not
     * available — history is a best-effort feature, not a liveness check.
     */
    @GetMapping("/predictions")
    public ResponseEntity<?> predictions(@RequestParam(defaultValue = "20") int limit) {
        if (history == null) {
            return databaseUnavailable();
        }
        int n = Math.min(Math.max(limit, 1), 100);
        try {
            List<PredictionRecord> records =
                    history.findAllByOrderByCreatedAtDescIdDesc(PageRequest.of(0, n));
            List<Map<String, Object>> out = new ArrayList<>();
            for (PredictionRecord r : records) {
                out.add(toHistoryJson(r));
            }
            return ResponseEntity.ok(out);
        } catch (Exception e) {
            log.warn("Could not read prediction history: {}", e.getMessage());
            return databaseUnavailable();
        }
    }

    /** One stored prediction by its database id, or 404. */
    @GetMapping("/predictions/{id}")
    public ResponseEntity<?> predictionById(@PathVariable long id) {
        if (history == null) {
            return databaseUnavailable();
        }
        try {
            return history.findById(id)
                    .map(r -> ResponseEntity.ok(toHistoryJson(r)))
                    .orElseGet(() -> ResponseEntity.status(HttpStatus.NOT_FOUND)
                            .body(Map.of("error", "No prediction with id " + id)));
        } catch (Exception e) {
            log.warn("Could not read prediction {}: {}", id, e.getMessage());
            return databaseUnavailable();
        }
    }

    /**
     * Records a doctor-verified outcome for a stored prediction:
     * {@code {"verifiedOutcome": 0}} (no PCOS) or {@code {"verifiedOutcome": 1}}
     * (PCOS confirmed). This is the ONLY way prediction-history rows become
     * training data: retraining uses rows whose verified outcome is filled
     * in, and never trains on unverified model predictions. A verified
     * outcome is a clinical fact, not a model output — it must come from a
     * doctor, and it cannot be changed once set through this endpoint
     * (re-PATCHing overwrites it; the latest value is what retraining uses).
     */
    @PatchMapping("/predictions/{id}/outcome")
    @Operation(summary = "Record a doctor-verified outcome for a prediction",
            description = "Body: {\"verifiedOutcome\": 0 or 1}. Only verified rows are ever used for retraining.")
    public ResponseEntity<?> setVerifiedOutcome(@PathVariable long id,
                                                @RequestBody(required = false) JsonNode body) {
        if (history == null) {
            return databaseUnavailable();
        }
        if (body == null || !body.isObject() || !body.has("verifiedOutcome")) {
            return badRequest("Body must be {\"verifiedOutcome\": 0} or {\"verifiedOutcome\": 1}.");
        }
        JsonNode v = body.get("verifiedOutcome");
        if (!v.isInt() || (v.asInt() != 0 && v.asInt() != 1)) {
            return badRequest("verifiedOutcome must be 0 (no PCOS) or 1 (PCOS confirmed).");
        }
        try {
            return history.findById(id)
                    .map(r -> {
                        r.setVerifiedOutcome(v.asInt());
                        r.setVerifiedAt(LocalDateTime.now());
                        history.save(r);
                        log.info("Verified outcome recorded for prediction {}: {}", id, v.asInt());
                        return ResponseEntity.ok(toHistoryJson(r));
                    })
                    .orElseGet(() -> ResponseEntity.status(HttpStatus.NOT_FOUND)
                            .body(Map.of("error", "No prediction with id " + id)));
        } catch (Exception e) {
            log.warn("Could not record verified outcome for prediction {}: {}", id, e.getMessage());
            return databaseUnavailable();
        }
    }

    /**
     * Stores the prediction just made. The features saved are the values the
     * model actually saw — a feature missing from the request is stored as
     * its training mean. Any database failure is logged and swallowed: the
     * prediction response has already been built and must still be returned.
     */
    private void saveHistory(double[] raw, ModelStore.Prediction p) {
        if (history == null) {
            return;
        }
        try {
            Map<String, Object> features = new LinkedHashMap<>();
            for (int j = 0; j < FeatureSpec.N_FEATURES; j++) {
                double v = raw[j];
                if (Double.isNaN(v)) {
                    v = store.getPreprocessing().features.get(j).defaultValue;
                }
                features.put(FeatureSpec.FEATURES.get(j).key(), v);
            }
            PredictionRecord record = new PredictionRecord(
                    mapper.writeValueAsString(features),
                    p.prediction(), p.label(),
                    round(p.probability()), round(p.confidence()), p.model());
            history.save(record);
        } catch (Exception e) {
            log.warn("Could not save prediction to database (prediction still returned): {}",
                    e.getMessage());
        }
    }

    private Map<String, Object> toHistoryJson(PredictionRecord r) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("id", r.getId());
        try {
            out.put("features", mapper.readValue(r.getFeaturesJson(), Map.class));
        } catch (JsonProcessingException e) {
            // Should not happen (we wrote the JSON ourselves); fall back to the raw string.
            out.put("features", r.getFeaturesJson());
        }
        out.put("prediction", r.getPrediction());
        out.put("label", r.getLabel());
        out.put("probability", r.getProbability());
        out.put("confidence", r.getConfidence());
        out.put("model", r.getModel());
        out.put("createdAt", r.getCreatedAt() == null ? null : r.getCreatedAt().toString());
        // Doctor-verified outcome (null until a doctor records one).
        out.put("verifiedOutcome", r.getVerifiedOutcome());
        out.put("verifiedAt", r.getVerifiedAt() == null ? null : r.getVerifiedAt().toString());
        return out;
    }

    private static ResponseEntity<Map<String, String>> databaseUnavailable() {
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                .body(Map.of("error", "Prediction history database is not available — "
                        + "check that MySQL is running and the datasource is configured (see README)."));
    }

    /**
     * Retrains all models on the dataset CSV and reloads them in memory.
     * Protected by an admin key when ADMIN_KEY is set: the request must
     * carry header {@code X-Admin-Key} with the same value, otherwise 401.
     * When ADMIN_KEY is not set the endpoint is unprotected (and a warning
     * is logged) — set it in any shared deployment.
     *
     * <p>Retraining also appends doctor-verified prediction-history rows
     * (verified outcome filled in) to the TRAIN split. Unverified model
     * predictions are NEVER used as training data.</p>
     */
    @PostMapping("/retrain")
    @Operation(summary = "Retrain all models (admin key required when ADMIN_KEY is set)")
    public synchronized ResponseEntity<?> retrain(
            @RequestHeader(value = "X-Admin-Key", required = false) String providedKey) {
        if (adminKey != null && !adminKey.isBlank() && !adminKey.equals(providedKey)) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Missing or invalid X-Admin-Key header. "
                            + "POST /api/retrain requires the admin key (ADMIN_KEY)."));
        }
        if (adminKey == null || adminKey.isBlank()) {
            log.warn("ADMIN_KEY is not set — POST /api/retrain is unprotected. "
                    + "Set the ADMIN_KEY environment variable in any shared deployment.");
        }
        Path csv = Path.of(datasetPath);
        if (!Files.exists(csv)) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(Map.of(
                    "error", "Dataset CSV not found at " + csv.toAbsolutePath()
                            + ". Set pcod.dataset.path (or PCOD_DATASET_PATH) to the dataset location."));
        }
        Path resources = Path.of(resourcesDir);
        try {
            List<TrainingService.Row> verified = verifiedRows();
            TrainingService.TrainingResult result =
                    new TrainingService().train(csv, resources, verified);
            store.reload(resources);
            log.info("Retrained; best model: {}; verified history rows used: {}",
                    result.metrics().bestModel, result.metrics().verifiedRowsUsed);
            return ResponseEntity.ok(result.metrics());
        } catch (Exception e) {
            log.error("Retraining failed", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", "Retraining failed: " + e.getMessage()));
        }
    }

    /**
     * Doctor-verified prediction-history rows as training rows (features +
     * verified outcome). Only rows with a filled verified outcome are
     * returned; rows whose stored features cannot be parsed are skipped
     * (and logged) rather than failing the retrain. Returns an empty list
     * when the database is unavailable.
     */
    private List<TrainingService.Row> verifiedRows() {
        List<TrainingService.Row> out = new ArrayList<>();
        if (history == null) {
            return out;
        }
        try {
            for (PredictionRecord r : history.findByVerifiedOutcomeNotNull()) {
                try {
                    JsonNode features = mapper.readTree(r.getFeaturesJson());
                    double[] x = new double[FeatureSpec.N_FEATURES];
                    for (int j = 0; j < FeatureSpec.N_FEATURES; j++) {
                        JsonNode v = features.get(FeatureSpec.FEATURES.get(j).key());
                        x[j] = (v == null || v.isNull() || !v.isNumber())
                                ? Double.NaN : v.asDouble();
                    }
                    out.add(new TrainingService.Row(x, r.getVerifiedOutcome()));
                } catch (Exception e) {
                    log.warn("Skipping verified history row {} (unparseable features): {}",
                            r.getId(), e.getMessage());
                }
            }
        } catch (Exception e) {
            log.warn("Could not read verified history rows (retraining on CSV only): {}", e.getMessage());
        }
        return out;
    }

    private static ResponseEntity<Map<String, String>> badRequest(String message) {
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(Map.of("error", message));
    }

    private static double round(double v) {
        return Math.round(v * 10000.0) / 10000.0;
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, String>> handleBadInput(IllegalArgumentException e) {
        return badRequest(e.getMessage());
    }

    @ExceptionHandler(IllegalStateException.class)
    public ResponseEntity<Map<String, String>> handleNotReady(IllegalStateException e) {
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                .body(Map.of("error", e.getMessage()));
    }
}
