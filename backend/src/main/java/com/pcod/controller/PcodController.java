package com.pcod.controller;

import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
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
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
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
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

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
 *   <li>GET  /api/predictions      — prediction history, latest first (?limit=20)
 *       — requires the admin key (see below)</li>
 *   <li>GET  /api/predictions/{id} — one stored prediction — requires the admin key</li>
 *   <li>PATCH /api/predictions/{id}/outcome — record a doctor-verified outcome (0/1)
 *       — requires the admin key</li>
 *   <li>POST /api/retrain  — start an asynchronous retrain (202 + job id);
 *       poll GET /api/retrain/status (or /api/retrain/{jobId}) — requires the admin key</li>
 * </ul>
 *
 * <p><b>Admin key (fail-closed).</b> The history and retrain endpoints carry
 * patient health data or change the deployed model, so they require header
 * {@code X-Admin-Key} with the value of the {@code ADMIN_KEY} environment
 * variable. When {@code ADMIN_KEY} is empty or unset, these endpoints are
 * DISABLED and answer 503 — they never silently run unprotected. Set
 * {@code ADMIN_KEY} in any shared deployment (and in {@code .env} for
 * Docker Compose, which now requires it). A missing or wrong key answers
 * 401. {@code POST /api/predict} and the read-only model endpoints stay
 * public — they reveal no patient data.</p>
 */
@RestController
@RequestMapping("/api")
@Tag(name = "PCOD Prediction API", description = "PCOS screening predictions from the trained Java ML models")
public class PcodController {

    private static final Logger log = LoggerFactory.getLogger(PcodController.class);

    /** Imputed-field count at/above which a prediction is flagged as low-trust (see predict()). */
    private static final int HIGH_IMPUTATION = 21;

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
     * Admin key protecting the history and retrain endpoints, from the
     * ADMIN_KEY environment variable. FAIL-CLOSED: when it is empty (not
     * set), those endpoints refuse every request with 503 — they are
     * never unprotected. Set ADMIN_KEY in any shared deployment.
     */
    @Value("${pcod.admin.key:}")
    private String adminKey;

    /** The latest (or currently running) retrain job; replaced atomically on every state change. */
    private final AtomicReference<RetrainJob> lastRetrainJob = new AtomicReference<>();

    /** True while a background retrain is running; a second POST /api/retrain then gets 409. */
    private final AtomicBoolean retrainRunning = new AtomicBoolean(false);

    /** Single-thread executor for background retraining (the HTTP request returns immediately). */
    private final ExecutorService retrainExecutor = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "pcod-retrain");
        t.setDaemon(true);
        return t;
    });

    /** One asynchronous retrain run. Jackson-serialised for GET /api/retrain/status. */
    public record RetrainJob(String jobId, String status, String startedAt, String finishedAt,
                             String message, String bestModel, Double bestModelTestAccuracy,
                             Integer verifiedRowsUsed, String generatedAt) {}

    public PcodController(ModelStore store, ObjectProvider<PredictionRecordRepository> historyProvider) {
        this.store = store;
        this.history = historyProvider.getIfAvailable();
    }

    @PostConstruct
    void warnIfAdminKeyUnset() {
        if (!adminKeyConfigured()) {
            log.error("ADMIN_KEY is not set — POST /api/retrain, GET /api/predictions, "
                    + "GET /api/predictions/{id}, PATCH /api/predictions/{id}/outcome and "
                    + "GET /api/retrain/status are DISABLED (fail-closed) until ADMIN_KEY is set. "
                    + "Set the ADMIN_KEY environment variable and restart.");
        }
    }

    @PreDestroy
    void shutdownRetrainExecutor() {
        retrainExecutor.shutdownNow();
    }

    private boolean adminKeyConfigured() {
        return adminKey != null && !adminKey.isBlank();
    }

    /**
     * Admin-key gate for the history and retrain endpoints. Returns
     * {@code null} when the request is authorised; otherwise the response
     * to return: 503 when no key is configured at all (fail-closed), 401
     * when the key is configured but the header is missing or wrong.
     */
    private ResponseEntity<Map<String, String>> requireAdmin(String providedKey) {
        if (!adminKeyConfigured()) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("error", "ADMIN_KEY is not set — this endpoint is disabled "
                            + "(fail-closed). Set the ADMIN_KEY environment variable and restart."));
        }
        if (!adminKey.equals(providedKey)) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Missing or invalid X-Admin-Key header. "
                            + "This endpoint requires the admin key (ADMIN_KEY)."));
        }
        return null;
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
     * ({@code imputedCount}) and which ones ({@code imputedFields}).
     * A present-but-non-numeric value is a 400, never a 500.
     * {@code ?mode=screening} uses the 23-feature screening model instead
     * of the deployed full model.
     *
     * <p>{@code modelAccuracy} is the DEPLOYED MODEL'S test accuracy on the
     * 108-row held-out test set — it describes the model, not this one
     * prediction (see {@code modelAccuracyNote} and the 95% CI fields).
     * When 21 or more fields were imputed, the response also carries a
     * {@code warning}: the result then rests mostly on training-mean
     * values, not on the patient's own data.</p>
     */
    @PostMapping("/predict")
    @Operation(summary = "Predict PCOS for one patient",
            description = "11 non-invasive fields are required; the other 30 default to training means "
                    + "(reported as imputedCount/imputedFields). mode=screening uses the 23-feature screening model. "
                    + "modelAccuracy is the model's test accuracy (108-row test set), not this prediction's accuracy.")
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
        List<String> imputedFields = new ArrayList<>();
        for (int j = 0; j < FeatureSpec.N_FEATURES; j++) {
            String key = FeatureSpec.FEATURES.get(j).key();
            JsonNode v = featuresNode.get(key);
            if (v == null || v.isNull()) {
                raw[j] = Double.NaN; // missing -> training mean (mean imputation)
                imputedFields.add(key);
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
        // The model's test accuracy describes the model, not this prediction:
        // pair it with its 95% bootstrap CI (test set n = 108) and say so.
        Metrics.ModelMetrics mm = store.getMetrics().byName(p.model());
        if (mm != null) {
            out.put("modelAccuracyCiLow", round(mm.accuracyCiLow));
            out.put("modelAccuracyCiHigh", round(mm.accuracyCiHigh));
            out.put("modelAccuracyNote", "Test accuracy of " + p.model() + " on the 108-row held-out "
                    + "test set (95% CI " + formatPct(mm.accuracyCiLow) + "–" + formatPct(mm.accuracyCiHigh)
                    + "); it describes the model, not this individual prediction.");
        }
        out.put("mode", "screening".equalsIgnoreCase(mode) ? "screening" : "full");
        out.put("threshold", round(p.threshold()));
        out.put("imputedCount", imputedFields.size());
        out.put("imputedOf", FeatureSpec.N_FEATURES);
        out.put("imputedFields", imputedFields);
        if (imputedFields.size() >= HIGH_IMPUTATION) {
            out.put("warning", "More than half of the 41 input fields were missing and were "
                    + "filled with training averages — this result rests mostly on average "
                    + "values, not on the patient's own data. Treat it with extra caution; "
                    + "it is not a diagnosis.");
        }
        saveHistory(raw, p, imputedFields);
        return ResponseEntity.ok(out);
    }

    /**
     * Prediction history, latest first. {@code ?limit=N} caps the number of
     * records (default 20, max 100). Requires the admin key (patient health
     * data). Returns 503 when the database is not available — history is a
     * best-effort feature, not a liveness check.
     */
    @GetMapping("/predictions")
    public ResponseEntity<?> predictions(
            @RequestParam(defaultValue = "20") int limit,
            @RequestHeader(value = "X-Admin-Key", required = false) String providedKey) {
        ResponseEntity<?> authError = requireAdmin(providedKey);
        if (authError != null) {
            return authError;
        }
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

    /** One stored prediction by its database id, or 404. Requires the admin key. */
    @GetMapping("/predictions/{id}")
    public ResponseEntity<?> predictionById(@PathVariable long id,
                                            @RequestHeader(value = "X-Admin-Key", required = false) String providedKey) {
        ResponseEntity<?> authError = requireAdmin(providedKey);
        if (authError != null) {
            return authError;
        }
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
     * doctor. Requires the admin key, because a verified outcome becomes
     * training data: anyone able to set outcomes could poison retraining.
     * Re-PATCHing overwrites the outcome (the latest value is what
     * retraining uses, and {@code verifiedAt} is updated); the endpoint
     * does not reject a second PATCH.
     */
    @PatchMapping("/predictions/{id}/outcome")
    @Operation(summary = "Record a doctor-verified outcome for a prediction",
            description = "Body: {\"verifiedOutcome\": 0 or 1}. Only verified rows are ever used for retraining. "
                    + "Requires the admin key. Re-PATCHing overwrites the outcome (latest value wins).")
    public ResponseEntity<?> setVerifiedOutcome(@PathVariable long id,
                                                @RequestBody(required = false) JsonNode body,
                                                @RequestHeader(value = "X-Admin-Key", required = false) String providedKey) {
        ResponseEntity<?> authError = requireAdmin(providedKey);
        if (authError != null) {
            return authError;
        }
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
     * its training mean — AND the list of which features were imputed is
     * stored alongside ({@code imputedFieldsJson}), so retraining can tell
     * real measurements apart from mean-filled values. Any database failure
     * is logged and swallowed: the prediction response has already been
     * built and must still be returned.
     */
    private void saveHistory(double[] raw, ModelStore.Prediction p, List<String> imputedFields) {
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
            record.setImputedFieldsJson(mapper.writeValueAsString(imputedFields));
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
        } catch (JacksonException e) {
            // Should not happen (we wrote the JSON ourselves); fall back to the raw string.
            out.put("features", r.getFeaturesJson());
        }
        out.put("imputedFields", parseImputedFields(r));
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

    /** Parses the stored imputed-fields JSON; null when the row predates imputed-field tracking. */
    private List<String> parseImputedFields(PredictionRecord r) {
        String json = r.getImputedFieldsJson();
        if (json == null) {
            return null;
        }
        try {
            List<?> list = mapper.readValue(json, List.class);
            List<String> out = new ArrayList<>();
            for (Object o : list) {
                out.add(String.valueOf(o));
            }
            return out;
        } catch (JacksonException e) {
            log.warn("Unparseable imputed-fields JSON on prediction {}: {}", r.getId(), e.getMessage());
            return null;
        }
    }

    private static ResponseEntity<Map<String, String>> databaseUnavailable() {
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                .body(Map.of("error", "Prediction history database is not available — "
                        + "check that MySQL is running and the datasource is configured (see README)."));
    }

    /**
     * Starts an asynchronous retrain and returns 202 immediately with the
     * job id — training takes minutes (dozens of cross-validation fits) and
     * must not block an HTTP request or hit a client timeout. Poll
     * GET /api/retrain/status (or /api/retrain/{jobId}) for the result.
     * A second retrain while one is running is rejected with 409.
     * Requires the admin key (fail-closed when ADMIN_KEY is unset).
     *
     * <p>Retraining also appends doctor-verified prediction-history rows
     * (verified outcome filled in) to the TRAIN split. Unverified model
     * predictions are NEVER used as training data, and verified rows with
     * any imputed (mean-filled) feature are excluded — their stored values
     * are training means, not real measurements (see {@link #verifiedRows()}).</p>
     */
    @PostMapping("/retrain")
    @Operation(summary = "Start an asynchronous retrain (admin key required)",
            description = "Returns 202 with a jobId immediately; poll GET /api/retrain/status for the result.")
    public ResponseEntity<?> retrain(
            @RequestHeader(value = "X-Admin-Key", required = false) String providedKey) {
        ResponseEntity<?> authError = requireAdmin(providedKey);
        if (authError != null) {
            return authError;
        }
        if (!retrainRunning.compareAndSet(false, true)) {
            RetrainJob current = lastRetrainJob.get();
            return ResponseEntity.status(HttpStatus.CONFLICT)
                    .body(Map.of("error", "A retrain is already running (job "
                            + (current == null ? "unknown" : current.jobId())
                            + "). Wait for it to finish — poll GET /api/retrain/status — before starting another."));
        }
        String jobId = UUID.randomUUID().toString();
        RetrainJob job = new RetrainJob(jobId, "RUNNING", LocalDateTime.now().toString(), null,
                "Retraining started in the background — poll GET /api/retrain/status for the result.",
                null, null, null, null);
        lastRetrainJob.set(job);
        retrainExecutor.submit(() -> runRetrain(job));
        log.info("Retrain job {} started", jobId);
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("jobId", jobId);
        body.put("status", "RUNNING");
        body.put("message", job.message());
        body.put("statusUrl", "/api/retrain/status");
        return ResponseEntity.status(HttpStatus.ACCEPTED).body(body);
    }

    /** The latest retrain job (RUNNING / DONE / FAILED), or 404 if none has ever run. Requires the admin key. */
    @GetMapping("/retrain/status")
    @Operation(summary = "Status of the latest retrain job")
    public ResponseEntity<?> retrainStatus(
            @RequestHeader(value = "X-Admin-Key", required = false) String providedKey) {
        ResponseEntity<?> authError = requireAdmin(providedKey);
        if (authError != null) {
            return authError;
        }
        RetrainJob job = lastRetrainJob.get();
        if (job == null) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(Map.of("error", "No retrain job has run yet — POST /api/retrain to start one."));
        }
        return ResponseEntity.ok(job);
    }

    /** One retrain job by id (only the latest job is retained). Requires the admin key. */
    @GetMapping("/retrain/{jobId}")
    @Operation(summary = "One retrain job by id")
    public ResponseEntity<?> retrainJob(@PathVariable String jobId,
                                        @RequestHeader(value = "X-Admin-Key", required = false) String providedKey) {
        ResponseEntity<?> authError = requireAdmin(providedKey);
        if (authError != null) {
            return authError;
        }
        RetrainJob job = lastRetrainJob.get();
        if (job == null || !job.jobId().equals(jobId)) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body(Map.of("error", "No retrain job with id " + jobId
                            + " (only the latest job is retained — see GET /api/retrain/status)."));
        }
        return ResponseEntity.ok(job);
    }

    /** Background retrain: trains, reloads the models atomically, and records the outcome on the job. */
    private void runRetrain(RetrainJob job) {
        try {
            Path csv = Path.of(datasetPath);
            if (!Files.exists(csv)) {
                failJob(job, "Dataset CSV not found at " + csv.toAbsolutePath()
                        + ". Set pcod.dataset.path (or PCOD_DATASET_PATH) to the dataset location.");
                return;
            }
            Path resources = Path.of(resourcesDir);
            List<TrainingService.Row> verified = verifiedRows();
            TrainingService.TrainingResult result =
                    new TrainingService().train(csv, resources, verified);
            store.reload(resources);
            Metrics m = result.metrics();
            lastRetrainJob.set(new RetrainJob(job.jobId(), "DONE", job.startedAt(),
                    LocalDateTime.now().toString(),
                    "Retrained; best model: " + m.bestModel + "; verified history rows used: "
                            + m.verifiedRowsUsed,
                    m.bestModel, m.bestModelTestAccuracy, m.verifiedRowsUsed, m.generatedAt));
            log.info("Retrain job {} done; best model: {}; verified history rows used: {}",
                    job.jobId(), m.bestModel, m.verifiedRowsUsed);
        } catch (Exception e) {
            log.error("Retrain job {} failed", job.jobId(), e);
            failJob(job, "Retraining failed: " + e.getMessage());
        } finally {
            retrainRunning.set(false);
        }
    }

    private void failJob(RetrainJob job, String message) {
        lastRetrainJob.set(new RetrainJob(job.jobId(), "FAILED", job.startedAt(),
                LocalDateTime.now().toString(), message, null, null, null, null));
    }

    /**
     * Doctor-verified prediction-history rows as training rows (features +
     * verified outcome). Only rows with a filled verified outcome are
     * returned, AND only rows whose features are all real measurements:
     * a row with any imputed (mean-filled) feature is EXCLUDED — its stored
     * feature values contain training means, and treating those as real
     * patient measurements would inject invented data into training.
     * Rows written before imputed-field tracking existed have no recorded
     * list, so they are excluded too (their features cannot be proven
     * complete). Rows whose stored features cannot be parsed are skipped
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
                List<String> imputed = parseImputedFields(r);
                if (imputed == null) {
                    log.warn("Skipping verified history row {} for retraining: imputed-fields "
                            + "were not recorded when it was stored (written before tracking), "
                            + "so its features cannot be proven complete.", r.getId());
                    continue;
                }
                if (!imputed.isEmpty()) {
                    log.warn("Skipping verified history row {} for retraining: {} feature(s) "
                            + "were mean-imputed ({}), not real measurements.",
                            r.getId(), imputed.size(), String.join(", ", imputed));
                    continue;
                }
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

    /** Formats a 0–1 fraction as a percentage with one decimal, e.g. 0.8241 -> "82.4%". */
    private static String formatPct(double v) {
        return String.format("%.1f%%", v * 100.0);
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
