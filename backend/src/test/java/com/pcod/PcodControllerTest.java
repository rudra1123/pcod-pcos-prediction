package com.pcod;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.hamcrest.Matchers.hasSize;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.patch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Smoke tests for the API. Requires the trainer to have run first so that
 * src/main/resources contains metrics.json / preprocessing.json / models/*.ser
 * (the trainer runs before `mvn package` in the README workflow).
 * Runs with pcod.admin.key=test-admin-key so the admin-protected history
 * endpoints can be exercised (see the 401/503 tests below).
 */
@SpringBootTest(properties = "pcod.admin.key=test-admin-key")
@AutoConfigureMockMvc
class PcodControllerTest {

    private static final String ADMIN_KEY = "test-admin-key";

    @Autowired
    private MockMvc mockMvc;

    // Sample patient 2 from data/sample_patients.json (true label 1), clean keys.
    private static final String SAMPLE2 = """
            {"age":33,"weightKg":68.8,"heightCm":165.0,"bmi":25.27089073,"bloodGroup":11,
            "pulseRate":72,"rr":18,"hb":11.8,"cycle":2,"cycleLength":5,"marriageYears":10.0,
            "pregnant":1,"abortions":0,"betaHcg1":494.08,"betaHcg2":494.08,"fsh":5.54,"lh":0.88,
            "fshLh":6.295454545,"hip":40,"waist":36,"waistHipRatio":0.9,"tsh":2.54,"amh":6.63,
            "prl":10.52,"vitD3":49.7,"prg":0.36,"rbs":84.0,"weightGain":0,"hairGrowth":0,
            "skinDarkening":0,"hairLoss":1,"pimples":1,"fastFood":1.0,"regularExercise":0,
            "bpSystolic":120,"bpDiastolic":80,"follicleL":13,"follicleR":15,
            "avgFollicleSizeL":18.0,"avgFollicleSizeR":20.0,"endometrium":10.0}""";

    @Test
    void health() throws Exception {
        mockMvc.perform(get("/api/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("UP"))
                .andExpect(jsonPath("$.service").value("PCOD Prediction API"));
    }

    @Test
    void modelsListsSix() throws Exception {
        // 4 ensembles + soft-voting ensemble + screening model.
        mockMvc.perform(get("/api/models"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(6)))
                .andExpect(jsonPath("$[0].accuracy").isNumber())
                .andExpect(jsonPath("$[0].rocAuc").isNumber())
                .andExpect(jsonPath("$[0].confusionMatrix.tp").isNumber());
    }

    @Test
    void modelsIncludesScreeningModel() throws Exception {
        mockMvc.perform(get("/api/models"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[?(@.mode == 'screening')].name", hasSize(1)))
                .andExpect(jsonPath("$[?(@.mode == 'screening')].nFeatures").value(23));
    }

    @Test
    void metricsEndpointHasBaselinesAndThresholds() throws Exception {
        mockMvc.perform(get("/api/metrics"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.baselines", hasSize(5)))
                .andExpect(jsonPath("$.deployedThreshold.threshold").isNumber())
                .andExpect(jsonPath("$.screeningThreshold.threshold").isNumber())
                .andExpect(jsonPath("$.featureImportance").isMap())
                .andExpect(jsonPath("$.gbmImportance").isMap());
    }

    @Test
    void importanceRandomForest() throws Exception {
        mockMvc.perform(get("/api/models/random-forest/importance"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.model").value("Random Forest"))
                .andExpect(jsonPath("$.importance", hasSize(41)))
                .andExpect(jsonPath("$.importance[0].feature").value("follicleR"));
    }

    @Test
    void importanceScreeningHas23() throws Exception {
        mockMvc.perform(get("/api/models/screening/importance"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.model").value("Gradient Boosting (Screening)"))
                .andExpect(jsonPath("$.importance", hasSize(23)));
    }

    @Test
    void importanceUnknownModelIs404() throws Exception {
        mockMvc.perform(get("/api/models/bagging/importance"))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.error").isString());
    }

    @Test
    void featuresLists41() throws Exception {
        mockMvc.perform(get("/api/features"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(41)))
                .andExpect(jsonPath("$[0].key").value("age"))
                .andExpect(jsonPath("$[0].default").isNumber());
    }

    @Test
    void predictSample2() throws Exception {
        mockMvc.perform(post("/api/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(SAMPLE2))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.prediction").value(1))
                .andExpect(jsonPath("$.probability").isNumber())
                .andExpect(jsonPath("$.model").isString())
                .andExpect(jsonPath("$.modelAccuracy").isNumber())
                .andExpect(jsonPath("$.modelAccuracyNote").isString())
                .andExpect(jsonPath("$.modelAccuracyCiLow").isNumber())
                .andExpect(jsonPath("$.modelAccuracyCiHigh").isNumber())
                .andExpect(jsonPath("$.mode").value("full"))
                .andExpect(jsonPath("$.threshold").isNumber())
                .andExpect(jsonPath("$.imputedCount").value(0))
                .andExpect(jsonPath("$.imputedFields").isArray())
                .andExpect(jsonPath("$.imputedFields").isEmpty());
    }

    @Test
    void predictReportsImputedFieldsAndWarningWhenMostlyImputed() throws Exception {
        // All 11 required fields present; the other 30 are imputed (>= 21).
        mockMvc.perform(post("/api/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"features\": {\"age\": 28, \"bmi\": 19.3, \"cycle\": 2, "
                                + "\"cycleLength\": 5, \"weightGain\": 0, \"hairGrowth\": 0, "
                                + "\"skinDarkening\": 0, \"hairLoss\": 0, \"pimples\": 0, "
                                + "\"fastFood\": 1, \"regularExercise\": 0}}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.imputedCount").value(30))
                .andExpect(jsonPath("$.imputedFields").isArray())
                .andExpect(jsonPath("$.imputedFields.length()").value(30))
                .andExpect(jsonPath("$.imputedFields[0]").value("weightKg"))
                .andExpect(jsonPath("$.warning").isString());
    }

    @Test
    void predictMissingFeaturesDefaultToMeans() throws Exception {
        // All 11 required fields present; the other 30 are imputed.
        mockMvc.perform(post("/api/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"features\": {\"age\": 28, \"bmi\": 19.3, \"cycle\": 2, "
                                + "\"cycleLength\": 5, \"weightGain\": 0, \"hairGrowth\": 0, "
                                + "\"skinDarkening\": 0, \"hairLoss\": 0, \"pimples\": 0, "
                                + "\"fastFood\": 1, \"regularExercise\": 0}}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.prediction").isNumber())
                .andExpect(jsonPath("$.imputedCount").value(30))
                .andExpect(jsonPath("$.imputedOf").value(41));
    }

    @Test
    void predictEmptyBodyIs400ListingMissingRequired() throws Exception {
        mockMvc.perform(post("/api/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").isString())
                .andExpect(jsonPath("$.error", org.hamcrest.Matchers.containsString("age")))
                .andExpect(jsonPath("$.error", org.hamcrest.Matchers.containsString("bmi")));
    }

    @Test
    void predictMissingRequiredListsThem() throws Exception {
        mockMvc.perform(post("/api/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"age\": 33}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error", org.hamcrest.Matchers.containsString("Missing required fields")));
    }

    @Test
    void predictScreeningMode() throws Exception {
        mockMvc.perform(post("/api/predict?mode=screening")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(SAMPLE2))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.mode").value("screening"))
                .andExpect(jsonPath("$.model").value("Gradient Boosting (Screening)"))
                .andExpect(jsonPath("$.threshold").isNumber())
                .andExpect(jsonPath("$.prediction").isNumber());
    }

    @Test
    void predictBadModeIs400() throws Exception {
        mockMvc.perform(post("/api/predict?mode=quantum")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(SAMPLE2))
                .andExpect(status().isBadRequest());
    }

    @Test
    void predictRejectsNonNumericFeature() throws Exception {
        mockMvc.perform(post("/api/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"age\": \"twenty-eight\"}"))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").isString());
    }

    @Test
    void predictRejectsNonObjectBody() throws Exception {
        mockMvc.perform(post("/api/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("[1, 2, 3]"))
                .andExpect(status().isBadRequest());
    }

    @Test
    void predictionsWithoutDatabaseIs503() throws Exception {
        // Test profile excludes the JPA auto-configuration, so no database
        // is configured: history is unavailable, but the API must say so
        // with a 503 instead of failing. (The admin key is supplied, so
        // this exercises the database path, not the auth gate.)
        mockMvc.perform(get("/api/predictions").header("X-Admin-Key", ADMIN_KEY))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.error").isString());
        mockMvc.perform(get("/api/predictions/1").header("X-Admin-Key", ADMIN_KEY))
                .andExpect(status().isServiceUnavailable());
    }

    @Test
    void predictionsWithoutAdminKeyIs401() throws Exception {
        mockMvc.perform(get("/api/predictions"))
                .andExpect(status().isUnauthorized())
                .andExpect(jsonPath("$.error").isString());
        mockMvc.perform(get("/api/predictions/1"))
                .andExpect(status().isUnauthorized());
        mockMvc.perform(patch("/api/predictions/1/outcome")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"verifiedOutcome\": 1}"))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void predictionsWithWrongAdminKeyIs401() throws Exception {
        mockMvc.perform(get("/api/predictions").header("X-Admin-Key", "wrong-key"))
                .andExpect(status().isUnauthorized());
        mockMvc.perform(patch("/api/predictions/1/outcome").header("X-Admin-Key", "wrong-key")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"verifiedOutcome\": 1}"))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void patchOutcomeWithoutDatabaseIs503() throws Exception {
        // Auth passes (correct key) but there is no database: 503, not 500.
        mockMvc.perform(patch("/api/predictions/1/outcome").header("X-Admin-Key", ADMIN_KEY)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"verifiedOutcome\": 1}"))
                .andExpect(status().isServiceUnavailable());
    }
}
