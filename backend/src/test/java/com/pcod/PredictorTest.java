package com.pcod;

import com.pcod.ml.ModelStore;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Predictor tests against the real trained artefacts (metrics.json +
 * models/*.ser produced by the trainer). Sample patients 1 and 2 are real
 * rows from data/sample_patients.json (true labels 0 and 1).
 */
@SpringBootTest
class PredictorTest {

    @Autowired
    private ModelStore store;

    // Sample patient 2 (true label 1), clean keys.
    private static final double[] SAMPLE2 = {
            33, 68.8, 165.0, 25.27089073, 11, 72, 18, 11.8, 2, 5, 10.0, 1, 0,
            494.08, 494.08, 5.54, 0.88, 6.295454545, 40, 36, 0.9, 2.54, 6.63,
            10.52, 49.7, 0.36, 84.0, 0, 0, 0, 1, 1, 1.0, 0, 120, 80, 13, 15,
            18.0, 20.0, 10.0};

    // Sample patient 1 (true label 0), clean keys.
    private static final double[] SAMPLE1 = {
            28, 44.6, 152.0, 19.3, 15, 78, 22, 10.48, 2, 5, 7.0, 0, 0,
            1.99, 1.99, 7.95, 3.68, 2.160326087, 36, 30, 0.8333333333, 0.68, 2.07,
            45.16, 17.1, 0.57, 92.0, 0, 0, 0, 0, 0, 1.0, 0, 110, 80, 3, 3,
            18.0, 18.0, 8.5};

    // Dataset row 9 (true label 0): screening P(PCOS) ~= 0.258, prediction 1.
    // (Found by scanning all 541 rows through the screening model;
    // re-scanned for the Smile 6.3.0 re-train, 2026-11-26.)
    private static final double[] SCREENING_BUG_CASE = {
            34.0, 64.0, 158.0, 25.63691716, 15.0, 73.0, 20.0, 11.8, 2.0, 6.0, 12.0, 1.0, 1.0,
            289.36, 180.3, 6.98, 1.83, 3.81420765, 42.0, 34.0, 0.8095238095, 2.96, 1.4, 24.89,
            56.9, 0.4, 110.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 120.0, 80.0, 3.0, 5.0,
            12.0, 12.0, 10.0};

    @Test
    void fullModelPredictsSamplePatientsCorrectly() {
        assertTrue(store.isReady(), "models must be loaded");
        ModelStore.Prediction p2 = store.predict(SAMPLE2);
        assertEquals(1, p2.prediction(), "sample 2 is a PCOS case");
        assertTrue(p2.probability() > p2.threshold(),
                "sample 2 probability must clear the tuned threshold");
        ModelStore.Prediction p1 = store.predict(SAMPLE1);
        assertEquals(0, p1.prediction(), "sample 1 is not a PCOS case");
        assertTrue(p1.probability() < p1.threshold(),
                "sample 1 probability must stay below the tuned threshold");
    }

    @Test
    void screeningModelUsesOnlyNonInvasiveFeatures() {
        ModelStore.Prediction p = store.predict(SAMPLE2, "screening");
        assertEquals("Gradient Boosting (Screening)", p.model());
        assertTrue(p.threshold() > 0 && p.threshold() < 1, "tuned threshold must be a probability");
        assertTrue(p.probability() >= 0 && p.probability() <= 1);
        // Lab/ultrasound values must not change a screening prediction:
        // zero them all out and the probability must be identical.
        double[] stripped = SAMPLE2.clone();
        int[] labOrUltrasound = {4, 7, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40};
        for (int j : labOrUltrasound) {
            stripped[j] = Double.NaN;
        }
        ModelStore.Prediction pStripped = store.predict(stripped, "screening");
        assertEquals(p.probability(), pStripped.probability(), 1e-12,
                "screening must ignore lab/ultrasound features entirely");
    }

    @Test
    void thresholdsAreValidProbabilities() {
        assertTrue(store.getMetrics().deployedThreshold.threshold > 0
                        && store.getMetrics().deployedThreshold.threshold < 1);
        assertTrue(store.getMetrics().screeningThreshold.threshold > 0
                        && store.getMetrics().screeningThreshold.threshold < 1);
    }

    @Test
    void screeningConfidenceIsProbabilityOfPredictedClass() {
        // Dataset row 157 (true label 0): screening P(PCOS) ~= 0.381 — above
        // the 0.30 threshold, so the prediction is PCOS, but below 0.5.
        // Confidence must be 0.381 (the PREDICTED class's probability), not
        // max(p, 1-p) = 0.619, which is the No-PCOS side's probability.
        // The old formula got this wrong for the 38 screening PCOS
        // predictions in the dataset whose probability sits below 0.5.
        ModelStore.Prediction p = store.predict(SCREENING_BUG_CASE, "screening");
        assertEquals(1, p.prediction(), "p ~= 0.381 clears the 0.30 screening threshold");
        assertTrue(p.probability() > p.threshold() && p.probability() < 0.5,
                "this case must sit between threshold and 0.5, was " + p.probability());
        assertEquals(p.probability(), p.confidence(), 1e-9,
                "confidence must be P(PCOS) when the prediction is PCOS, even below 0.5");
    }

    @Test
    void fullModeConfidenceIsProbabilityOfPredictedClass() {
        // Sample 1 (true 0): p ~= 0.099, prediction 0 -> confidence 1 - p.
        ModelStore.Prediction p1 = store.predict(SAMPLE1);
        assertEquals(0, p1.prediction());
        assertEquals(1.0 - p1.probability(), p1.confidence(), 1e-9,
                "confidence must be P(No PCOS) when the prediction is No PCOS");
        // Sample 2 (true 1): p ~= 0.857, prediction 1 -> confidence p.
        ModelStore.Prediction p2 = store.predict(SAMPLE2);
        assertEquals(1, p2.prediction());
        assertEquals(p2.probability(), p2.confidence(), 1e-9,
                "confidence must be P(PCOS) when the prediction is PCOS");
    }

    @Test
    void reloadFromSameDirectoryKeepsConsistentPredictions() throws Exception {
        Path resourcesDir = Path.of(getClass().getResource("/metrics.json").toURI()).getParent();
        ModelStore.Prediction before = store.predict(SAMPLE2);
        store.reload(resourcesDir);
        ModelStore.Prediction after = store.predict(SAMPLE2);
        assertTrue(store.isReady());
        assertEquals(before.prediction(), after.prediction());
        assertEquals(before.probability(), after.probability(), 1e-12);
        assertEquals(before.confidence(), after.confidence(), 1e-12);
        assertEquals(before.model(), after.model());
        assertEquals(before.threshold(), after.threshold(), 1e-12);
    }

    @Test
    void concurrentPredictDuringReloadNeverSeesMixedState() throws Exception {
        Path resourcesDir = Path.of(getClass().getResource("/metrics.json").toURI()).getParent();
        java.util.concurrent.atomic.AtomicBoolean failed = new java.util.concurrent.atomic.AtomicBoolean(false);
        Thread reader = new Thread(() -> {
            try {
                for (int i = 0; i < 200; i++) {
                    ModelStore.Prediction p = store.predict(SAMPLE2);
                    // Invariant that a mixed snapshot would break: confidence
                    // is always the probability of the predicted class.
                    double expected = p.prediction() == 1 ? p.probability() : 1.0 - p.probability();
                    if (Math.abs(p.confidence() - expected) > 1e-9) {
                        failed.set(true);
                    }
                    if (p.prediction() != 0 && p.prediction() != 1) {
                        failed.set(true);
                    }
                }
            } catch (Exception e) {
                failed.set(true);
            }
        });
        reader.start();
        for (int i = 0; i < 10; i++) {
            store.reload(resourcesDir);
        }
        reader.join();
        assertTrue(!failed.get(), "concurrent predict/reload must never observe a mixed snapshot");
    }

    @Test
    void missingValuesDoNotCrashPrediction() {
        double[] allMissing = new double[41];
        for (int j = 0; j < allMissing.length; j++) {
            allMissing[j] = Double.NaN;
        }
        ModelStore.Prediction p = store.predict(allMissing);
        assertTrue(p.probability() >= 0 && p.probability() <= 1);
        assertTrue(p.prediction() == 0 || p.prediction() == 1);
    }
}
