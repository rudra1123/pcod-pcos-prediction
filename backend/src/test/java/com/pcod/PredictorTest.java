package com.pcod;

import com.pcod.ml.ModelStore;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
