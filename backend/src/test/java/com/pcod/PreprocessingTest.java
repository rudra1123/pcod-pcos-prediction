package com.pcod;

import com.pcod.ml.FeatureSpec;
import com.pcod.ml.Preprocessing;
import com.pcod.ml.TrainingService;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the preprocessing pipeline (median imputation -> 1st/99th
 * percentile clipping -> standardisation), fitted on TRAIN rows only.
 * No Spring context, no dataset file: synthetic rows with known values.
 */
class PreprocessingTest {

    private static final TrainingService SERVICE = new TrainingService();

    /** 10 synthetic rows; feature 0 (age) has values 20..29, feature 1 (weightKg) has one NaN. */
    private static List<TrainingService.Row> rows() {
        List<TrainingService.Row> rows = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            double[] x = new double[FeatureSpec.N_FEATURES];
            for (int j = 0; j < FeatureSpec.N_FEATURES; j++) {
                x[j] = 10.0 + j;
            }
            x[0] = 20 + i;                    // age 20..29, mean 24.5
            x[1] = (i == 4) ? Double.NaN : 50 + i; // weightKg: one missing (mean of the other 9 = 54.444...)
            x[2] = 1000.0;                    // heightCm: constant -> std forced to 1.0
            rows.add(new TrainingService.Row(x, i % 2));
        }
        return rows;
    }

    @Test
    void missingValueIsImputedWithTrainingMedian() {
        List<TrainingService.Row> train = rows();
        Preprocessing prep = SERVICE.fitPreprocessing(train, train, true);
        // weightKg training median over the 9 present values
        // [50,51,52,53,55,56,57,58,59] (one missing): the 5th value, 55.
        // The median, not the mean (54.56): a few extreme outliers would
        // otherwise drag the fill value far from the typical value.
        assertEquals(55.0, prep.features.get(1).defaultValue, 1e-9);
        // Transforming NaN must equal transforming the median (same z-score).
        assertEquals(prep.transform(1, 55.0), prep.transform(1, Double.NaN), 1e-12);
    }

    @Test
    void standardisationCentresTrainingData() {
        List<TrainingService.Row> train = rows();
        Preprocessing prep = SERVICE.fitPreprocessing(train, train, true);
        // Mean of the standardised age values over train must be ~0.
        double sum = 0;
        for (TrainingService.Row r : train) {
            sum += prep.transform(0, r.x()[0]);
        }
        assertEquals(0.0, sum / train.size(), 1e-9);
    }

    @Test
    void extremeValueIsClippedTo99thPercentile() {
        List<TrainingService.Row> train = rows();
        Preprocessing prep = SERVICE.fitPreprocessing(train, train, true);
        double clipHigh = prep.features.get(0).clipHigh;
        assertTrue(clipHigh < 29.0, "99th percentile of 20..29 must be below the max");
        assertEquals(prep.transform(0, clipHigh), prep.transform(0, 10_000.0), 1e-12,
                "a huge value must standardise exactly like the clip bound");
        assertEquals(prep.transform(0, clipHigh), prep.transform(0, 29.0), 1e-12,
                "the training max (29) is above the 99th percentile, so it is clipped too");
    }

    @Test
    void constantFeatureGetsStdOne() {
        List<TrainingService.Row> train = rows();
        Preprocessing prep = SERVICE.fitPreprocessing(train, train, true);
        assertEquals(1.0, prep.features.get(2).std, 1e-12);
        assertEquals(0.0, prep.transform(2, 1000.0), 1e-12);
    }

    @Test
    void transformVectorMatchesPerFeatureTransform() {
        List<TrainingService.Row> train = rows();
        Preprocessing prep = SERVICE.fitPreprocessing(train, train, true);
        double[] raw = rows().get(0).x();
        double[] z = prep.transform(raw);
        assertEquals(FeatureSpec.N_FEATURES, z.length);
        for (int j = 0; j < FeatureSpec.N_FEATURES; j++) {
            assertEquals(prep.transform(j, raw[j]), z[j], 1e-12);
        }
    }
}
