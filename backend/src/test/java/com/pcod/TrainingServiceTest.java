package com.pcod;

import com.pcod.ml.FeatureSpec;
import com.pcod.ml.TrainingService;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the CSV parser and the per-fold feature ranking.
 * No Spring context, no trained models: synthetic rows with known values,
 * plus the real dataset header (located like ModelTrainer does).
 */
class TrainingServiceTest {

    private static final TrainingService SERVICE = new TrainingService();

    // ------------------------------------------------------------ CSV parser

    @Test
    void splitCsvLineHandlesQuotedCommas() {
        assertEquals(List.of("a", "b,c", "d"), TrainingService.splitCsvLine("a,\"b,c\",d"));
        assertEquals(List.of("a", "b\"c", "d"), TrainingService.splitCsvLine("a,\"b\"\"c\",d"));
        assertEquals(List.of("a", "", "c"), TrainingService.splitCsvLine("a,,c"));
        assertEquals(List.of("a", "b", ""), TrainingService.splitCsvLine("a,b,"));
        assertEquals(List.of("plain"), TrainingService.splitCsvLine("plain"));
        assertEquals(List.of("", ""), TrainingService.splitCsvLine(","));
    }

    /** Locates the dataset CSV the same way ModelTrainer does (working directory differs per runner). */
    private static Path findDataset() {
        List<Path> candidates = List.of(
                Path.of("../data/pcos_dataset.csv"),
                Path.of("data/pcos_dataset.csv"),
                Path.of("workspace/final-year-project-cognitive-analytics/pcod-working-project/data/pcos_dataset.csv"),
                Path.of("../pcod-working-project/data/pcos_dataset.csv"));
        for (Path p : candidates) {
            if (Files.exists(p)) {
                return p;
            }
        }
        return null;
    }

    @Test
    void loadCsvHandlesQuotedCommasAndResetsMalformedCount() throws IOException {
        Path dataset = findDataset();
        org.junit.jupiter.api.Assumptions.assumeTrue(dataset != null, "dataset CSV not found from " + Path.of("").toAbsolutePath());
        String header = Files.readAllLines(dataset).get(0);
        List<String> headers = TrainingService.splitCsvLine(header);
        int patientFileCol = headers.indexOf("Patient File No.");
        int amhCol = headers.indexOf("AMH(ng/mL)");
        int targetCol = headers.indexOf("PCOS (Y/N)");
        assertTrue(patientFileCol >= 0 && amhCol >= 0 && targetCol >= 0, "expected columns in header");

        // Two synthetic rows: row 1 carries a quoted comma inside the
        // (dropped) Patient File No. column — a naive split(",") would
        // shift every column after it and break the row. Row 2 carries one
        // malformed numeric cell ("a" in AMH).
        List<String> row1 = new ArrayList<>();
        List<String> row2 = new ArrayList<>();
        for (int i = 0; i < headers.size(); i++) {
            row1.add("1");
            row2.add("1");
        }
        row1.set(patientFileCol, "\"FILE,123\"");
        row1.set(targetCol, "0");
        row2.set(amhCol, "a");
        row2.set(targetCol, "1");
        Path tmp = Files.createTempFile("pcos-quoted", ".csv");
        Files.writeString(tmp, header + "\n" + String.join(",", row1) + "\n" + String.join(",", row2) + "\n");

        List<TrainingService.Row> rows = SERVICE.loadCsv(tmp);
        assertEquals(2, rows.size(), "both rows must parse");
        assertEquals(0, rows.get(0).y(), "row 1 target must survive the quoted comma");
        assertEquals(1, rows.get(1).y());
        assertEquals(1, SERVICE.malformedCells(), "exactly one malformed cell (AMH 'a')");
        assertTrue(Double.isNaN(rows.get(1).x()[FeatureSpec.indexOf("amh")]),
                "malformed AMH must become missing (NaN), not an invented value");

        // Loading again must NOT accumulate: the count is per load, not global.
        SERVICE.loadCsv(tmp);
        assertEquals(1, SERVICE.malformedCells(), "malformed count must reset per loadCsv call");
        Files.deleteIfExists(tmp);
    }

    // ------------------------------------------------- per-fold feature ranking

    /** 40 rows where feature 0 (age) perfectly separates the classes; the rest is noise. */
    private static List<TrainingService.Row> signalRows() {
        List<TrainingService.Row> rows = new ArrayList<>();
        Random rng = new Random(7);
        for (int i = 0; i < 40; i++) {
            int y = i % 2;
            double[] x = new double[FeatureSpec.N_FEATURES];
            for (int j = 0; j < FeatureSpec.N_FEATURES; j++) {
                x[j] = rng.nextDouble();
            }
            x[0] = y == 1 ? 100.0 : 0.0;
            rows.add(new TrainingService.Row(x, y));
        }
        return rows;
    }

    @Test
    void foldRankingsArePerFoldPermutationsWithSignalFirst() {
        List<List<Integer>> rankings = SERVICE.foldRankings(signalRows());
        assertEquals(5, rankings.size(), "one ranking per CV fold");
        Set<Integer> all = new HashSet<>();
        for (int j = 0; j < FeatureSpec.N_FEATURES; j++) {
            all.add(j);
        }
        for (List<Integer> ranking : rankings) {
            assertEquals(FeatureSpec.N_FEATURES, ranking.size());
            assertEquals(all, new HashSet<>(ranking), "each ranking must be a permutation of all features");
            assertEquals(0, ranking.get(0),
                    "feature 0 carries the whole signal, so it must rank first in every fold's ranking");
        }
    }

    @Test
    void cvAccuracyTopKUsesOnlyTopFeatures() {
        List<TrainingService.Row> rows = signalRows();
        List<List<Integer>> rankings = SERVICE.foldRankings(rows);
        TrainingService.ModelConfig cfg = new TrainingService.ModelConfig("Random Forest",
                Map.of("ntrees", 50, "mtry", 0, "maxDepth", 5, "nodeSize", 1, "subsample", 1.0),
                TrainingService.allFeatures(), true);
        double cvTop1 = SERVICE.cvAccuracyTopK(cfg, rows, 1, rankings);
        assertTrue(cvTop1 > 0.9,
                "top-1 (feature 0 alone) must classify the perfectly separable data, was " + cvTop1);
        double cvAll = SERVICE.cvAccuracy(cfg, rows);
        assertTrue(cvAll > 0.9, "all-41 CV accuracy on separable data, was " + cvAll);
    }
}
