package com.pcod.trainer;

import com.pcod.ml.TrainingService;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Standalone trainer. Run from the backend directory:
 * <pre>
 *   mvn exec:java -Dexec.mainClass=com.pcod.trainer.ModelTrainer
 * </pre>
 * Reads {@code ../data/pcos_dataset.csv} (or the path in arg 0), trains all
 * four models and writes {@code metrics.json}, {@code preprocessing.json} and
 * {@code models/*.ser} into {@code src/main/resources} (or the path in arg 1).
 * The Spring Boot app loads those saved models at startup; it never retrains
 * on boot.
 */
public class ModelTrainer {

    public static void main(String[] args) throws Exception {
        Path csv = args.length > 0 ? Path.of(args[0]) : findDataset();
        Path resources = args.length > 1 ? Path.of(args[1]) : findResourcesDir();
        System.out.println("Training from: " + csv.toAbsolutePath());
        System.out.println("Writing artefacts to: " + resources.toAbsolutePath());
        new TrainingService().train(csv, resources);
        System.out.println("Training complete.");
    }

    static Path findDataset() {
        List<Path> candidates = List.of(
                Path.of("../data/pcos_dataset.csv"),
                Path.of("data/pcos_dataset.csv"),
                Path.of("../../data/pcos_dataset.csv"),
                Path.of("../pcod-working-project/data/pcos_dataset.csv"));
        for (Path p : candidates) {
            if (Files.exists(p)) {
                return p;
            }
        }
        return Path.of("../data/pcos_dataset.csv"); // default; train() will report the real error
    }

    static Path findResourcesDir() {
        List<Path> candidates = List.of(
                Path.of("src/main/resources"),
                Path.of("backend/src/main/resources"));
        for (Path p : candidates) {
            if (Files.isDirectory(p)) {
                return p;
            }
        }
        return Path.of("src/main/resources");
    }
}
