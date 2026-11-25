package com.pcod.db;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.PrePersist;
import jakarta.persistence.Table;

import java.time.LocalDateTime;

/**
 * One stored prediction, in the {@code prediction_records} table (MySQL).
 *
 * The 41 input features are stored as a single JSON object in
 * {@code featuresJson} (values actually used for the prediction — any
 * feature missing from the request is stored as its training mean, the
 * same value the model saw). {@code imputedFieldsJson} records WHICH
 * features were mean-filled (a JSON array of feature keys, "[]" when
 * none were) — retraining must be able to tell real measurements apart
 * from filled-in means, and rows with any imputed field are excluded
 * from retraining data. {@code prediction}, {@code label},
 * {@code probability}, {@code confidence} and {@code model} are the values
 * returned by {@code POST /api/predict}; {@code createdAt} is set when the
 * record is first saved. Rows written before imputed-field tracking
 * existed have {@code imputedFieldsJson == null} and are treated as
 * unverifiable (excluded from retraining).
 */
@Entity
@Table(name = "prediction_records")
public class PredictionRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /** The 41 input features as a JSON object (clean snake_case keys). */
    @Column(name = "features_json", columnDefinition = "TEXT", nullable = false)
    private String featuresJson;

    /**
     * Feature keys that were mean-imputed for this prediction, as a JSON
     * array (e.g. ["amh","tsh"] or "[]"). Null for rows stored before
     * imputed-field tracking existed. Retraining excludes any verified row
     * with a non-empty list (and any row with a null list).
     */
    @Column(name = "imputed_fields_json", columnDefinition = "TEXT")
    private String imputedFieldsJson;

    /** 0 = no PCOS detected, 1 = PCOS detected. */
    @Column(nullable = false)
    private int prediction;

    @Column(nullable = false, length = 32)
    private String label;

    @Column(nullable = false)
    private double probability;

    @Column(nullable = false)
    private double confidence;

    /** Name of the model that made the prediction (e.g. "Soft Voting Ensemble"). */
    @Column(nullable = false, length = 64)
    private String model;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    /**
     * Doctor-verified outcome: null until a doctor records one
     * (PATCH /api/predictions/{id}/outcome), then 0 (no PCOS) or 1 (PCOS
     * confirmed). Only rows with a filled verified outcome are ever used
     * as training data at retraining time — the model's own prediction is
     * never treated as ground truth.
     */
    @Column(name = "verified_outcome")
    private Integer verifiedOutcome;

    @Column(name = "verified_at")
    private LocalDateTime verifiedAt;

    protected PredictionRecord() {
        // JPA only.
    }

    public PredictionRecord(String featuresJson, int prediction, String label,
                            double probability, double confidence, String model) {
        this.featuresJson = featuresJson;
        this.prediction = prediction;
        this.label = label;
        this.probability = probability;
        this.confidence = confidence;
        this.model = model;
    }

    @PrePersist
    void onCreate() {
        if (createdAt == null) {
            createdAt = LocalDateTime.now();
        }
    }

    public Long getId() {
        return id;
    }

    public String getFeaturesJson() {
        return featuresJson;
    }

    public String getImputedFieldsJson() {
        return imputedFieldsJson;
    }

    public void setImputedFieldsJson(String imputedFieldsJson) {
        this.imputedFieldsJson = imputedFieldsJson;
    }

    public int getPrediction() {
        return prediction;
    }

    public String getLabel() {
        return label;
    }

    public double getProbability() {
        return probability;
    }

    public double getConfidence() {
        return confidence;
    }

    public String getModel() {
        return model;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public Integer getVerifiedOutcome() {
        return verifiedOutcome;
    }

    public void setVerifiedOutcome(Integer verifiedOutcome) {
        this.verifiedOutcome = verifiedOutcome;
    }

    public LocalDateTime getVerifiedAt() {
        return verifiedAt;
    }

    public void setVerifiedAt(LocalDateTime verifiedAt) {
        this.verifiedAt = verifiedAt;
    }
}
