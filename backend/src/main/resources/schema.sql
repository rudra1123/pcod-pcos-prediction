-- Prediction history table for the PCOD Prediction API.
-- Created automatically at startup (spring.sql.init, see application.properties).
-- Hibernate does NOT manage this schema (ddl-auto=none); this file is the
-- single source of truth for the table. Idempotent: safe to re-run.
-- The runtime database is MySQL 8.x. (The JPA repository test uses H2,
-- which accepts the same DDL.) TIMESTAMP (not DATETIME): H2 2.3, bundled
-- with Spring Boot 4.1, no longer accepts DATETIME, while MySQL accepts
-- TIMESTAMP for both columns.
CREATE TABLE IF NOT EXISTS prediction_records (
    id               BIGINT       NOT NULL AUTO_INCREMENT,
    features_json    TEXT         NOT NULL,
    imputed_fields_json TEXT      NULL,
    prediction       INT          NOT NULL,
    label            VARCHAR(32)  NOT NULL,
    probability      DOUBLE       NOT NULL,
    confidence       DOUBLE       NOT NULL,
    model            VARCHAR(64)  NOT NULL,
    model_version    VARCHAR(16)  NULL,
    patient_name     VARCHAR(100) NULL,
    patient_id       VARCHAR(64)  NULL,
    created_at       TIMESTAMP    NOT NULL,
    verified_outcome INT          NULL,
    verified_at      TIMESTAMP    NULL,
    PRIMARY KEY (id),
    INDEX idx_prediction_records_created_at (created_at)
);
-- Existing deployments (table created before imputed_fields_json existed):
--   ALTER TABLE prediction_records ADD COLUMN imputed_fields_json TEXT NULL;
-- Existing deployments (table created before patient_name / patient_id existed):
--   ALTER TABLE prediction_records ADD COLUMN patient_name VARCHAR(100) NULL, ADD COLUMN patient_id VARCHAR(64) NULL;
-- Existing deployments (table created before model_version existed):
--   ALTER TABLE prediction_records ADD COLUMN model_version VARCHAR(16) NULL;
