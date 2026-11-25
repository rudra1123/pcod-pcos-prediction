-- Prediction history table for the PCOD Prediction API.
-- Created automatically at startup (spring.sql.init, see application.properties).
-- Hibernate does NOT manage this schema (ddl-auto=none); this file is the
-- single source of truth for the table. Idempotent: safe to re-run.
-- The runtime database is MySQL 8.x. (The JPA repository test uses H2,
-- which accepts the same DDL.)
CREATE TABLE IF NOT EXISTS prediction_records (
    id               BIGINT       NOT NULL AUTO_INCREMENT,
    features_json    TEXT         NOT NULL,
    prediction       INT          NOT NULL,
    label            VARCHAR(32)  NOT NULL,
    probability      DOUBLE       NOT NULL,
    confidence       DOUBLE       NOT NULL,
    model            VARCHAR(64)  NOT NULL,
    created_at       DATETIME     NOT NULL,
    verified_outcome INT          NULL,
    verified_at      DATETIME     NULL,
    PRIMARY KEY (id),
    INDEX idx_prediction_records_created_at (created_at)
);
