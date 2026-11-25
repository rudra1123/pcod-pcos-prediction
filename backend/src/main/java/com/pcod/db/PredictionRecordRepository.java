package com.pcod.db;

import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

/**
 * Data access for {@link PredictionRecord}. History is always read
 * latest-first (newest {@code createdAt} first, {@code id} breaks ties
 * for records created within the same second).
 */
public interface PredictionRecordRepository extends JpaRepository<PredictionRecord, Long> {

    List<PredictionRecord> findAllByOrderByCreatedAtDescIdDesc(Pageable pageable);

    /** Rows with a doctor-verified outcome — the only history rows retraining may use. */
    List<PredictionRecord> findByVerifiedOutcomeNotNull();
}
