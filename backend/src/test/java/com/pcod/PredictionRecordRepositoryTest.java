package com.pcod;

import com.pcod.db.PredictionRecord;
import com.pcod.db.PredictionRecordRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.data.domain.PageRequest;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * JPA-layer tests for prediction history. Runs against H2 (TEST scope
 * only — the runtime database stays MySQL). The table is created by
 * Hibernate from the entity (ddl-auto=create-drop); the columns match
 * schema.sql, which is what the MySQL deployment uses.
 *
 * <p>The test profile (src/test/resources/application.properties) excludes
 * the JPA auto-configuration so the other tests run without a database;
 * this test re-enables it with its own properties and points it at H2.
 */
@DataJpaTest(properties = {
        "spring.autoconfigure.exclude=",
        "spring.datasource.url=jdbc:h2:mem:testdb;DB_CLOSE_DELAY=-1;MODE=MySQL",
        "spring.datasource.driver-class-name=org.h2.Driver",
        "spring.datasource.username=sa",
        "spring.datasource.password=",
        "spring.jpa.hibernate.ddl-auto=create-drop",
        "spring.sql.init.mode=never",
        "spring.jpa.properties.hibernate.dialect=org.hibernate.dialect.H2Dialect"
})
class PredictionRecordRepositoryTest {

    @Autowired
    private PredictionRecordRepository repository;

    private static PredictionRecord record(int prediction, String label) {
        return new PredictionRecord(
                "{\"age\":33,\"bmi\":25.27}", prediction, label, 0.85, 0.85,
                "Soft Voting Ensemble");
    }

    @Test
    void saveAndFindById() {
        PredictionRecord saved = repository.save(record(1, "PCOS Detected"));
        assertTrue(saved.getId() != null);
        assertTrue(saved.getCreatedAt() != null, "@PrePersist must set createdAt");
        assertNull(saved.getVerifiedOutcome(), "verified outcome starts empty");

        PredictionRecord found = repository.findById(saved.getId()).orElseThrow();
        assertEquals(1, found.getPrediction());
        assertEquals("PCOS Detected", found.getLabel());
        assertEquals(0.85, found.getProbability(), 1e-9);
    }

    @Test
    void latestFirstOrdering() {
        repository.save(record(0, "No PCOS Detected"));
        repository.save(record(1, "PCOS Detected"));
        repository.save(record(0, "No PCOS Detected"));
        List<PredictionRecord> latest = repository.findAllByOrderByCreatedAtDescIdDesc(PageRequest.of(0, 2));
        assertEquals(2, latest.size());
        // Same-second inserts: id breaks the tie, so the last saved comes first.
        assertTrue(latest.get(0).getId() > latest.get(1).getId());
    }

    @Test
    void findByVerifiedOutcomeNotNullOnlyReturnsVerified() {
        PredictionRecord a = repository.save(record(1, "PCOS Detected"));
        repository.save(record(0, "No PCOS Detected"));
        a.setVerifiedOutcome(1);
        repository.save(a);

        List<PredictionRecord> verified = repository.findByVerifiedOutcomeNotNull();
        assertEquals(1, verified.size());
        assertEquals(a.getId(), verified.get(0).getId());
        assertEquals(1, verified.get(0).getVerifiedOutcome());
    }
}
