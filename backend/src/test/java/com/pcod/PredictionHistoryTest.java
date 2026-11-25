package com.pcod;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.io.ClassPathResource;
import org.springframework.http.MediaType;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import java.nio.charset.StandardCharsets;
import java.sql.DriverManager;

import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.is;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.patch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Live MySQL test for prediction history: POST a prediction, then check it
 * appears in GET /api/predictions (latest first) and in
 * GET /api/predictions/{id}.
 *
 * <p>Requires a running MySQL Server 8.x (see README "Database (MySQL)").
 * The connection comes from the same environment variables as the app
 * ({@code MYSQL_URL}, {@code MYSQL_USERNAME}, {@code MYSQL_PASSWORD});
 * the table is created fresh for each test from the production
 * {@code schema.sql} (see setUp()). If MySQL is not reachable the whole
 * class is SKIPPED (not failed), so {@code mvn test} still passes on a
 * machine without MySQL — but a skip here means history was NOT verified
 * against a real database.
 */
@SpringBootTest(properties = {
        // Re-enable the JPA auto-configuration excluded by the test profile.
        "spring.autoconfigure.exclude=",
        // History endpoints require the admin key (fail-closed when unset).
        "pcod.admin.key=test-admin-key",        "spring.datasource.url=${MYSQL_URL:jdbc:mysql://localhost:3306/pcod_db?createDatabaseIfNotExist=true&useSSL=false&serverTimezone=UTC}",
        "spring.datasource.username=${MYSQL_USERNAME:root}",
        "spring.datasource.password=${MYSQL_PASSWORD:}",
        // The context must load even when MySQL is DOWN, so that the
        // requireMysql() assumption inside the test methods can SKIP the
        // tests instead of the context failing first. Three things make that
        // possible: the dialect is pinned (Hibernate then does not need a
        // live connection to determine it), Hikari does not fail fast
        // (initialization-fail-timeout=-1) and no DDL runs at startup
        // (ddl-auto=none, sql.init.mode=never). The table is created by the
        // test itself, from the production schema.sql, after requireMysql()
        // has confirmed the server is reachable (see setUp()).
        "spring.jpa.hibernate.ddl-auto=none",
        "spring.jpa.properties.hibernate.dialect=org.hibernate.dialect.MySQLDialect",
        "spring.datasource.hikari.initialization-fail-timeout=-1",
        "spring.datasource.hikari.maximum-pool-size=1",
        "spring.datasource.hikari.minimum-idle=0",
        // Hibernate still tries one connection at startup (JDBC metadata);
        // when MySQL is down that attempt must fail in 2 s, not in the
        // 30 s default, or the skipped tests stall the whole suite.
        "spring.datasource.hikari.connection-timeout=2000",
        "spring.sql.init.mode=never"
})
@AutoConfigureMockMvc
class PredictionHistoryTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    /**
     * Creates the prediction_records table from the production schema.sql
     * before each test — but only after requireMysql() has confirmed the
     * server is reachable, so on a machine without MySQL this aborts the
     * test as SKIPPED instead of failing it. No DDL runs at context load
     * (see the ddl-auto=none property above).
     */
    @BeforeEach
    void setUp() throws Exception {
        requireMysql();
        String ddl = new String(new ClassPathResource("schema.sql").getInputStream()
                .readAllBytes(), StandardCharsets.UTF_8);
        jdbcTemplate.execute(ddl);
        jdbcTemplate.execute("DELETE FROM prediction_records");
    }

    // Sample patient 2 from data/sample_patients.json (true label 1), clean keys.
    private static final String SAMPLE2 = """
            {"age":33,"weightKg":68.8,"heightCm":165.0,"bmi":25.27089073,"bloodGroup":11,
            "pulseRate":72,"rr":18,"hb":11.8,"cycle":2,"cycleLength":5,"marriageYears":10.0,
            "pregnant":1,"abortions":0,"betaHcg1":494.08,"betaHcg2":494.08,"fsh":5.54,"lh":0.88,
            "fshLh":6.295454545,"hip":40,"waist":36,"waistHipRatio":0.9,"tsh":2.54,"amh":6.63,
            "prl":10.52,"vitD3":49.7,"prg":0.36,"rbs":84.0,"weightGain":0,"hairGrowth":0,
            "skinDarkening":0,"hairLoss":1,"pimples":1,"fastFood":1.0,"regularExercise":0,
            "bpSystolic":120,"bpDiastolic":80,"follicleL":13,"follicleR":15,
            "avgFollicleSizeL":18.0,"avgFollicleSizeR":20.0,"endometrium":10.0}""";

    /**
     * Skips this test (rather than failing it) when no MySQL server is
     * reachable. The assumption is checked inside the test method so that
     * the surefire report shows the test as SKIPPED — visible proof that
     * history was not verified against a real database on that machine.
     * The probe result is cached: a black-holed connection can take
     * seconds to fail, and it must not be re-probed for every test.
     */
    private static Boolean mysqlReachable;

    private static void requireMysql() {
        if (mysqlReachable == null) {
            String url = System.getenv().getOrDefault("MYSQL_URL",
                    "jdbc:mysql://localhost:3306/pcod_db?createDatabaseIfNotExist=true&useSSL=false&serverTimezone=UTC");
            String user = System.getenv().getOrDefault("MYSQL_USERNAME", "root");
            String password = System.getenv().getOrDefault("MYSQL_PASSWORD", "");
            boolean reachable = false;
            try {
                // Fail fast: a black-holed connection must not stall the suite.
                DriverManager.setLoginTimeout(5);
                DriverManager.getConnection(url, user, password).close();
                reachable = true;
            } catch (Exception e) {
                // MySQL is not reachable in this environment — skip, do not fail.
            }
            mysqlReachable = reachable;
        }
        Assumptions.assumeTrue(mysqlReachable,
                "MySQL not reachable — skipping live history test");
    }

    @Test
    void predictIsSavedAndListed() throws Exception {
        requireMysql();
        mockMvc.perform(post("/api/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(SAMPLE2))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.prediction").value(1));

        MvcResult list = mockMvc.perform(get("/api/predictions").param("limit", "5")
                        .header("X-Admin-Key", "test-admin-key"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", hasSize(1)))
                .andExpect(jsonPath("$[0].prediction").value(1))
                .andExpect(jsonPath("$[0].label").value("PCOS Detected"))
                .andExpect(jsonPath("$[0].model").isString())
                .andExpect(jsonPath("$[0].features.age").value(33))
                .andExpect(jsonPath("$[0].features.endometrium").value(10.0))
                .andExpect(jsonPath("$[0].createdAt").isString())
                .andReturn();
        String body = list.getResponse().getContentAsString();
        long id = Long.parseLong(body.replaceAll(".*\"id\"\\s*:\\s*(\\d+).*", "$1"));

        mockMvc.perform(get("/api/predictions/" + id).header("X-Admin-Key", "test-admin-key"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id").value((int) id))
                .andExpect(jsonPath("$.prediction").value(1))
                .andExpect(jsonPath("$.features.age").value(33))
                .andExpect(jsonPath("$.imputedFields").isArray())
                .andExpect(jsonPath("$.imputedFields").isEmpty());

        mockMvc.perform(get("/api/predictions/999999999").header("X-Admin-Key", "test-admin-key"))
                .andExpect(status().isNotFound());
    }

    @Test
    void verifiedOutcomeCanBeRecorded() throws Exception {
        requireMysql();
        mockMvc.perform(post("/api/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(SAMPLE2))
                .andExpect(status().isOk());

        MvcResult list = mockMvc.perform(get("/api/predictions").param("limit", "1")
                        .header("X-Admin-Key", "test-admin-key"))
                .andExpect(status().isOk())
                .andReturn();
        String body = list.getResponse().getContentAsString();
        long id = Long.parseLong(body.replaceAll(".*\"id\"\\s*:\\s*(\\d+).*", "$1"));

        // Invalid value -> 400.
        mockMvc.perform(patch("/api/predictions/" + id + "/outcome").header("X-Admin-Key", "test-admin-key")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"verifiedOutcome\": 2}"))
                .andExpect(status().isBadRequest());

        // Valid value -> 200, stored on the record.
        mockMvc.perform(patch("/api/predictions/" + id + "/outcome").header("X-Admin-Key", "test-admin-key")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"verifiedOutcome\": 1}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.verifiedOutcome").value(1))
                .andExpect(jsonPath("$.verifiedAt").isString());

        mockMvc.perform(get("/api/predictions/" + id).header("X-Admin-Key", "test-admin-key"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.verifiedOutcome").value(1));

        mockMvc.perform(patch("/api/predictions/999999999").header("X-Admin-Key", "test-admin-key")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"verifiedOutcome\": 1}"))
                .andExpect(status().isNotFound());
    }
}
