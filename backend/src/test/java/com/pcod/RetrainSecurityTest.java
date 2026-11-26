package com.pcod;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * POST /api/retrain is protected by the X-Admin-Key header when ADMIN_KEY
 * is set, and runs asynchronously (202 + job id; poll GET /api/retrain/status).
 * This test runs with pcod.admin.key=test-admin-key and a dataset path that
 * does not exist, so the background job fails fast on "dataset not found"
 * without touching any real artefacts (the success path is NOT tested here
 * — it would run the full retrain).
 */
@SpringBootTest(properties = {
        "pcod.admin.key=test-admin-key",
        "pcod.dataset.path=/nonexistent/pcos_dataset.csv",
        "pcod.resources.dir=/tmp/pcod-test-retrain-resources"
})
@AutoConfigureMockMvc
class RetrainSecurityTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    void retrainWithoutKeyIs401() throws Exception {
        mockMvc.perform(post("/api/retrain"))
                .andExpect(status().isUnauthorized())
                .andExpect(jsonPath("$.error").isString());
    }

    @Test
    void retrainWithWrongKeyIs401() throws Exception {
        mockMvc.perform(post("/api/retrain").header("X-Admin-Key", "wrong-key"))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void retrainWithKeyIs202WithJobId() throws Exception {
        mockMvc.perform(post("/api/retrain").header("X-Admin-Key", "test-admin-key"))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.jobId").isString())
                .andExpect(jsonPath("$.status").value("RUNNING"))
                .andExpect(jsonPath("$.statusUrl").value("/api/retrain/status"));
    }

    @Test
    void retrainStatusRequiresKeyAndReportsFailedJob() throws Exception {
        // Status without the key is refused even though a job exists.
        mockMvc.perform(get("/api/retrain/status"))
                .andExpect(status().isUnauthorized());

        MvcResult started = mockMvc.perform(post("/api/retrain").header("X-Admin-Key", "test-admin-key"))
                .andExpect(status().isAccepted())
                .andReturn();
        String jobId = started.getResponse().getContentAsString()
                .replaceAll(".*\"jobId\"\\s*:\\s*\"([^\"]+)\".*", "$1");

        // The background job fails fast (dataset path does not exist);
        // poll until it reaches a terminal state.
        String status = "";
        String body = "";
        for (int i = 0; i < 25 && !"FAILED".equals(status) && !"DONE".equals(status); i++) {
            Thread.sleep(200);
            MvcResult r = mockMvc.perform(get("/api/retrain/status").header("X-Admin-Key", "test-admin-key"))
                    .andExpect(status().isOk())
                    .andReturn();
            body = r.getResponse().getContentAsString();
            status = body.replaceAll(".*\"status\"\\s*:\\s*\"([^\"]+)\".*", "$1");
        }
        assertEquals("FAILED", status, "job must fail fast on the missing dataset; body was: " + body);
        assertTrue(body.contains(jobId), "status must report the job that was started");
        assertTrue(body.contains("Dataset CSV not found"), "failure message must name the missing dataset");

        // The same job is also reachable by id; an unknown id is 404.
        mockMvc.perform(get("/api/retrain/" + jobId).header("X-Admin-Key", "test-admin-key"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.jobId").value(jobId))
                .andExpect(jsonPath("$.status").value("FAILED"));
        mockMvc.perform(get("/api/retrain/no-such-job").header("X-Admin-Key", "test-admin-key"))
                .andExpect(status().isNotFound());
        mockMvc.perform(get("/api/retrain/" + jobId))
                .andExpect(status().isUnauthorized());
    }

}
