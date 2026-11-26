package com.pcod;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.patch;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Fail-closed admin key: this context sets NO admin key (the default from
 * application.properties is empty), so every protected endpoint must
 * refuse with 503 — even with a key supplied, because no key is
 * configured to compare against. The history and retrain endpoints are
 * never silently unprotected.
 */
@SpringBootTest
@AutoConfigureMockMvc
class AdminKeyFailClosedTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    void retrainWithoutConfiguredKeyIs503() throws Exception {
        mockMvc.perform(post("/api/retrain"))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.error").isString());
        // Even a "correct-looking" key cannot help: nothing is configured.
        mockMvc.perform(post("/api/retrain").header("X-Admin-Key", "anything"))
                .andExpect(status().isServiceUnavailable());
    }

    @Test
    void predictionsWithoutConfiguredKeyIs503() throws Exception {
        mockMvc.perform(get("/api/predictions"))
                .andExpect(status().isServiceUnavailable());
        mockMvc.perform(get("/api/predictions/1"))
                .andExpect(status().isServiceUnavailable());
    }

    @Test
    void patchOutcomeWithoutConfiguredKeyIs503() throws Exception {
        mockMvc.perform(patch("/api/predictions/1/outcome")
                        .contentType(org.springframework.http.MediaType.APPLICATION_JSON)
                        .content("{\"verifiedOutcome\": 1}"))
                .andExpect(status().isServiceUnavailable());
    }

    @Test
    void retrainStatusWithoutConfiguredKeyIs503() throws Exception {
        mockMvc.perform(get("/api/retrain/status"))
                .andExpect(status().isServiceUnavailable());
        mockMvc.perform(get("/api/retrain/some-job"))
                .andExpect(status().isServiceUnavailable());
    }
}
