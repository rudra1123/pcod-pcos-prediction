package com.pcod;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * POST /api/retrain is protected by the X-Admin-Key header when ADMIN_KEY
 * is set. This test runs with pcod.admin.key=test-admin-key and asserts
 * that retraining is refused without the key and with a wrong key.
 * (The success path is NOT tested here — it would run the full retrain.)
 */
@SpringBootTest(properties = "pcod.admin.key=test-admin-key")
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
}
