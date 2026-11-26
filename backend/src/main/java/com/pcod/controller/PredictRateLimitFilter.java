package com.pcod.controller;

import jakarta.servlet.Filter;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Fixed-window rate limit on POST /api/predict: {@value #LIMIT_PER_MINUTE}
 * requests per minute per client IP. The predict endpoint is public (no
 * admin key) and writes a row to the database on every call, so without a
 * limit one client could flood the history table. Dependency-free: one
 * in-memory counter per IP per one-minute window, no Redis, no library.
 * Other endpoints are not limited (health/models/metrics are read-only and
 * cheap; history/retrain already need the admin key). A limited request
 * gets 429 with a JSON error, not a stack trace.
 *
 * <p>Size bound: the map holds at most one entry per distinct IP seen in
 * the current minute — windows from earlier minutes are evicted on the
 * first request of a new minute (see {@link #evictStaleWindows}), so it
 * cannot keep one entry per distinct IP forever. Within a single minute
 * the map can still grow by one entry per distinct IP that calls the
 * endpoint; that is inherent to an in-memory per-IP limiter with no
 * external store.</p>
 */
@Component
public class PredictRateLimitFilter implements Filter {

    /** Requests per minute per IP on POST /api/predict. */
    public static final int LIMIT_PER_MINUTE = 120;

    private record Window(long windowStartMinute, AtomicInteger count) {}

    private final ConcurrentHashMap<String, Window> windows = new ConcurrentHashMap<>();

    /** Minute of the last stale-window sweep (-1 = never swept). */
    private volatile long lastSweepMinute = -1;

    /**
     * Client IP: the first entry of X-Forwarded-For when present, else the
     * direct remote address. The header is trusted only because this
     * deployment sits behind the project's own nginx (frontend container),
     * which sets it; a client reaching the backend directly cannot spoof a
     * shared limit — the worst case is its own IP being limited.
     */
    private static String clientIp(HttpServletRequest request) {
        String forwarded = request.getHeader("X-Forwarded-For");
        if (forwarded != null && !forwarded.isBlank()) {
            return forwarded.split(",", 2)[0].trim();
        }
        return request.getRemoteAddr();
    }

    /**
     * Drops windows from earlier minutes, at most once per minute. Runs on
     * the request path, so entries only linger while no request arrives.
     */
    private void evictStaleWindows(long minute) {
        if (minute != lastSweepMinute) {
            lastSweepMinute = minute;
            windows.entrySet().removeIf(e -> e.getValue().windowStartMinute() < minute);
        }
    }

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {
        if (request instanceof HttpServletRequest httpRequest
                && response instanceof HttpServletResponse httpResponse
                && "POST".equalsIgnoreCase(httpRequest.getMethod())
                && httpRequest.getRequestURI().endsWith("/api/predict")) {
            long minute = System.currentTimeMillis() / 60_000L;
            evictStaleWindows(minute);
            String ip = clientIp(httpRequest);
            Window window = windows.compute(ip, (key, old) ->
                    (old == null || old.windowStartMinute() != minute)
                            ? new Window(minute, new AtomicInteger(0))
                            : old);
            if (window.count().incrementAndGet() > LIMIT_PER_MINUTE) {
                httpResponse.setStatus(429);
                httpResponse.setContentType("application/json");
                httpResponse.getWriter().write("{\"error\":\"Rate limit exceeded: at most "
                        + LIMIT_PER_MINUTE + " predictions per minute per IP. "
                        + "Wait a minute and try again.\"}");
                return;
            }
        }
        chain.doFilter(request, response);
    }
}
