package middleware

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/ai-hiking/gateway/handler"

	"github.com/gin-gonic/gin"
)

func newTestRouter() *gin.Engine {
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.Use(RequestObservabilityMiddleware())
	r.GET("/probe", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"request_id":  c.Request.Header.Get(RequestIDHeader),
			"traceparent": c.Request.Header.Get(TraceparentHeader),
		})
	})
	r.GET("/boom", func(c *gin.Context) {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "boom"})
	})
	return r
}

func doRequest(t *testing.T, router *gin.Engine, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/probe", nil)
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)
	return recorder
}

func TestRequestIDGeneratedWhenMissing(t *testing.T) {
	recorder := doRequest(t, newTestRouter(), nil)
	requestID := recorder.Header().Get(RequestIDHeader)
	if !requestIDPattern.MatchString(requestID) {
		t.Fatalf("generated request id %q does not match expected format", requestID)
	}
	var body map[string]string
	if err := json.Unmarshal(recorder.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["request_id"] != requestID {
		t.Fatalf("upstream request id %q != response %q", body["request_id"], requestID)
	}
}

func TestValidRequestIDIsPreserved(t *testing.T) {
	const incoming = "20261020T220000000-42"
	recorder := doRequest(t, newTestRouter(), map[string]string{RequestIDHeader: incoming})
	if got := recorder.Header().Get(RequestIDHeader); got != incoming {
		t.Fatalf("request id not preserved: %q", got)
	}
}

func TestInvalidRequestIDIsReplaced(t *testing.T) {
	recorder := doRequest(t, newTestRouter(), map[string]string{RequestIDHeader: "bad id!"})
	got := recorder.Header().Get(RequestIDHeader)
	if got == "bad id!" || !requestIDPattern.MatchString(got) {
		t.Fatalf("invalid request id not replaced: %q", got)
	}
}

func TestValidTraceparentIsForwarded(t *testing.T) {
	const traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
	recorder := doRequest(t, newTestRouter(), map[string]string{TraceparentHeader: traceparent})
	var body map[string]string
	if err := json.Unmarshal(recorder.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["traceparent"] != traceparent {
		t.Fatalf("valid traceparent not forwarded: %q", body["traceparent"])
	}
}

func TestInvalidTraceparentIsDroppedAndLogged(t *testing.T) {
	var buffer bytes.Buffer
	original := logOutput
	logOutput = &buffer
	t.Cleanup(func() { logOutput = original })

	recorder := doRequest(t, newTestRouter(), map[string]string{
		TraceparentHeader: "01-not-hex",
		RequestIDHeader:   "20261020T220000000-7",
	})
	var body map[string]string
	if err := json.Unmarshal(recorder.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["traceparent"] != "" {
		t.Fatalf("invalid traceparent should be dropped, got %q", body["traceparent"])
	}
	logs := buffer.String()
	if !strings.Contains(logs, "traceparent.invalid") {
		t.Fatalf("missing invalid-traceparent log: %s", logs)
	}
	if !strings.Contains(logs, `"request_id":"20261020T220000000-7"`) {
		t.Fatalf("invalid-traceparent log missing request_id: %s", logs)
	}
	if !strings.Contains(logs, `"level":"WARNING"`) {
		t.Fatalf("invalid-traceparent level must be WARNING: %s", logs)
	}
}

func TestSummaryLineCarriesPinnedKeys(t *testing.T) {
	var buffer bytes.Buffer
	original := logOutput
	logOutput = &buffer
	t.Cleanup(func() { logOutput = original })

	doRequest(t, newTestRouter(), nil)

	lines := strings.Split(strings.TrimSpace(buffer.String()), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected exactly one summary line, got %d: %s", len(lines), buffer.String())
	}
	var payload map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &payload); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{
		"ts", "level", "event", "logger", "service", "schema", "caller",
		"msg", "request_id", "trace_id", "route", "status", "duration_ms",
	} {
		if _, ok := payload[key]; !ok {
			t.Fatalf("summary line missing pinned key %q: %s", key, lines[0])
		}
	}
	if payload["service"] != "gateway" || payload["logger"] != "gateway.middleware" {
		t.Fatalf("unexpected service/logger: %s", lines[0])
	}
	if payload["schema"] != float64(1) {
		t.Fatalf("schema must be 1: %s", lines[0])
	}
	if payload["level"] != "INFO" {
		t.Fatalf("successful request level must be INFO: %s", lines[0])
	}
	if !regexp.MustCompile(`^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$`).
		MatchString(payload["ts"].(string)) {
		t.Fatalf("ts must be UTC milliseconds: %v", payload["ts"])
	}
	if !regexp.MustCompile(`^requestlog\.go:\d+\[.+\]$`).
		MatchString(payload["caller"].(string)) {
		t.Fatalf("caller must be file:line[func]: %v", payload["caller"])
	}
}

func TestBareServerErrorKeepsSingleErrorSummary(t *testing.T) {
	var buffer bytes.Buffer
	original := logOutput
	logOutput = &buffer
	t.Cleanup(func() { logOutput = original })

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(RequestObservabilityMiddleware())
	router.GET("/boom", func(c *gin.Context) {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "boom"})
	})
	req := httptest.NewRequest(http.MethodGet, "/boom", nil)
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	lines := strings.Split(strings.TrimSpace(buffer.String()), "\n")
	if len(lines) != 1 {
		t.Fatalf("a bare 5xx has no cause; expected one summary line, got %d: %s", len(lines), buffer.String())
	}
	if !strings.Contains(lines[0], `"event":"request.completed"`) {
		t.Fatalf("missing summary line: %s", lines[0])
	}
	var summary map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &summary); err != nil {
		t.Fatal(err)
	}
	if summary["level"] != "ERROR" {
		t.Fatalf("5xx summary level = %v, want ERROR", summary["level"])
	}
}

func TestGeneratedRequestIDsAreUnique(t *testing.T) {
	seen := map[string]bool{}
	for i := 0; i < 100; i++ {
		id := newRequestID()
		if seen[id] {
			t.Fatalf("duplicate request id %q", id)
		}
		seen[id] = true
	}
	if !regexp.MustCompile(`^\d+-\d+$`).MatchString(newRequestID()) {
		t.Fatal("request id is not timestamp+sequence")
	}
}

func TestHTTPStatusOutcome(t *testing.T) {
	for _, tc := range []struct {
		status  int
		outcome string
	}{
		{200, "ok"}, {302, "ok"}, {400, "client_error"},
		{401, "client_error"}, {403, "client_error"}, {404, "client_error"},
		{422, "client_error"}, {429, "client_error"}, {500, "error"},
	} {
		t.Run(http.StatusText(tc.status), func(t *testing.T) {
			var buffer bytes.Buffer
			original := logOutput
			logOutput = &buffer
			t.Cleanup(func() { logOutput = original })
			router := gin.New()
			router.Use(RequestObservabilityMiddleware())
			router.GET("/probe", func(c *gin.Context) { c.Status(tc.status) })
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/probe", nil))
			lines := strings.Split(strings.TrimSpace(buffer.String()), "\n")
			var summary map[string]any
			if err := json.Unmarshal([]byte(lines[0]), &summary); err != nil {
				t.Fatal(err)
			}
			if summary["outcome"] != tc.outcome {
				t.Fatalf("status %d outcome = %v, want %s", tc.status, summary["outcome"], tc.outcome)
			}
			wantLevel := "INFO"
			if tc.status >= 500 {
				wantLevel = "ERROR"
			} else if tc.status >= 400 {
				wantLevel = "WARNING"
			}
			if summary["level"] != wantLevel {
				t.Fatalf("status %d level = %v, want %s", tc.status, summary["level"], wantLevel)
			}
			// A status alone never earns the extra failure line.
			if len(lines) != 1 {
				t.Fatalf("status %d: got %d log lines, want 1", tc.status, len(lines))
			}
		})
	}
}

func TestPanicIsReportedAsStructuredErrorLine(t *testing.T) {
	var buffer bytes.Buffer
	original := logOutput
	logOutput = &buffer
	t.Cleanup(func() { logOutput = original })

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(RequestObservabilityMiddleware())
	router.Use(RecoveryWithLog())
	router.GET("/panic", func(c *gin.Context) {
		panic("kaboom")
	})

	req := httptest.NewRequest(http.MethodGet, "/panic", nil)
	req.Header.Set(RequestIDHeader, "20261020T220000000-77")
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusInternalServerError {
		t.Fatalf("panic should map to 500, got %d", recorder.Code)
	}
	logs := buffer.String()
	if !strings.Contains(logs, `"event":"request.failed"`) {
		t.Fatalf("missing error line: %s", logs)
	}
	if !strings.Contains(logs, `"error_type":"panic"`) {
		t.Fatalf("panic not classified: %s", logs)
	}
	if !strings.Contains(logs, `"request_id":"20261020T220000000-77"`) {
		t.Fatalf("panic line missing request_id: %s", logs)
	}
	if !strings.Contains(logs, `"error_message":"kaboom"`) {
		t.Fatalf("panic message missing: %s", logs)
	}
	if strings.Contains(logs, "goroutine") {
		t.Fatalf("stack trace must not be logged: %s", logs)
	}
}

func TestUpstreamUnreachableReportsHandlerError(t *testing.T) {
	var buffer bytes.Buffer
	original := logOutput
	logOutput = &buffer
	t.Cleanup(func() { logOutput = original })

	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(RequestObservabilityMiddleware())
	handlerUnderTest := handler.NewChatV2Handler("http://127.0.0.1:1")
	router.GET("/api/v2/chat/snapshot/:run_id", handlerUnderTest.ChatV2Snapshot)

	req := httptest.NewRequest(http.MethodGet, "/api/v2/chat/snapshot/run-1", nil)
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("unreachable upstream should map to 503, got %d", recorder.Code)
	}
	logs := buffer.String()
	if !strings.Contains(logs, `"error_type":"handler_error"`) {
		t.Fatalf("handler error not classified: %s", logs)
	}
	var payload map[string]any
	for _, line := range strings.Split(strings.TrimSpace(logs), "\n") {
		var decoded map[string]any
		if err := json.Unmarshal([]byte(line), &decoded); err != nil {
			t.Fatalf("log line is not JSON: %s", line)
		}
		if decoded["event"] == "request.failed" {
			payload = decoded
		}
	}
	if payload == nil {
		t.Fatalf("missing request.failed line: %s", logs)
	}
	if message, _ := payload["error_message"].(string); message == "" {
		t.Fatalf("error_message must describe the upstream failure: %s", logs)
	}
}

// ---- sanitization: one physical line, no secrets --------------------------

func captureRequestLog(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buffer bytes.Buffer
	original := logOutput
	logOutput = &buffer
	t.Cleanup(func() { logOutput = original })
	return &buffer
}

func decodeLogLines(t *testing.T, raw string) []map[string]any {
	t.Helper()
	var records []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(raw), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var decoded map[string]any
		if err := json.Unmarshal([]byte(line), &decoded); err != nil {
			t.Fatalf("log line is not JSON (raw newline leaked?): %q", line)
		}
		records = append(records, decoded)
	}
	return records
}

func TestHandlerErrorIsSingleLineAndRedacted(t *testing.T) {
	buffer := captureRequestLog(t)
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(RequestObservabilityMiddleware())
	router.GET("/fail", func(c *gin.Context) {
		_ = c.Error(fmt.Errorf(
			"upstream failed\nAuthorization: Bearer top-secret-token password=hunter2",
		))
		c.Status(http.StatusBadGateway)
	})

	router.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/fail", nil))

	raw := buffer.String()
	for _, leaked := range []string{"top-secret-token", "hunter2"} {
		if strings.Contains(raw, leaked) {
			t.Fatalf("secret %q leaked into logs: %s", leaked, raw)
		}
	}
	var failed map[string]any
	for _, record := range decodeLogLines(t, raw) {
		if record["event"] == "request.failed" {
			failed = record
		}
	}
	if failed == nil {
		t.Fatalf("missing request.failed line: %s", raw)
	}
	message, _ := failed["error_message"].(string)
	if message == "" || !strings.Contains(message, "[REDACTED]") {
		t.Fatalf("error_message must be redacted, got %q", message)
	}
	if strings.ContainsAny(message, "\r\n\t") {
		t.Fatalf("error_message must stay on one line, got %q", message)
	}
}

func TestPanicLogIsSingleLineAndRedacted(t *testing.T) {
	buffer := captureRequestLog(t)
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(RequestObservabilityMiddleware(), RecoveryWithLog())
	router.GET("/panic", func(c *gin.Context) {
		panic("kaboom\nsecret=top-secret-value")
	})

	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/panic", nil))

	if recorder.Code != http.StatusInternalServerError {
		t.Fatalf("panic must map to 500, got %d", recorder.Code)
	}
	raw := buffer.String()
	if strings.Contains(raw, "top-secret-value") {
		t.Fatalf("panic secret leaked: %s", raw)
	}
	var failed map[string]any
	for _, record := range decodeLogLines(t, raw) {
		if record["event"] == "request.failed" {
			failed = record
		}
	}
	if failed == nil {
		t.Fatalf("missing request.failed for panic: %s", raw)
	}
	if failed["error_type"] != "panic" {
		t.Fatalf("panic must be classified, got %v", failed["error_type"])
	}
	message, _ := failed["error_message"].(string)
	if strings.ContainsAny(message, "\r\n\t") || !strings.Contains(message, "[REDACTED]") {
		t.Fatalf("panic error_message must be single-line and redacted, got %q", message)
	}
}

func TestRouteFallbackStripsControlCharacters(t *testing.T) {
	buffer := captureRequestLog(t)
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.Use(RequestObservabilityMiddleware())

	router.ServeHTTP(
		httptest.NewRecorder(),
		httptest.NewRequest(http.MethodGet, "/missing%0Aevil", nil),
	)

	records := decodeLogLines(t, buffer.String())
	if len(records) == 0 {
		t.Fatalf("expected a summary line, got none")
	}
	route, _ := records[len(records)-1]["route"].(string)
	if strings.ContainsAny(route, "\r\n\t") {
		t.Fatalf("route must not carry control characters, got %q", route)
	}
	if !strings.Contains(route, "evil") {
		t.Fatalf("route should still describe the request, got %q", route)
	}
}
