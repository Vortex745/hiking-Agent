package middleware

import (
	"bytes"
	"encoding/json"
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
		"ts", "level", "event", "msg", "request_id", "trace_id",
		"route", "status", "duration_ms",
	} {
		if _, ok := payload[key]; !ok {
			t.Fatalf("summary line missing pinned key %q: %s", key, lines[0])
		}
	}
}

func TestServerErrorAlwaysKeepsErrorLine(t *testing.T) {
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
	if len(lines) != 2 {
		t.Fatalf("expected summary + error line, got %d: %s", len(lines), buffer.String())
	}
	if !strings.Contains(lines[1], `"event":"request.failed"`) {
		t.Fatalf("missing error line: %s", lines[1])
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
	if !strings.Contains(logs, `"error_msg":"kaboom"`) {
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
	if message, _ := payload["error_msg"].(string); message == "" {
		t.Fatalf("error_msg must describe the upstream failure: %s", logs)
	}
}
