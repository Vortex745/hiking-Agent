package middleware

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
)

const (
	// RequestIDHeader is the correlation key propagated to the AI service.
	RequestIDHeader = "X-Request-ID"
	// TraceparentHeader follows the W3C Trace Context format.
	TraceparentHeader = "traceparent"
)

var (
	requestIDPattern   = regexp.MustCompile(`^[A-Za-z0-9._-]{8,128}$`)
	traceparentPattern = regexp.MustCompile(`^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$`)
	ansiPattern        = regexp.MustCompile(`\x1b\[[0-9;]*[A-Za-z]`)
	// Mirrors the ai-service reject list (exact keys plus suffixed variants:
	// session_token, client_secret, new_password, secret_key, ...).
	secretPatterns = []*regexp.Regexp{
		regexp.MustCompile(`(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+`),
		regexp.MustCompile(`\bsk-[A-Za-z0-9._-]{8,}`),
		regexp.MustCompile(`(?i)\bauthorization\s*:\s*(?:basic|bearer)\s+[^\s,;]+`),
		regexp.MustCompile(`(?i)\bcookie\s*:\s*[^\n;]+`),
		regexp.MustCompile(`(?i)\bx-api-key\s*:\s*[^\s,;]+`),
		regexp.MustCompile(`(?i)\b[\w-]*(?:token|secret(?:_key)?|private_key|password|passwd|api[_-]?key|apikey|ticket|pairing_link|assertion|authorization(?:_url|_attempt)?|cookie)\s*[=:]\s*[^\s,;&]+`),
	}
	requestSequence uint64
	logMu           sync.Mutex
	logOutput       io.Writer = os.Stdout
)

// sanitizeForLog keeps a dynamic value on one physical line and strips
// credentials before it reaches the structured stream.
func sanitizeForLog(value string) string {
	if value == "" {
		return ""
	}
	value = ansiPattern.ReplaceAllString(value, "")
	for _, pattern := range secretPatterns {
		value = pattern.ReplaceAllString(value, "[REDACTED]")
	}
	var builder strings.Builder
	builder.Grow(len(value))
	for _, char := range value {
		switch {
		case char == '\r' || char == '\n' || char == '\t':
			builder.WriteByte(' ')
		case char < 0x20 || char == 0x7f:
			// Drop remaining control characters.
		default:
			builder.WriteRune(char)
		}
	}
	return builder.String()
}

// newRequestID mirrors the AI service format: timestamp + atomic sequence.
func newRequestID() string {
	return fmt.Sprintf(
		"%d-%d",
		time.Now().UnixNano()/int64(time.Millisecond),
		atomic.AddUint64(&requestSequence, 1),
	)
}

// parseTraceparent validates W3C Trace Context and returns the trace id.
// Invalid versions, all-zero trace/span ids and malformed values are dropped.
func parseTraceparent(value string) (string, bool) {
	match := traceparentPattern.FindStringSubmatch(value)
	if match == nil {
		return "", false
	}
	if match[1] == "ff" || match[2] == "00000000000000000000000000000000" ||
		match[3] == "0000000000000000" {
		return "", false
	}
	return match[2], true
}

func callSite(skip int) string {
	pc, file, line, ok := runtime.Caller(skip)
	if !ok {
		return ""
	}
	name := "unknown"
	if fn := runtime.FuncForPC(pc); fn != nil {
		name = path.Base(fn.Name())
		if idx := strings.Index(name, "."); idx >= 0 {
			name = name[idx+1:]
		}
	}
	return fmt.Sprintf("%s:%d[%s]", filepath.Base(file), line, name)
}

func writeLog(payload map[string]any) {
	payload["ts"] = time.Now().UTC().Format("2006-01-02T15:04:05.000Z")
	payload["schema"] = 1
	payload["service"] = "gateway"
	payload["logger"] = "gateway.middleware"
	if _, ok := payload["caller"]; !ok {
		payload["caller"] = callSite(2)
	}
	logMu.Lock()
	defer logMu.Unlock()
	encoded, err := json.Marshal(payload)
	if err != nil {
		return
	}
	encoded = append(encoded, '\n')
	_, _ = logOutput.Write(encoded)
}

// RecoveryWithLog turns a panic into a 500 and records the panic value as a
// structured error. Gin's default recovery only writes a stack trace to
// stderr; the request summary line would then report a bare 500 with no
// cause. Stack traces are deliberately not logged.
func RecoveryWithLog() gin.HandlerFunc {
	return gin.CustomRecovery(func(c *gin.Context, recovered any) {
		message := fmt.Sprintf("%v", recovered)
		c.Set("panic_error", message)
		_ = c.Error(fmt.Errorf("panic: %s", message))
		c.AbortWithStatus(http.StatusInternalServerError)
	})
}

// RequestObservabilityMiddleware assigns/validates correlation ids and emits
// one structured summary line per request. Errors always keep an extra line.
func RequestObservabilityMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		started := time.Now()
		incoming := c.GetHeader(RequestIDHeader)
		requestID := incoming
		if !requestIDPattern.MatchString(incoming) {
			requestID = newRequestID()
		}
		c.Set("request_id", requestID)
		c.Request.Header.Set(RequestIDHeader, requestID)
		c.Writer.Header().Set(RequestIDHeader, requestID)

		traceID := ""
		if raw := c.GetHeader(TraceparentHeader); raw != "" {
			if parsed, ok := parseTraceparent(raw); ok {
				traceID = parsed
			} else {
				c.Request.Header.Del(TraceparentHeader)
				writeLog(map[string]any{
					"level":      "WARNING",
					"event":      "traceparent.invalid",
					"msg":        "invalid traceparent dropped",
					"request_id": requestID,
					"trace_id":   "",
				})
			}
		}

		c.Next()

		status := c.Writer.Status()
		route := c.FullPath()
		if route == "" {
			route = c.Request.URL.Path
		}
		route = sanitizeForLog(route)
		durationMS := float64(time.Since(started).Microseconds()) / 1000.0
		failed := status >= 500 || len(c.Errors) > 0
		outcome := "ok"
		if failed {
			outcome = "error"
		} else if status >= 400 {
			outcome = "client_error"
		}
		level := "INFO"
		if failed {
			level = "ERROR"
		} else if status >= 400 {
			level = "WARNING"
		}
		writeLog(map[string]any{
			"level":       level,
			"event":       "request.completed",
			"msg":         "request completed",
			"request_id":  requestID,
			"trace_id":    traceID,
			"route":       route,
			"status":      status,
			"duration_ms": durationMS,
			"outcome":     outcome,
		})

		// A bare 5xx has no cause to report; only a handler error/panic earns
		// the extra failure line.
		if len(c.Errors) > 0 {
			errorType := "upstream_error"
			errorMessage := ""
			if panicValue, ok := c.Get("panic_error"); ok {
				errorType = "panic"
				errorMessage = sanitizeForLog(fmt.Sprintf("%v", panicValue))
			} else if len(c.Errors) > 0 {
				errorType = "handler_error"
				errorMessage = sanitizeForLog(c.Errors.Last().Error())
			}
			writeLog(map[string]any{
				"level":         "ERROR",
				"event":         "request.failed",
				"msg":           "request failed",
				"request_id":    requestID,
				"trace_id":      traceID,
				"route":         route,
				"status":        status,
				"duration_ms":   durationMS,
				"outcome":       outcome,
				"error_type":    errorType,
				"error_message": errorMessage,
			})
		}
	}
}
