package middleware

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
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
	requestSequence    uint64
	logMu              sync.Mutex
	logOutput          io.Writer = os.Stdout
)

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

func writeLog(payload map[string]any) {
	payload["ts"] = time.Now().UTC().Format(time.RFC3339Nano)
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
					"level":      "WARN",
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
		durationMS := float64(time.Since(started).Microseconds()) / 1000.0
		writeLog(map[string]any{
			"level":       "INFO",
			"event":       "request.completed",
			"msg":         "request completed",
			"request_id":  requestID,
			"trace_id":    traceID,
			"route":       route,
			"status":      status,
			"duration_ms": durationMS,
		})

		if status >= 500 || len(c.Errors) > 0 {
			errorType := "upstream_error"
			errorMessage := ""
			if panicValue, ok := c.Get("panic_error"); ok {
				errorType = "panic"
				errorMessage = fmt.Sprintf("%v", panicValue)
			} else if len(c.Errors) > 0 {
				errorType = "handler_error"
				errorMessage = c.Errors.Last().Error()
			}
			writeLog(map[string]any{
				"level":       "ERROR",
				"event":       "request.failed",
				"msg":         "request failed",
				"request_id":  requestID,
				"trace_id":    traceID,
				"route":       route,
				"status":      status,
				"duration_ms": durationMS,
				"error_type":  errorType,
				"error_msg":   errorMessage,
			})
		}
	}
}
