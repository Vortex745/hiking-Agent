package handler

import (
	"io"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
)

// Shared proxy helpers used by the RAG and v2 handlers.

func copyCorrelationHeaders(c *gin.Context, req *http.Request) {
	for _, header := range []string{"X-Request-ID", "traceparent"} {
		if value := c.Request.Header.Get(header); value != "" {
			req.Header.Set(header, value)
		}
	}
}

// proxyRequest forwards a non-streaming request to the AI service and
// passes the response through verbatim including status code.
func proxyRequest(c *gin.Context, method, url string, body io.Reader) {
	req, err := http.NewRequestWithContext(c.Request.Context(), method, url, body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Copy relevant headers
	if contentType := c.Request.Header.Get("Content-Type"); contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	copyCorrelationHeaders(c, req)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		_ = c.Error(err)
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "AI service unreachable"})
		return
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.Data(resp.StatusCode, resp.Header.Get("Content-Type"), respBody)
}

// writeBufferedStreamOnVercel buffers an SSE stream when running on
// Vercel Functions (no streaming response support there).
func writeBufferedStreamOnVercel(c *gin.Context, resp *http.Response) bool {
	if os.Getenv("VERCEL") == "" {
		return false
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return true
	}

	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "text/event-stream"
	}
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.Header().Set("X-Accel-Buffering", "no")
	c.Data(resp.StatusCode, contentType, respBody)
	return true
}

// proxyHealthRequest proxies a module health check to the AI service,
// falling back to the AI service root /health when the module route
// returns 404.
func proxyHealthRequest(c *gin.Context, module, url, fallbackURL string) {
	client := &http.Client{}
	req, err := http.NewRequestWithContext(c.Request.Context(), http.MethodGet, url, nil)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	req.Header.Set("Accept", "application/json")
	copyCorrelationHeaders(c, req)

	resp, err := client.Do(req)
	if err != nil {
		_ = c.Error(err)
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "AI service unreachable"})
		return
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if resp.StatusCode != http.StatusNotFound {
		c.Data(resp.StatusCode, resp.Header.Get("Content-Type"), respBody)
		return
	}

	fallbackReq, err := http.NewRequestWithContext(c.Request.Context(), http.MethodGet, fallbackURL, nil)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	fallbackReq.Header.Set("Accept", "application/json")
	copyCorrelationHeaders(c, fallbackReq)

	fallbackResp, err := client.Do(fallbackReq)
	if err != nil {
		_ = c.Error(err)
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "AI service unreachable"})
		return
	}
	defer fallbackResp.Body.Close()

	if fallbackResp.StatusCode < http.StatusOK || fallbackResp.StatusCode >= http.StatusMultipleChoices {
		fallbackBody, readErr := io.ReadAll(fallbackResp.Body)
		if readErr != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": readErr.Error()})
			return
		}
		c.Data(fallbackResp.StatusCode, fallbackResp.Header.Get("Content-Type"), fallbackBody)
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status":   "ok",
		"module":   module,
		"service":  "gateway",
		"fallback": "ai-service-health",
	})
}
