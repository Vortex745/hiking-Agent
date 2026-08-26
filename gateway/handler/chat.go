package handler

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/gin-gonic/gin"
)

// ChatHandler handles chat-related requests.
type ChatHandler struct {
	aiServiceURL string
}

// NewChatHandler creates a new ChatHandler.
func NewChatHandler(aiServiceURL string) *ChatHandler {
	return &ChatHandler{aiServiceURL: aiServiceURL}
}

// ChatSync forwards a sync chat request to the AI service.
func (h *ChatHandler) ChatSync(c *gin.Context) {
	url := fmt.Sprintf("%s/api/v1/chat/sync", h.aiServiceURL)
	proxyRequest(c, http.MethodPost, url, c.Request.Body)
}

// ChatHealth forwards an Agent health check to the AI service.
func (h *ChatHandler) ChatHealth(c *gin.Context) {
	url := fmt.Sprintf("%s/api/v1/chat/health", h.aiServiceURL)
	fallbackURL := fmt.Sprintf("%s/health", h.aiServiceURL)
	proxyHealthRequest(c, "agent", url, fallbackURL)
}

// ChatSSE proxies streaming SSE chat from the AI service.
func (h *ChatHandler) ChatSSE(c *gin.Context) {
	message := c.Query("message")
	chatID := c.Query("chat_id")
	var requestBody io.Reader

	if message == "" {
		rawBody, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		var req map[string]any
		if err := json.Unmarshal(rawBody, &req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "message is required"})
			return
		}
		if value, ok := req["message"].(string); ok {
			message = value
		}
		if value, ok := req["chat_id"].(string); ok {
			chatID = value
		}
		requestBody = bytes.NewReader(rawBody)
	} else {
		rawBody, err := json.Marshal(gin.H{"message": message, "chat_id": chatID})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		requestBody = bytes.NewReader(rawBody)
	}
	if message == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "message is required"})
		return
	}

	url := fmt.Sprintf("%s/api/v1/chat/sse", h.aiServiceURL)
	proxyStreamRequest(c, http.MethodPost, url, requestBody, "application/json")
}

// ChatHistory forwards a chat history request to the AI service.
func (h *ChatHandler) ChatHistory(c *gin.Context) {
	chatID := c.Param("chatId")
	url := fmt.Sprintf("%s/api/v1/chat/history/%s", h.aiServiceURL, chatID)
	proxyRequest(c, http.MethodGet, url, nil)
}

// ChatClearHistory clears chat history in the AI service.
func (h *ChatHandler) ChatClearHistory(c *gin.Context) {
	chatID := c.Param("chatId")
	url := fmt.Sprintf("%s/api/v1/chat/history/%s", h.aiServiceURL, chatID)
	proxyRequest(c, http.MethodDelete, url, nil)
}

// ChatConfirm forwards a high-risk tool confirmation request to the AI service.
func (h *ChatHandler) ChatConfirm(c *gin.Context) {
	url := fmt.Sprintf("%s/api/v1/chat/confirm", h.aiServiceURL)
	proxyRequest(c, http.MethodPost, url, c.Request.Body)
}

// ChatPending forwards a pending confirmation lookup to the AI service.
func (h *ChatHandler) ChatPending(c *gin.Context) {
	chatID := c.Param("chatId")
	url := fmt.Sprintf("%s/api/v1/chat/pending/%s", h.aiServiceURL, url.PathEscape(chatID))
	proxyRequest(c, http.MethodGet, url, nil)
}

// ArtifactDownload forwards generated workspace artifact downloads to the AI service.
func (h *ChatHandler) ArtifactDownload(c *gin.Context) {
	filePath := strings.TrimPrefix(c.Param("filePath"), "/")
	if filePath == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "file path is required"})
		return
	}

	url := fmt.Sprintf("%s/api/v1/artifacts/%s", h.aiServiceURL, escapePathSegments(filePath))
	proxyStreamResponse(c, http.MethodGet, url, nil)
}

func escapePathSegments(filePath string) string {
	parts := strings.Split(filePath, "/")
	for i, part := range parts {
		parts[i] = url.PathEscape(part)
	}
	return strings.Join(parts, "/")
}

// proxyStreamResponse streams the response body to the client without buffering.
// Used for large responses like file downloads to avoid OOM on the gateway.
func proxyStreamResponse(c *gin.Context, method, url string, body io.Reader) {
	req, err := http.NewRequestWithContext(c.Request.Context(), method, url, body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if contentType := c.Request.Header.Get("Content-Type"); contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "AI service unreachable"})
		return
	}
	defer resp.Body.Close()

	// Copy response headers
	for key, values := range resp.Header {
		for _, value := range values {
			c.Writer.Header().Add(key, value)
		}
	}

	c.Writer.WriteHeader(resp.StatusCode)

	// Stream the body directly to the client
	if _, err := io.Copy(c.Writer, resp.Body); err != nil {
		// Client disconnected or write error — nothing we can do
		return
	}
}

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

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
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

func proxyStreamRequest(c *gin.Context, method, url string, body io.Reader, contentType string) {
	req, err := http.NewRequestWithContext(c.Request.Context(), method, url, body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	} else if requestContentType := c.Request.Header.Get("Content-Type"); requestContentType != "" {
		req.Header.Set("Content-Type", requestContentType)
	}
	req.Header.Set("Accept", "text/event-stream")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "AI service unreachable"})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		respBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": readErr.Error()})
			return
		}
		c.Data(resp.StatusCode, resp.Header.Get("Content-Type"), respBody)
		return
	}

	if writeBufferedStreamOnVercel(c, resp) {
		return
	}

	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "streaming not supported"})
		return
	}

	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.Header().Set("X-Accel-Buffering", "no")
	c.Writer.WriteHeader(resp.StatusCode)

	reader := bufio.NewReader(resp.Body)
	for {
		line, readErr := reader.ReadString('\n')
		if line != "" {
			if _, err := fmt.Fprint(c.Writer, line); err != nil {
				break
			}
			flusher.Flush()
		}
		if readErr != nil {
			break
		}
	}
}

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

func proxyHealthRequest(c *gin.Context, module, url, fallbackURL string) {
	client := &http.Client{}
	req, err := http.NewRequestWithContext(c.Request.Context(), http.MethodGet, url, nil)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
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

	fallbackResp, err := client.Do(fallbackReq)
	if err != nil {
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
