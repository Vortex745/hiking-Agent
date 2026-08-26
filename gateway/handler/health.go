package handler

import (
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// HealthHandler handles GET /health requests.
type HealthHandler struct {
	aiServiceURL string
}

// NewHealthHandler creates a new HealthHandler.
func NewHealthHandler(aiServiceURL string) *HealthHandler {
	return &HealthHandler{aiServiceURL: aiServiceURL}
}

// Health returns the health status of the gateway and AI service.
func (h *HealthHandler) Health(c *gin.Context) {
	aiStatus := "ok"
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(fmt.Sprintf("%s/health", h.aiServiceURL))
	if err != nil {
		aiStatus = "unreachable"
	} else {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			aiStatus = fmt.Sprintf("error: %s", string(body))
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"status":    "ok",
		"service":   "gateway",
		"ai_service": aiStatus,
	})
}
