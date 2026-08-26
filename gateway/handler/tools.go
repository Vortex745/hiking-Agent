package handler

import (
	"fmt"

	"github.com/gin-gonic/gin"
)

// ToolsHandler handles Agent tool registry requests.
type ToolsHandler struct {
	aiServiceURL string
}

// NewToolsHandler creates a new ToolsHandler.
func NewToolsHandler(aiServiceURL string) *ToolsHandler {
	return &ToolsHandler{aiServiceURL: aiServiceURL}
}

// ToolsHealth forwards the tool registry health check to the AI service.
func (h *ToolsHandler) ToolsHealth(c *gin.Context) {
	url := fmt.Sprintf("%s/api/v1/tools/health", h.aiServiceURL)
	fallbackURL := fmt.Sprintf("%s/health", h.aiServiceURL)
	proxyHealthRequest(c, "tools", url, fallbackURL)
}
