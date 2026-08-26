package handler

import (
	"fmt"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"
)

type ModelsHandler struct {
	aiServiceURL string
}

func NewModelsHandler(aiServiceURL string) *ModelsHandler {
	return &ModelsHandler{aiServiceURL: aiServiceURL}
}

// ModelsFetch 直接请求上游服务商的 /models 接口获取模型列表
// 不再转发到 AI Service，避免 AI Service 未启动时无法获取模型列表
func (h *ModelsHandler) ModelsFetch(c *gin.Context) {
	var req struct {
		BaseURL string `json:"base_url"`
		APIKey  string `json:"api_key"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	if req.BaseURL == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "base_url is required"})
		return
	}

	// 构建上游 /models 接口 URL
	modelsURL := fmt.Sprintf("%s/models", req.BaseURL)

	httpReq, err := http.NewRequestWithContext(c.Request.Context(), http.MethodGet, modelsURL, nil)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", req.APIKey))
	httpReq.Header.Set("Accept", "application/json")

	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": fmt.Sprintf("upstream unreachable: %v", err)})
		return
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// 直接透传上游响应体，让前端自行解析
	c.Data(resp.StatusCode, "application/json", respBody)
}
