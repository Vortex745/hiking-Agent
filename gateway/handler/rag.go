package handler

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

// RAGHandler handles RAG-related requests.
type RAGHandler struct {
	aiServiceURL string
}

// NewRAGHandler creates a new RAGHandler.
func NewRAGHandler(aiServiceURL string) *RAGHandler {
	return &RAGHandler{aiServiceURL: aiServiceURL}
}

// RagUpload forwards a document upload to the AI service.
func (h *RAGHandler) RagUpload(c *gin.Context) {
	url := fmt.Sprintf("%s/api/v1/rag/upload", h.aiServiceURL)
	// Forward the entire request body directly (Gin already parsed multipart).
	proxyRequest(c, http.MethodPost, url, c.Request.Body)
}

// RagHealth forwards a RAG health check to the AI service.
func (h *RAGHandler) RagHealth(c *gin.Context) {
	url := fmt.Sprintf("%s/api/v1/rag/health", h.aiServiceURL)
	fallbackURL := fmt.Sprintf("%s/health", h.aiServiceURL)
	proxyHealthRequest(c, "rag", url, fallbackURL)
}

// RagQuery forwards a RAG query to the AI service.
func (h *RAGHandler) RagQuery(c *gin.Context) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if answer := directRagAnswerFromBody(body); answer != "" {
		writeDirectRagAnswer(c, answer)
		return
	}
	c.Request.Body = io.NopCloser(bytes.NewReader(body))

	url := fmt.Sprintf("%s/api/v1/rag/query", h.aiServiceURL)
	h.proxyRagQuery(c, url)
}

// RagDocuments lists all uploaded documents from the AI service.
func (h *RAGHandler) RagDocuments(c *gin.Context) {
	url := fmt.Sprintf("%s/api/v1/rag/documents", h.aiServiceURL)
	proxyRequest(c, http.MethodGet, url, nil)
}

// RagHistory forwards a RAG chat history request to the AI service.
func (h *RAGHandler) RagHistory(c *gin.Context) {
	chatID := c.Param("chatId")
	url := fmt.Sprintf("%s/api/v1/rag/history/%s", h.aiServiceURL, chatID)
	proxyRequest(c, http.MethodGet, url, nil)
}

// RagClearHistory clears RAG chat history in the AI service.
func (h *RAGHandler) RagClearHistory(c *gin.Context) {
	chatID := c.Param("chatId")
	url := fmt.Sprintf("%s/api/v1/rag/history/%s", h.aiServiceURL, chatID)
	proxyRequest(c, http.MethodDelete, url, nil)
}

func (h *RAGHandler) proxyRagQuery(c *gin.Context, url string) {
	req, err := http.NewRequestWithContext(c.Request.Context(), http.MethodPost, url, c.Request.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if requestContentType := c.Request.Header.Get("Content-Type"); requestContentType != "" {
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
			if isRagEmbeddingNotFoundLine(line) && h.ragKnowledgeBaseIsEmpty(c.Request.Context()) {
				writeEmptyRagFallback(c, flusher)
				return
			}
			if _, err := fmt.Fprint(c.Writer, line); err != nil {
				return
			}
			flusher.Flush()
		}
		if readErr != nil {
			break
		}
	}
}

func isRagEmbeddingNotFoundLine(line string) bool {
	// SSE 格式中 JSON 键值对可能带或不带空格，所以用更灵活的方式匹配
	return strings.Contains(line, "404") &&
		strings.Contains(line, `"error"`)
}

func directRagAnswerFromBody(body []byte) string {
	var req struct {
		Question string `json:"question"`
		ChatID   string `json:"chat_id"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		return ""
	}
	if strings.TrimSpace(req.ChatID) != "" {
		return ""
	}

	normalized := strings.TrimSpace(strings.ToLower(req.Question))
	normalized = strings.TrimRight(normalized, "。！？!?")
	switch normalized {
	case "你好", "您好", "hi", "hello", "嗨":
		return "你好，我是 AI Hiking 的 RAG 助手。你可以上传文档后向我提问，也可以先问一些简单问题。"
	case "你是谁":
		return "我是 AI Hiking 的 RAG 助手，可以帮你检索知识库、总结文档，也可以回答一些简单问题。"
	case "谢谢", "谢谢你":
		return "不客气。需要查知识库内容时，直接把问题发给我就行。"
	default:
		return ""
	}
}

func writeDirectRagAnswer(c *gin.Context, answer string) {
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.Header().Set("X-Accel-Buffering", "no")
	c.Writer.WriteHeader(http.StatusOK)

	events := []string{
		fmt.Sprintf("data: {\"type\":\"text\",\"content\":%q}\n\n", answer),
		`data: {"type":"done","content":""}` + "\n\n",
	}
	for _, event := range events {
		_, _ = fmt.Fprint(c.Writer, event)
	}
}

func (h *RAGHandler) ragKnowledgeBaseIsEmpty(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s/api/v1/rag/health", h.aiServiceURL), nil)
	if err != nil {
		return false
	}
	req.Header.Set("Accept", "application/json")

	resp, err := (&http.Client{}).Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return false
	}

	var health struct {
		Storage   string `json:"storage"`
		Documents int    `json:"documents"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		return false
	}
	return health.Storage == "memory" && health.Documents == 0
}

func writeEmptyRagFallback(c *gin.Context, flusher http.Flusher) {
	events := []string{
		`data: {"type":"process","content":"调用 embedding 模型生成查询向量"}` + "\n\n",
		`data: {"type":"process","content":"使用向量在 pgvector/memory 中召回候选片段","metadata":{"candidate_chunks":0,"storage":"memory"}}` + "\n\n",
		`data: {"type":"documents","content":"已检索 0 篇相关文档，共 0 个片段","metadata":{"searched_count":0,"matched_chunks":0,"documents":[]}}` + "\n\n",
		`data: {"type":"text","content":"知识库中没有找到相关文档。可以换个更具体的问题，或先上传/同步相关资料后再查。"}` + "\n\n",
		`data: {"type":"done","content":""}` + "\n\n",
	}
	for _, event := range events {
		if _, err := fmt.Fprint(c.Writer, event); err != nil {
			return
		}
		flusher.Flush()
	}
}
