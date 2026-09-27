package app

import (
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ai-hiking/gateway/config"
	"github.com/ai-hiking/gateway/handler"
	"github.com/ai-hiking/gateway/middleware"
)

// NewRouter builds the gateway HTTP router for both local servers and Vercel Functions.
func NewRouter(cfg *config.Config) *gin.Engine {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	// Trust only the configured proxies so c.ClientIP() (rate limiting,
	// subject cookies) resolves the real client IP behind the frontend
	// nginx container. Without this, every proxied request shares the
	// nginx container IP and one global rate-limit bucket. Default trusts
	// loopback only for local dev; Docker deployments set TRUSTED_PROXIES
	// to the compose bridge network (172.16.0.0/12).
	trustedProxies := os.Getenv("TRUSTED_PROXIES")
	if trustedProxies == "" {
		trustedProxies = "127.0.0.1,::1"
	}
	if err := r.SetTrustedProxies(strings.Split(trustedProxies, ",")); err != nil {
		// Invalid list: trust no proxies — ClientIP falls back to RemoteAddr.
		_ = r.SetTrustedProxies(nil)
	}

	r.Use(middleware.CORSMiddleware(cfg.AllowedOrigins))
	r.Use(middleware.RateLimitMiddleware(60, 1*time.Minute))

	healthHandler := handler.NewHealthHandler(cfg.AIServiceURL)
	chatV2Handler := handler.NewChatV2Handler(cfg.AIServiceURL)
	modelsHandler := handler.NewModelsHandler(cfg.AIServiceURL)
	ragHandler := handler.NewRAGHandler(cfg.AIServiceURL)

	r.GET("/health", healthHandler.Health)

	// RAG / models routes (still served under /api/v1 by the AI service).
	api := r.Group("/api/v1")
	{
		api.POST("/models/fetch", modelsHandler.ModelsFetch)

		api.GET("/rag/health", ragHandler.RagHealth)
		api.POST("/rag/upload", ragHandler.RagUpload)
		api.POST("/rag/query", ragHandler.RagQuery)
		api.GET("/rag/documents", ragHandler.RagDocuments)
		api.GET("/rag/history/:chatId", ragHandler.RagHistory)
		api.DELETE("/rag/history/:chatId", ragHandler.RagClearHistory)
	}

	// Chat v2 routes — the only chat链路 (v1 chat removed). The SSE
	// endpoint rejects Vercel at the handler level (durable run state
	// requires the long-lived Docker gateway).
	v2 := r.Group("/api/v2")
	{
		v2.POST("/chat/sse", chatV2Handler.ChatV2SSE)
		v2.GET("/chat/snapshot/:run_id", chatV2Handler.ChatV2Snapshot)
		v2.POST("/chat/cancel/:run_id", chatV2Handler.ChatV2Cancel)
		v2.GET("/chat/events/:run_id", chatV2Handler.ChatV2Events)
		v2.POST("/chat/resume/:run_id", chatV2Handler.ChatV2Resume)
		v2.POST("/approvals/:approval_id/decision", chatV2Handler.ApprovalDecision)
		v2.GET("/memory/settings", chatV2Handler.MemorySettings)
		v2.PUT("/memory/settings", chatV2Handler.MemorySettings)
		v2.GET("/memories", chatV2Handler.Memories)
		v2.DELETE("/memories", chatV2Handler.Memories)
		v2.GET("/memories/export", chatV2Handler.MemoryExport)
		v2.PATCH("/memories/:memory_id", chatV2Handler.MemoryItem)
		v2.DELETE("/memories/:memory_id", chatV2Handler.MemoryItem)
		v2.POST("/memory/candidates/:candidate_id/decision", chatV2Handler.MemoryCandidateDecision)
		v2.GET("/artifacts/:artifact_id", chatV2Handler.Artifact)
		v2.POST("/subject/rotate", chatV2Handler.RotateSubject)
	}

	return r
}
