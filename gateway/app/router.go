package app

import (
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

	r.Use(middleware.CORSMiddleware(cfg.AllowedOrigins))
	r.Use(middleware.RateLimitMiddleware(60, 1*time.Minute))

	healthHandler := handler.NewHealthHandler(cfg.AIServiceURL)
	chatHandler := handler.NewChatHandler(cfg.AIServiceURL)
	chatV2Handler := handler.NewChatV2Handler(cfg.AIServiceURL, handler.RuntimeSelector{
		Secret: cfg.RuntimeSecret, Canary: cfg.ChatV2Canary,
		Force: cfg.RuntimeForce, Environment: cfg.RuntimeEnv,
	})
	modelsHandler := handler.NewModelsHandler(cfg.AIServiceURL)
	ragHandler := handler.NewRAGHandler(cfg.AIServiceURL)
	toolsHandler := handler.NewToolsHandler(cfg.AIServiceURL)

	r.GET("/health", healthHandler.Health)
	r.GET("/api/chat/runtime/:conversation_id", chatV2Handler.Runtime)

	api := r.Group("/api/v1")
	{
		api.GET("/chat/health", chatHandler.ChatHealth)
		api.POST("/chat/sync", chatHandler.ChatSync)
		api.POST("/chat/sse", chatHandler.ChatSSE)
		api.GET("/chat/history/:chatId", chatHandler.ChatHistory)
		api.DELETE("/chat/history/:chatId", chatHandler.ChatClearHistory)
		api.POST("/chat/confirm", chatHandler.ChatConfirm)
		api.GET("/chat/pending/:chatId", chatHandler.ChatPending)
		api.GET("/artifacts/*filePath", chatHandler.ArtifactDownload)

		api.POST("/models/fetch", modelsHandler.ModelsFetch)

		api.GET("/rag/health", ragHandler.RagHealth)
		api.POST("/rag/upload", ragHandler.RagUpload)
		api.POST("/rag/query", ragHandler.RagQuery)
		api.GET("/rag/documents", ragHandler.RagDocuments)
		api.GET("/rag/history/:chatId", ragHandler.RagHistory)
		api.DELETE("/rag/history/:chatId", ragHandler.RagClearHistory)

		api.GET("/tools/health", toolsHandler.ToolsHealth)
	}

	// Ticket 07: Chat v2 routes. The SSE endpoint rejects Vercel at the
	// handler level (durable run state requires the long-lived Docker
	// gateway). All v1 routes above remain unchanged.
	v2 := r.Group("/api/v2")
	{
		v2.POST("/chat/sse", chatV2Handler.ChatV2SSE)
		v2.GET("/chat/snapshot/:run_id", chatV2Handler.ChatV2Snapshot)
		v2.POST("/chat/cancel/:run_id", chatV2Handler.ChatV2Cancel)
		v2.GET("/chat/events/:run_id", chatV2Handler.ChatV2Events)
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
