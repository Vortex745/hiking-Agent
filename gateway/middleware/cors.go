package middleware

import (
	"net/http"
	"strings"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// CORSMiddleware creates a CORS middleware with the given allowed origins.
func CORSMiddleware(allowedOrigins string) gin.HandlerFunc {
	originParts := strings.Split(allowedOrigins, ",")
	origins := make([]string, 0, len(originParts))
	for _, origin := range originParts {
		if trimmed := strings.TrimSpace(origin); trimmed != "" {
			origins = append(origins, trimmed)
		}
	}
	// 空来源时返回空中间件以避免 gin-contrib/cors panic，直接拒绝所有跨域请求
	if len(origins) == 0 {
		return func(c *gin.Context) {
			c.AbortWithStatus(http.StatusForbidden)
		}
	}
	config := cors.Config{
		AllowOrigins:     origins,
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		AllowCredentials: true,
	}
	return cors.New(config)
}
