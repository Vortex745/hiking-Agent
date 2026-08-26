package middleware

import (
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
)

type bucket struct {
	tokens    int
	lastCheck time.Time
}

// RateLimitMiddleware creates a token-bucket rate limiter per IP.
func RateLimitMiddleware(rate int, per time.Duration) gin.HandlerFunc {
	var mu sync.Mutex
	buckets := make(map[string]*bucket)

	if os.Getenv("VERCEL") == "" {
		// Cleanup stale buckets for long-running local/container processes.
		go func() {
			for {
				time.Sleep(5 * time.Minute)
				mu.Lock()
				now := time.Now()
				for ip, b := range buckets {
					if now.Sub(b.lastCheck) > 10*time.Minute {
						delete(buckets, ip)
					}
				}
				mu.Unlock()
			}
		}()
	}

	return func(c *gin.Context) {
		ip := c.ClientIP()

		mu.Lock()
		b, exists := buckets[ip]
		if !exists {
			b = &bucket{tokens: rate, lastCheck: time.Now()}
			buckets[ip] = b
		}

		now := time.Now()
		elapsed := now.Sub(b.lastCheck)
		b.tokens += int(elapsed.Seconds() * float64(rate) / per.Seconds())
		if b.tokens > rate {
			b.tokens = rate
		}
		b.lastCheck = now

		if b.tokens > 0 {
			b.tokens--
			mu.Unlock()
			c.Next()
		} else {
			mu.Unlock()
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error": "rate limit exceeded",
			})
			c.Abort()
		}
	}
}
