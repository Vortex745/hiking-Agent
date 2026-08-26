package main

import (
	"fmt"

	"github.com/ai-hiking/gateway/app"
	"github.com/ai-hiking/gateway/config"
)

func main() {
	cfg := config.Load()
	r := app.NewRouter(cfg)

	addr := fmt.Sprintf(":%s", cfg.Port)
	fmt.Printf("Gateway starting on %s\n", addr)
	fmt.Printf("AI Service URL: %s\n", cfg.AIServiceURL)
	fmt.Printf("Allowed Origins: %s\n", cfg.AllowedOrigins)

	if err := r.Run(addr); err != nil {
		panic(fmt.Sprintf("Failed to start server: %v", err))
	}
}
