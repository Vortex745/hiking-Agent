package config

import "os"

const defaultAllowedOrigins = "http://localhost:5173,http://127.0.0.1:5173,https://530745.xyz,https://ai-hiking-eight.vercel.app,https://ai-hiking-zijins-projects-8a18dbb0.vercel.app,https://ai-hiking-762618186-2434-zijins-projects-8a18dbb0.vercel.app"

// Config holds the gateway configuration.
type Config struct {
	Port           string
	AIServiceURL   string
	AllowedOrigins string
}

// Load reads configuration from environment variables.
func Load() *Config {
	return &Config{
		Port:           getEnv("PORT", "8080"),
		AIServiceURL:   getEnv("AI_SERVICE_URL", "http://localhost:8000"),
		AllowedOrigins: getEnv("ALLOWED_ORIGINS", defaultAllowedOrigins),
	}
}

func getEnv(key, fallback string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return fallback
}
