package handler

import (
	"net/http"
	"strings"

	"github.com/ai-hiking/gateway/app"
	"github.com/ai-hiking/gateway/config"
)

var router = app.NewRouter(config.Load())

// Handler is the Vercel Go Function entrypoint.
func Handler(w http.ResponseWriter, r *http.Request) {
	if rewrittenPath := r.URL.Query().Get("path"); rewrittenPath != "" {
		r.URL.Path = "/" + strings.TrimLeft(rewrittenPath, "/")
	}

	router.ServeHTTP(w, r)
}
