package handler

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestHealthRequestsForwardCorrelationHeaders(t *testing.T) {
	const requestID = "health-20261021-001"
	const traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
	for _, surface := range []string{"root", "module", "fallback"} {
		t.Run(surface, func(t *testing.T) {
			requests := make(chan *http.Request, 2)
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests <- r
				if surface == "fallback" && r.URL.Path == "/module-health" {
					w.WriteHeader(http.StatusNotFound)
					return
				}
				_, _ = w.Write([]byte(`{"status":"ok"}`))
			}))
			defer upstream.Close()
			gin.SetMode(gin.TestMode)
			router := gin.New()
			if surface == "root" {
				router.GET("/probe", NewHealthHandler(upstream.URL).Health)
			} else {
				router.GET("/probe", func(c *gin.Context) {
					proxyHealthRequest(c, "chat", upstream.URL+"/module-health", upstream.URL+"/health")
				})
			}
			req := httptest.NewRequest(http.MethodGet, "/probe", nil)
			req.Header.Set("X-Request-ID", requestID)
			req.Header.Set("traceparent", traceparent)
			recorder := httptest.NewRecorder()
			router.ServeHTTP(recorder, req)
			if recorder.Code != http.StatusOK {
				t.Fatalf("health status = %d, want 200", recorder.Code)
			}
			wantRequests := 1
			if surface == "fallback" {
				wantRequests = 2
			}
			if len(requests) != wantRequests {
				t.Fatalf("got %d upstream requests, want %d", len(requests), wantRequests)
			}
			for i := 0; i < wantRequests; i++ {
				forwarded := <-requests
				if forwarded.Header.Get("X-Request-ID") != requestID || forwarded.Header.Get("traceparent") != traceparent {
					t.Errorf("correlation lost on %s", forwarded.URL.Path)
				}
			}
		})
	}
}
