package handler

import (
	"bufio"
	"context"
	"encoding/base64"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

func v2TestRouter(upstream string) *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	h := NewChatV2Handler(upstream)
	router.POST("/chat", h.ChatV2SSE)
	router.GET("/snapshot/:run_id", h.ChatV2Snapshot)
	router.GET("/events/:run_id", h.ChatV2Events)
	router.POST("/resume/:run_id", h.ChatV2Resume)
	router.POST("/cancel/:run_id", h.ChatV2Cancel)
	router.POST("/approvals/:approval_id/decision", h.ApprovalDecision)
	router.POST("/candidates/:candidate_id/decision", h.MemoryCandidateDecision)
	router.GET("/memories", h.Memories)
	router.GET("/artifacts/:artifact_id", h.Artifact)
	router.POST("/sessions/lifecycle", h.SessionLifecycle)
	return router
}

func TestV2ForwardingPreservesIdentityAndRecovery(t *testing.T) {
	t.Setenv("VERCEL", "")
	token := base64.RawURLEncoding.EncodeToString(make([]byte, 32))
	for _, tc := range []struct{ method, path, target string }{
		{"POST", "/chat", "/api/v2/chat/sse"},
		{"GET", "/snapshot/run?request_id=retry%20key", "/api/v2/chat/snapshot/run?request_id=retry%20key"},
		{"GET", "/events/run?from_seq=8", "/api/v2/chat/events/run?from_seq=8"},
		{"POST", "/resume/run?from_seq=8", "/api/v2/chat/resume/run?from_seq=8"},
		{"POST", "/cancel/run", "/api/v2/chat/cancel/run"},
		{"POST", "/approvals/a/decision", "/api/v2/approvals/a/decision"},
		{"POST", "/candidates/c/decision", "/api/v2/memory/candidates/c/decision"},
		{"GET", "/memories?limit=5", "/api/v2/memories?limit=5"},
		{"GET", "/artifacts/pdf", "/api/v2/artifacts/pdf"},
		{"POST", "/sessions/lifecycle", "/api/v2/sessions/lifecycle"},
	} {
		t.Run(tc.path, func(t *testing.T) {
			body := `{"idempotency_key":"retry-key"}`
			if tc.method == http.MethodGet {
				body = ""
			}
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				data, _ := io.ReadAll(r.Body)
				cookie, err := r.Cookie(subjectCookieName)
				if r.Method != tc.method || r.URL.RequestURI() != tc.target || string(data) != body ||
					r.Header.Get("Content-Type") != "application/json" ||
					r.Header.Get("X-Idempotency-Key") != "retry-key" || r.Header.Get("X-Request-Id") != "request-key" ||
					err != nil || cookie.Value != token {
					t.Errorf("forwarded request changed: %s %s, body=%s, headers=%v", r.Method, r.URL, data, r.Header)
				}
				w.Header().Set("Content-Type", "application/pdf")
				w.Header().Set("Content-Disposition", `attachment; filename="plan.pdf"`)
				http.SetCookie(w, &http.Cookie{Name: subjectCookieName, Value: token})
				w.WriteHeader(http.StatusConflict)
				_, _ = io.WriteString(w, "upstream conflict")
			}))
			defer upstream.Close()
			req := httptest.NewRequest(tc.method, tc.path, strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-Idempotency-Key", "retry-key")
			req.Header.Set("X-Request-Id", "request-key")
			req.Header.Set("X-Forwarded-Proto", "https")
			req.AddCookie(&http.Cookie{Name: subjectCookieName, Value: token})
			recorder := httptest.NewRecorder()
			v2TestRouter(upstream.URL).ServeHTTP(recorder, req)
			if recorder.Code != http.StatusConflict || recorder.Body.String() != "upstream conflict" {
				t.Fatalf("upstream failure changed: %d %s", recorder.Code, recorder.Body.String())
			}
			if recorder.Header().Get("Content-Type") != "application/pdf" {
				t.Fatal("upstream content type changed")
			}
			cookies := recorder.Result().Cookies()
			if len(cookies) != 1 || cookies[0].Value != token || !cookies[0].HttpOnly || !cookies[0].Secure {
				t.Fatalf("subject cookie lost protections: %v", cookies)
			}
			if strings.HasPrefix(tc.path, "/artifacts/") && recorder.Header().Get("Content-Disposition") == "" {
				t.Fatal("artifact download filename lost")
			}
		})
	}
}

func TestV2StreamFlushesBeforeUpstreamFinishes(t *testing.T) {
	t.Setenv("VERCEL", "")
	first := "data: {\"type\":\"heartbeat\",\"sequence\":1}\n\n"
	last := "data: {\"type\":\"done\",\"sequence\":2}\n\n"
	release := make(chan struct{}, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, first)
		w.(http.Flusher).Flush()
		select {
		case <-release:
			_, _ = io.WriteString(w, last)
		case <-r.Context().Done():
		}
	}))
	defer upstream.Close()
	gateway := httptest.NewServer(v2TestRouter(upstream.URL))
	defer gateway.Close()
	defer close(release)
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Post(gateway.URL+"/chat", "application/json", strings.NewReader("{}"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	reader := bufio.NewReader(resp.Body)
	line, err := reader.ReadString('\n')
	if err != nil || line != strings.SplitAfter(first, "\n")[0] {
		t.Fatalf("first frame buffered or changed: %q, %v", line, err)
	}
	// The upstream remains blocked until the client has received its frame.
	release <- struct{}{}
	tail, err := io.ReadAll(reader)
	if err != nil || line+string(tail) != first+last {
		t.Fatalf("stream envelopes changed: %q, %v", line+string(tail), err)
	}
	if resp.Header.Get("X-Accel-Buffering") != "no" || resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatal("streaming headers missing")
	}
}

func TestV2ClientDisconnectCancelsUpstream(t *testing.T) {
	t.Setenv("VERCEL", "")
	cancelled := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: heartbeat\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		close(cancelled)
	}))
	defer upstream.Close()
	gateway := httptest.NewServer(v2TestRouter(upstream.URL))
	defer gateway.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, "POST", gateway.URL+"/chat", strings.NewReader("{}"))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	_, err = bufio.NewReader(resp.Body).ReadString('\n')
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	_ = resp.Body.Close()
	select {
	case <-cancelled:
	case <-time.After(3 * time.Second):
		t.Fatal("client disconnect did not cancel upstream")
	}
}

func TestV2EOFNeverFabricatesTerminal(t *testing.T) {
	t.Setenv("VERCEL", "")
	for _, status := range []int{http.StatusOK, http.StatusServiceUnavailable} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			body := "data: {\"type\":\"heartbeat\"}\n\n"
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(status)
				_, _ = io.WriteString(w, body)
			}))
			defer upstream.Close()
			recorder := httptest.NewRecorder()
			v2TestRouter(upstream.URL).ServeHTTP(recorder, httptest.NewRequest("POST", "/chat", strings.NewReader("{}")))
			if recorder.Code != status || recorder.Body.String() != body {
				t.Fatalf("EOF was rewritten: %d %q", recorder.Code, recorder.Body.String())
			}
			cookies := recorder.Result().Cookies()
			if len(cookies) != 1 || !validSubjectToken(cookies[0].Value) || !cookies[0].HttpOnly {
				t.Fatal("missing durable subject identity")
			}
		})
	}
}
