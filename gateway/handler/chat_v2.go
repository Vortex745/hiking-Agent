package handler

import (
	"bufio"
	"bytes"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/gin-gonic/gin"
)

// ChatV2Handler proxies Chat v2 requests to the AI service.
//
// Ticket 07 — true streaming proxy for explicit v2 traffic. Unlike the
// v1 SSE proxy, v2 NEVER buffers on Vercel: the handler rejects v2 SSE
// with HTTP 503 when VERCEL is set, because durable run state cannot
// live on ephemeral serverless storage (spec AC4). Non-streaming v2
// endpoints (snapshot/cancel/events) are plain pass-through proxies.
//
// All v1 routes and handlers remain unchanged — v2 is additive.
type ChatV2Handler struct {
	aiServiceURL string
	selector     *RuntimeSelector
}

type RuntimeSelector struct {
	Secret      string
	Canary      string
	Force       string
	Environment string
}

func (s RuntimeSelector) Select(conversationID string) string {
	if s.Force == "v1" {
		return "v1"
	}
	trusted := s.Environment == "staging" || s.Environment == "trusted" || s.Environment == "internal"
	if s.Force == "v2" {
		if trusted {
			return "v2"
		}
		return "v1"
	}
	if s.Canary == "internal" {
		if trusted {
			return "v2"
		}
		return "v1"
	}

	percentages := map[string]uint64{
		"0": 0, "1": 1, "5": 5, "25": 25, "50": 50, "100": 100,
	}
	percentage, ok := percentages[s.Canary]
	if !ok || s.Secret == "" || conversationID == "" {
		return "v1"
	}
	mac := hmac.New(sha256.New, []byte(s.Secret))
	_, _ = mac.Write([]byte(conversationID))
	bucket := binary.BigEndian.Uint64(mac.Sum(nil)[:8]) % 100
	if bucket < percentage {
		return "v2"
	}
	return "v1"
}

const subjectCookieName = "hiking_subject"
const subjectCookieMaxAge = 365 * 24 * 60 * 60

func ensureSubjectCookie(c *gin.Context) bool {
	if cookie, err := c.Cookie(subjectCookieName); err == nil && validSubjectToken(cookie) {
		return true
	}

	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "subject identity unavailable"})
		return false
	}
	token := base64.RawURLEncoding.EncodeToString(raw)
	replaceRequestCookie(c.Request, subjectCookieName, token)
	http.SetCookie(c.Writer, &http.Cookie{
		Name: subjectCookieName, Value: token, Path: "/",
		MaxAge: subjectCookieMaxAge, HttpOnly: true, Secure: requestIsHTTPS(c),
		SameSite: http.SameSiteLaxMode,
	})
	return true
}

func requestIsHTTPS(c *gin.Context) bool {
	return c.Request.TLS != nil || strings.EqualFold(c.GetHeader("X-Forwarded-Proto"), "https")
}

func validSubjectToken(token string) bool {
	raw, err := base64.RawURLEncoding.DecodeString(token)
	return err == nil && len(token) == 43 && len(raw) == 32
}

func replaceRequestCookie(req *http.Request, name, value string) {
	cookies := req.Cookies()
	req.Header.Del("Cookie")
	for _, cookie := range cookies {
		if cookie.Name != name {
			req.AddCookie(cookie)
		}
	}
	req.AddCookie(&http.Cookie{Name: name, Value: value})
}

func forwardResponseCookies(c *gin.Context, resp *http.Response) {
	for _, cookie := range resp.Cookies() {
		if cookie.Name == subjectCookieName {
			cookie.HttpOnly = true
			cookie.Secure = requestIsHTTPS(c)
			cookie.SameSite = http.SameSiteLaxMode
			cookie.Path = "/"
			if cookie.MaxAge == 0 {
				cookie.MaxAge = subjectCookieMaxAge
			}
		}
		http.SetCookie(c.Writer, cookie)
	}
}

// NewChatV2Handler creates a new ChatV2Handler.
func NewChatV2Handler(aiServiceURL string, selectors ...RuntimeSelector) *ChatV2Handler {
	handler := &ChatV2Handler{aiServiceURL: aiServiceURL}
	if len(selectors) > 0 {
		handler.selector = &selectors[0]
	}
	return handler
}

func (h *ChatV2Handler) Runtime(c *gin.Context) {
	conversationID := c.Param("conversation_id")
	if conversationID == "" || h.selector == nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "runtime selection unavailable"})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"conversation_id": conversationID,
		"runtime":         h.selector.Select(conversationID),
	})
}

// ChatV2SSE proxies a Chat v2 SSE run to the AI service with true
// streaming — each frame is flushed to the client as soon as the
// upstream emits it, never buffered until completion (spec AC2).
//
// Vercel rejection (spec AC4): when VERCEL env is set, the handler
// returns 503 immediately. Vercel Functions cannot host v2 streaming
// because the durable run state requires a long-lived Docker single
// writer, and buffering v2 frames would defeat the streaming contract.
//
// Upstream errors (spec AC3): non-2xx responses are passed through
// with their status code and body — the gateway never fabricates a
// success terminal event. Client disconnects propagate to the upstream
// via request context cancellation.
func (h *ChatV2Handler) ChatV2SSE(c *gin.Context) {
	if os.Getenv("VERCEL") != "" {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error": "Chat v2 streaming is not available on Vercel; use the long-lived Docker gateway.",
		})
		return
	}
	if h.selector != nil {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
			return
		}
		c.Request.Body = io.NopCloser(bytes.NewReader(body))
		var request struct {
			ConversationID string `json:"conversation_id"`
		}
		if json.Unmarshal(body, &request) != nil || request.ConversationID == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "conversation_id is required"})
			return
		}
		if h.selector.Select(request.ConversationID) != "v2" {
			c.JSON(http.StatusConflict, gin.H{
				"error": gin.H{"code": "runtime_v1_required", "retryable": true},
			})
			return
		}
	}
	if !ensureSubjectCookie(c) {
		return
	}

	url := fmt.Sprintf("%s/api/v2/chat/sse", h.aiServiceURL)
	proxyV2Stream(c, http.MethodPost, url, c.Request.Body)
}

// ChatV2Snapshot proxies GET /snapshot/:run_id to the AI service.
func (h *ChatV2Handler) ChatV2Snapshot(c *gin.Context) {
	if !ensureSubjectCookie(c) {
		return
	}
	runID := c.Param("run_id")
	url := fmt.Sprintf("%s/api/v2/chat/snapshot/%s", h.aiServiceURL, runID)
	proxyV2Request(c, http.MethodGet, url, nil)
}

// ChatV2Cancel proxies POST /cancel/:run_id to the AI service.
// The idempotency_key is in the request body; the gateway passes it
// through verbatim so the AI service can dedupe double-clicks.
func (h *ChatV2Handler) ChatV2Cancel(c *gin.Context) {
	if !ensureSubjectCookie(c) {
		return
	}
	runID := c.Param("run_id")
	url := fmt.Sprintf("%s/api/v2/chat/cancel/%s", h.aiServiceURL, runID)
	proxyV2Request(c, http.MethodPost, url, c.Request.Body)
}

// ChatV2Events proxies GET /events/:run_id?from_seq=N to the AI service.
// This is the reconnect endpoint — clients that lost the stream replay
// stored events from the last sequence they observed.
func (h *ChatV2Handler) ChatV2Events(c *gin.Context) {
	if !ensureSubjectCookie(c) {
		return
	}
	runID := c.Param("run_id")
	url := fmt.Sprintf("%s/api/v2/chat/events/%s", h.aiServiceURL, runID)
	if rawQuery := c.Request.URL.RawQuery; rawQuery != "" {
		url = url + "?" + rawQuery
	}
	proxyV2Request(c, http.MethodGet, url, nil)
}

func (h *ChatV2Handler) MemorySettings(c *gin.Context) {
	if !ensureSubjectCookie(c) {
		return
	}
	url := fmt.Sprintf("%s/api/v2/memory/settings", h.aiServiceURL)
	proxyV2Request(c, c.Request.Method, url, c.Request.Body)
}

func (h *ChatV2Handler) Memories(c *gin.Context) {
	if !ensureSubjectCookie(c) {
		return
	}
	url := fmt.Sprintf("%s/api/v2/memories", h.aiServiceURL)
	if rawQuery := c.Request.URL.RawQuery; rawQuery != "" {
		url += "?" + rawQuery
	}
	proxyV2Request(c, c.Request.Method, url, c.Request.Body)
}

func (h *ChatV2Handler) MemoryItem(c *gin.Context) {
	if !ensureSubjectCookie(c) {
		return
	}
	url := fmt.Sprintf("%s/api/v2/memories/%s", h.aiServiceURL, c.Param("memory_id"))
	proxyV2Request(c, c.Request.Method, url, c.Request.Body)
}

func (h *ChatV2Handler) MemoryCandidateDecision(c *gin.Context) {
	if !ensureSubjectCookie(c) {
		return
	}
	url := fmt.Sprintf(
		"%s/api/v2/memory/candidates/%s/decision",
		h.aiServiceURL,
		c.Param("candidate_id"),
	)
	proxyV2Request(c, http.MethodPost, url, c.Request.Body)
}

func (h *ChatV2Handler) MemoryExport(c *gin.Context) {
	if !ensureSubjectCookie(c) {
		return
	}
	url := fmt.Sprintf("%s/api/v2/memories/export", h.aiServiceURL)
	proxyV2Request(c, http.MethodGet, url, nil)
}

func (h *ChatV2Handler) Artifact(c *gin.Context) {
	if !ensureSubjectCookie(c) {
		return
	}
	url := fmt.Sprintf("%s/api/v2/artifacts/%s", h.aiServiceURL, c.Param("artifact_id"))
	proxyV2Request(c, http.MethodGet, url, nil)
}

func (h *ChatV2Handler) RotateSubject(c *gin.Context) {
	if !ensureSubjectCookie(c) {
		return
	}
	url := fmt.Sprintf("%s/api/v2/subject/rotate", h.aiServiceURL)
	proxyV2Request(c, http.MethodPost, url, c.Request.Body)
}

// proxyV2Stream forwards a v2 SSE stream with line-by-line flushing.
//
// Differences from the v1 proxyStreamRequest:
//   - No Vercel buffering fallback (v2 rejects Vercel at the handler).
//   - Cookies and select headers are forwarded so idempotency keys and
//     auth state survive the proxy hop (spec AC1).
//   - Upstream non-2xx is passed through with body intact — no fake
//     terminal event is synthesized (spec AC3).
func proxyV2Stream(c *gin.Context, method, url string, body io.Reader) {
	req, err := http.NewRequestWithContext(c.Request.Context(), method, url, body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Forward Content-Type so the AI service parses the JSON body.
	if ct := c.Request.Header.Get("Content-Type"); ct != "" {
		req.Header.Set("Content-Type", ct)
	}
	// Forward cookies so auth/session state survives the proxy hop.
	if cookie := c.Request.Header.Get("Cookie"); cookie != "" {
		req.Header.Set("Cookie", cookie)
	}
	// Forward idempotency-relevant headers. X-Idempotency-Key is the
	// canonical v2 header; X-Request-Id is a common alias.
	for _, h := range []string{"X-Idempotency-Key", "X-Request-Id"} {
		if v := c.Request.Header.Get(h); v != "" {
			req.Header.Set(h, v)
		}
	}
	req.Header.Set("Accept", "text/event-stream")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		// Upstream unreachable — 503, no fake terminal (spec AC3).
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "AI service unreachable"})
		return
	}
	defer resp.Body.Close()
	forwardResponseCookies(c, resp)

	// Upstream non-2xx: pass through status + body, no synthesis (AC3).
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		respBody, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": readErr.Error()})
			return
		}
		ct := resp.Header.Get("Content-Type")
		if ct == "" {
			ct = "application/json"
		}
		c.Data(resp.StatusCode, ct, respBody)
		return
	}

	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "streaming not supported"})
		return
	}

	// SSE headers. X-Accel-Buffering: no tells reverse proxies (nginx)
	// not to buffer the stream — critical for true unbuffered delivery.
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.Header().Set("X-Accel-Buffering", "no")
	c.Writer.WriteHeader(resp.StatusCode)

	// Line-by-line read + flush. Each SSE frame is terminated by \n\n;
	// reading by line and flushing immediately delivers frames to the
	// browser as the upstream emits them, not after the run completes.
	reader := bufio.NewReader(resp.Body)
	for {
		line, readErr := reader.ReadString('\n')
		if line != "" {
			if _, werr := fmt.Fprint(c.Writer, line); werr != nil {
				// Client disconnected — stop reading upstream.
				return
			}
			flusher.Flush()
		}
		if readErr != nil {
			return
		}
	}
}

// proxyV2Request forwards a non-streaming v2 request (snapshot/cancel/
// events). Cookies and Content-Type are forwarded; the response is
// passed through verbatim including status code.
func proxyV2Request(c *gin.Context, method, url string, body io.Reader) {
	req, err := http.NewRequestWithContext(c.Request.Context(), method, url, body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if ct := c.Request.Header.Get("Content-Type"); ct != "" {
		req.Header.Set("Content-Type", ct)
	}
	if cookie := c.Request.Header.Get("Cookie"); cookie != "" {
		req.Header.Set("Cookie", cookie)
	}
	for _, h := range []string{"X-Idempotency-Key", "X-Request-Id"} {
		if v := c.Request.Header.Get(h); v != "" {
			req.Header.Set(h, v)
		}
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "AI service unreachable"})
		return
	}
	defer resp.Body.Close()
	forwardResponseCookies(c, resp)

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	ct := resp.Header.Get("Content-Type")
	if ct == "" {
		ct = "application/json"
	}
	if disposition := resp.Header.Get("Content-Disposition"); disposition != "" {
		c.Header("Content-Disposition", disposition)
	}
	c.Data(resp.StatusCode, ct, respBody)
}
