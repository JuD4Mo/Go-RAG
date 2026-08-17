package web

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io/fs"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/JuD4Mo/rag-course/llm"
	"github.com/JuD4Mo/rag-course/rag"
	"github.com/JuD4Mo/rag-course/vector"
	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

//go:embed templates/*.gohtml
var templatesFS embed.FS

type Options struct {
	Addr             string
	SystemPromptFile string
	Title            string
	Store            vector.Store
	ProcessedDir     string
	ImagesDir        string
}

type Server struct {
	client       *llm.Client
	embedder     *llm.Client
	retriever    *rag.Retriever
	store        vector.Store
	processedDir string
	imagesDir    string
	tpl          *template.Template
	system       string
	title        string
}

func New(client, embedder *llm.Client, retriever *rag.Retriever, opts Options) (*Server, error) {
	tpl, err := template.ParseFS(templatesFS, "templates/*.gohtml")
	if err != nil {
		return nil, fmt.Errorf("parse templates: %w", err)
	}

	title := opts.Title
	if title == "" {
		title = "RAG Chat"
	}

	return &Server{
		client:       client,
		embedder:     embedder,
		retriever:    retriever,
		store:        opts.Store,
		processedDir: opts.ProcessedDir,
		imagesDir:    opts.ImagesDir,
		tpl:          tpl,
		system:       readSystemPrompt(opts.SystemPromptFile),
		title:        title,
	}, nil
}

func (s *Server) Routes() http.Handler {
	r := chi.NewRouter()
	r.Use(middleware.Recoverer)

	r.Get("/chat", s.handleChatPage)
	r.Post("/api/chat/stream", s.handleChatStream)

	return r
}

type chatRequest struct {
	Messages []llm.Message `json:"messages"`
}

func (s *Server) handleChatStream(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalud json: "+err.Error(), http.StatusBadRequest)
		return
	}

	if len(req.Messages) == 0 {
		http.Error(w, "messages must not be empty", http.StatusBadRequest)
		return
	}

	if last := req.Messages[len(req.Messages)-1]; last.Role != "user" {
		http.Error(w, "last message must be from user", http.StatusBadRequest)
		return
	}

	history := req.Messages
	if s.system != "" {
		history = append([]llm.Message{{
			Role:    "system",
			Content: s.system,
		}}, history...)
	}

	turn := history
	if s.retriever != nil {
		ctxText, err := s.retriever.Retrieve(r.Context(), history)
		if err != nil {
			log.Printf("[web] retrieval error: %v", err)
		} else {
			turn = withInlineContext(history, ctxText)
		}
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	flusher.Flush()

	send := func(event, data string) {
		if event != "" {
			fmt.Fprintf(w, "event: %s\n", event)
		}
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	_, err := s.client.ChatStream(r.Context(), turn, func(delta string) {
		enc, _ := json.Marshal(delta)
		send("delta", string(enc))
	})
	if err != nil {
		enc, _ := json.Marshal(err.Error())
		send("error", string(enc))
		return
	}

	send("done", `""`)
}

func withInlineContext(history []llm.Message, contextText string) []llm.Message {
	if len(history) == 0 || contextText == "" {
		return history
	}

	last := history[len(history)-1]
	if last.Role != "user" {
		return history
	}

	out := make([]llm.Message, len(history))
	copy(out, history)

	out[len(out)-1] = llm.Message{
		Role:    "user",
		Content: contextText + "\n\n--- Question ---\n\n" + last.Content,
	}
	return out
}

func (s *Server) handleChatPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if err := s.tpl.ExecuteTemplate(w, "chat.gohtml", map[string]interface{}{
		"Title": s.title,
		//"CaptionEnabled": s.client.HasVision(),
	}); err != nil {
		log.Printf("[web] template error: %v", err)
	}
}

func (s *Server) Run(ctx context.Context, addr string) error {
	srv := &http.Server{
		Addr:              addr,
		Handler:           s.Routes(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	errChan := make(chan error, 1)

	go func() {
		err := srv.ListenAndServe()
		if errors.Is(err, http.ErrServerClosed) {
			errChan <- err
		}
		errChan <- nil
	}()

	select {
	case <-ctx.Done():
		shutDownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutDownCtx)
		return nil
	case err := <-errChan:
		return err
	}
}

func readSystemPrompt(path string) string {
	if path == "" {
		return ""
	}

	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return ""
	}

	return strings.TrimSpace(string(data))
}
