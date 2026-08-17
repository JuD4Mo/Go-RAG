package rag

import (
	"context"
	"fmt"

	"github.com/JuD4Mo/rag-course/llm"
	"github.com/JuD4Mo/rag-course/vector"
)

const defaultTopK = 5

type Options struct {
	Topk     int
	Rewriter *Rewriter
}

type Retriever struct {
	embedder llm.Embedder
	store    vector.Store
	rewriter *Rewriter
	topK     int
}

func New(embedder llm.Embedder, store vector.Store, opts Options) *Retriever {
	topK := opts.Topk
	if topK <= 0 {
		topK = defaultTopK
	}

	return &Retriever{
		embedder: embedder,
		store:    store,
		rewriter: opts.Rewriter,
		topK:     topK,
	}
}

func (r *Retriever) Retrieve(ctx context.Context, history []llm.Message) (string, error) {
	query := r.buildQuery(ctx, history)
	if query == "" {
		return "", nil
	}

	vecs, err := r.embedder.Embed(ctx, []string{query})
	if err != nil {
		return "", fmt.Errorf("embed query: %w", err)
	}

	if len(vecs) == 0 {
		return "", nil
	}

	hits, err := r.store.Query(ctx, vecs[0], r.topK)
	if err != nil {
		return "", fmt.Errorf("vector query: %w", err)
	}

	if len(hits) == 0 {
		return "", nil
	}

	return formatContext(hits), nil
}

func (r *Retriever) buildQuery(ctx context.Context, history []llm.Message) string {
	if r.rewriter != nil {
		if q, err := r.rewriter.Rewrite(ctx, history); err == nil && q != "" {
			return q
		}
	}

	return lastUserMessage(history)
}

func lastUserMessage(history []llm.Message) string {
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == "user" {
			return history[i].Content
		}
	}
	return ""
}
