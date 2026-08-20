package ingest

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/JuD4Mo/rag-course/llm"
	"github.com/JuD4Mo/rag-course/vector"
)

const ImagePathPrefix = "/images/"

var imageExtensions = map[string]bool{
	".png":  true,
	".jpg":  true,
	".jpeg": true,
	".webp": true,
	".gif":  true,
}

func IsImage(name string) bool {
	return imageExtensions[strings.ToLower(filepath.Ext(name))]
}

func ProcessImage(ctx context.Context, name, description string, opts Options, embedder llm.Embedder, store vector.Store) (int, error) {
	if embedder == nil {
		return 0, errors.New("embedder is required")
	}
	if store == nil {
		return 0, errors.New("vector store is required")
	}

	base := filepath.Base(name)
	if !IsImage(base) {
		return 0, errors.New("unsupported image format")
	}

	desc := strings.TrimSpace(description)
	if desc == "" {
		return 0, errors.New("description is required")
	}

	size := opts.ChunkSize
	if size <= 0 {
		size = defaultChunkSize
	}

	overlap := opts.ChunkOverlap
	if size <= 0 {
		size = defaultChunkOverlap
	}

	chunks := chunk(desc, size, overlap)
	if len(chunks) == 0 {
		return 0, errors.New("no chunks produced")
	}

	vectors, err := embedder.Embed(ctx, chunks)
	if err != nil {
		return 0, fmt.Errorf("embed: %w", err)
	}

	if len(vectors) != len(chunks) {
		return 0, fmt.Errorf("embed: got %d vectors for %d chunks", len(vectors), len(chunks))
	}

	if err := store.DeleteBySource(ctx, base); err != nil {
		return 0, fmt.Errorf("clear previous chunks: %w", err)
	}

	ingestedAt := time.Now().UTC().Format(time.RFC3339)
	imagePath := ImagePathPrefix + base
	docs := make([]vector.Document, len(chunks))

	for i, c := range chunks {
		docs[i] = vector.Document{
			ID:      fmt.Sprintf("%s#%d", base, i),
			Content: c,
			Metadata: map[string]string{
				"source":      base,
				"type":        "image",
				"image_path":  imagePath,
				"chunk_index": strconv.Itoa(i),
				"chunks":      strconv.Itoa(len(chunks)),
				"ingested_at": ingestedAt,
			},
			Embedding: vectors[i],
		}
	}

	if err := store.Upsert(ctx, docs); err != nil {
		return 0, err
	}

	return len(chunks), nil
}
