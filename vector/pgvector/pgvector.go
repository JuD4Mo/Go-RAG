package pgvector

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/JuD4Mo/rag-course/vector"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	pgxvec "github.com/pgvector/pgvector-go/pgx"
)

type Options struct {
	DSN          string
	EmbeddingDim int
}

type Store struct {
	pool *pgxpool.Pool
}

func New(ctx context.Context, opts Options) (*Store, error) {
	if opts.DSN == "" {
		return nil, errors.New("pgvector: DSN is required")
	}

	if opts.EmbeddingDim <= 0 {
		return nil, errors.New("pgvector: EmbeddingDim must be > 0")
	}

	cfg, err := pgxpool.ParseConfig(opts.DSN)
	if err != nil {
		return nil, fmt.Errorf("parse DSN: %w", err)
	}

	if err := ensureExtension(ctx, opts.DSN); err != nil {
		return nil, fmt.Errorf("install extension: %w", err)
	}

	cfg.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		return pgxvec.RegisterTypes(ctx, conn)
	}

	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("connect: %w", err)
	}

	s := &Store{
		pool: pool,
	}

	if err := s.migrate(ctx, opts.EmbeddingDim); err != nil {
		pool.Close()
		return nil, fmt.Errorf("migrate: %w", err)
	}

	return s, nil
}

func ensureExtension(ctx context.Context, dsn string) error {
	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		return err
	}

	defer conn.Close(ctx)

	_, err = conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector")

	return err
}

func (s *Store) migrate(ctx context.Context, dim int) error {
	stmts := []string{
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS documents (
		id TEXT PRIMARY KEY,
		content TEXT NOT NULL,
		metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
		embedding vector(%d) NOT NULL,
		created_at TIMESTAMPZ NOT NULL DEFAULT now())`, dim),
		`CREATE INDEX IF NOT EXISTS documents_embedding_idx
			ON documents USING hnsw (embedding vector_cosine_ops)`,
	}

	for _, q := range stmts {
		if _, err := s.pool.Exec(ctx, q); err != nil {
			return fmt.Errorf("exec %q: %w", firstLine(q), err)
		}
	}

	return nil
}

func firstLine(s string) string {
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			return s[:i]
		}
	}
	return s
}

func (s *Store) Upsert(ctx context.Context, docs []vector.Document) error {
	if len(docs) == 0 {
		return nil
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	const stmt = `
		INSERT INTO documents (id, content, metadata, embedding)
		VALUES ($1, $2, $3, $4)
		ON CONFLICT (id) DO UPDATE SET
			content = EXCLUDED.content,
			metadata = EXCLUDED.metadata,
			embedding = EXCLUDED.embedding
	`

	for _, d := range docs {
		meta, err := marshalMetadata(d.Metadata)
		if err != nil {
			return fmt.Errorf("metadata for %s: %w", d.ID, err)
		}

		if _, err := tx.Exec(ctx, stmt, d.ID, d.Content, meta, pgvector.NewVector(d.Embedding)); err != nil {
			return fmt.Errorf("upsert: %s: %w", d.ID, err)
		}

	}

	return tx.Commit(ctx)
}

func marshalMetadata(m map[string]string) ([]byte, error) {
	if len(m) == 0 {
		return []byte("{}"), nil
	}

	return json.Marshal(m)
}

func unmarshalMetadata(raw []byte, dst *map[string]string) error {
	if len(raw) == 0 {
		*dst = nil
		return nil
	}

	return json.Unmarshal(raw, dst)
}

func (s *Store) Query(ctx context.Context, embedding []float32, topK int) ([]vector.Result, error) {
	if topK <= 0 {
		return nil, nil
	}

	const stmt = `
		select id, content, metadata, embedding <=> $1 as distance
		from documents
		order by embedding <=> $1
		limit $2
	`

	rows, err := s.pool.Query(ctx, stmt, pgvector.NewVector(embedding), topK)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []vector.Result
	for rows.Next() {
		var (
			r        vector.Result
			metaRaw  []byte
			distance float64
		)

		if err := rows.Scan(&r.ID, &r.Content, &metaRaw, &distance); err != nil {
			return nil, err
		}
		if err := unmarshalMetadata(metaRaw, &r.Metadata); err != nil {
			return nil, fmt.Errorf("metadata for %s: %w", r.ID, err)
		}

		r.Score = float32(1 - distance)
		results = append(results, r)
	}

	return results, rows.Err()
}
