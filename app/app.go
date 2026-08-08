package app

import (
	"context"

	"github.com/JuD4Mo/rag-course/chat"
	"github.com/JuD4Mo/rag-course/config"
	"github.com/JuD4Mo/rag-course/llm"
)

func Run(ctx context.Context, cfg config.Config) error {
	client := llm.New(cfg)
	return chat.RunREPL(ctx, client, chat.Options{
		SystemPromptFile: cfg.SystemPromptFile,
	})
}
