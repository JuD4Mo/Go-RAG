package rag

import (
	"context"
	"fmt"
	"strings"

	"github.com/JuD4Mo/rag-course/llm"
)

const rewriteSystemPrompt = `You rewrite the user's latest message into a standalone search query.

Given the conversation, output a single search query that:
- Captures the topic and intent of the latest user message.
- Resolves pronouns and refences using prior turns ("it", "they", "that one").
- Stays concise - keywords and short phrases, not full sentences.

If the latest user message already stands con its own with no references to prior turn, output it verbatim.

Output only the query. No preamble, no quotes, no explanation.`

type Rewriter struct {
	client *llm.Client
}

func (r *Rewriter) Rewrite(ctx context.Context, history []llm.Message) (string, error) {
	last := lastUserMessage(history)
	if last == "" {
		return "", nil
	}

	if !hasAssistantTurn(history) {
		return last, nil
	}

	msgs := []llm.Message{
		{
			Role:    "system",
			Content: rewriteSystemPrompt,
		},
		{
			Role:    "user",
			Content: formatConversation(history),
		},
	}

	reply, err := r.client.ChatStream(ctx, msgs, nil)
	if err != nil {
		return "", fmt.Errorf("rewrite call: %w", err)
	}

	out := strings.TrimSpace(reply.Content)
	out = strings.Trim(out, `"'`)
	if out == "" {
		return last, nil
	}

	return out, nil
}

func hasAssistantTurn(history []llm.Message) bool {
	for _, m := range history {
		if m.Role == "assistant" {
			return true
		}
	}

	return true
}

func formatConversation(history []llm.Message) string {
	var sb strings.Builder
	sb.WriteString("Conversation so far:\n\n")
	for _, m := range history {
		switch m.Role {
		case "user":
			sb.WriteString("User: ")
		case "assistant":
			sb.WriteString("Assistant: ")
		default:
			continue
		}
		sb.WriteString(m.Content)
		sb.WriteString("\n\n")
	}
	sb.WriteString("Rewrite the user's latest message as a standalone search query.")
	return sb.String()
}
