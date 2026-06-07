/**
 * LLM backend dispatcher.
 *
 * All call sites (pipeline / enrich / trading-commentary) import `runLlm`
 * from this module instead of binding to a specific backend. The actual
 * backend is selected at runtime by the LLM_BACKEND environment variable:
 *
 *   LLM_BACKEND=claude-cli   (default; uses local Claude Code CLI, Max billing)
 *   LLM_BACKEND=anthropic    (Anthropic Messages API)
 *   LLM_BACKEND=openai       (OpenAI Chat Completions)
 *   LLM_BACKEND=deepseek     (DeepSeek, OpenAI-compatible)
 *   LLM_BACKEND=minimax      (MiniMax, OpenAI-compatible)
 *
 * Per-backend config (API keys, models, base URLs) lives in .env.local.
 * See .env.example for the full list.
 */

import { CLAUDE_MODEL, runClaudeCli } from "./backends/claude-cli";
import { anthropicModel, runAnthropic } from "./backends/anthropic";
import {
  PRESETS,
  openaiCompatModel,
  runOpenAICompat,
} from "./backends/openai-compat";

export interface LlmRunOptions {
  systemPrompt: string;
  userPrompt: string;
  timeoutMs?: number;
  /**
   * Override the active backend for this specific call. Used by the
   * pipeline's fallback retry to swap in a different model when the
   * primary rejects the request (e.g. content moderation, rate limit).
   * If omitted, falls back to the LLM_BACKEND env var.
   */
  backendOverride?: LlmBackendId;
}

export interface LlmRunResult {
  text: string;
  durationMs: number;
}

export type LlmBackendId =
  | "claude-cli"
  | "anthropic"
  | "openai"
  | "deepseek"
  | "minimax";

const VALID_BACKENDS: ReadonlySet<LlmBackendId> = new Set([
  "claude-cli",
  "anthropic",
  "openai",
  "deepseek",
  "minimax",
]);

export function getBackend(): LlmBackendId {
  const raw = (process.env.LLM_BACKEND?.trim() || "claude-cli").toLowerCase();
  if (!VALID_BACKENDS.has(raw as LlmBackendId)) {
    throw new Error(
      `Unknown LLM_BACKEND='${raw}'. Valid values: ${[...VALID_BACKENDS].join(", ")}`,
    );
  }
  return raw as LlmBackendId;
}

function getActiveModel(backend: LlmBackendId): string {
  switch (backend) {
    case "claude-cli":
      return CLAUDE_MODEL;
    case "anthropic":
      return anthropicModel();
    case "openai":
    case "deepseek":
    case "minimax":
      return openaiCompatModel(PRESETS[backend]);
  }
}

/**
 * Resolve the effective backend for a call: explicit override wins, else the
 * LLM_BACKEND env var. Exported so the pipeline can log the actual backend
 * that produced a report (which may differ from the configured one if a
 * fallback kicked in).
 */
export function resolveBackend(override?: LlmBackendId): LlmBackendId {
  return override ?? getBackend();
}

/** A short tag suitable for embedding in report JSON: "<backend>-<model>" */
export function getModelTag(backendOverride?: LlmBackendId): string {
  const backend = resolveBackend(backendOverride);
  return `${backend}-${getActiveModel(backend)}`;
}

export async function runLlm(opts: LlmRunOptions): Promise<LlmRunResult> {
  const backend = resolveBackend(opts.backendOverride);
  switch (backend) {
    case "claude-cli":
      return runClaudeCli(opts);
    case "anthropic":
      return runAnthropic(opts);
    case "openai":
    case "deepseek":
    case "minimax":
      return runOpenAICompat(opts, PRESETS[backend]);
  }
}

