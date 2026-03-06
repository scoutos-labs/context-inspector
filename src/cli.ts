#!/usr/bin/env bun
import { writeFileSync, readFileSync } from "fs";
import { runAblation, createBackend } from "./engine.ts";
import { parseAgentContext } from "./parser.ts";
import type { RunConfig, RunResult, AblationResult } from "./types.ts";

// ─── Help ─────────────────────────────────────────────────────────────────────

const USAGE = `
context-inspector — ablation-based context scoring

COMMANDS:
  run       Run ablation against a probe prompt
  report    Pretty-print results from a run output file

RUN OPTIONS:
  --file <id>=<label>=<path>   Add a file as a context segment (repeatable)
  --split <id>                 Split this segment's file by ## H2 headers
                               for granular section-level ablation (repeatable)
  --conversation <path>        Add a conversation JSON as a segment
                               Format: {messages: [{role, content}]}
  --probe <text>               Probe prompt to run against the context
  --backend <ollama|anthropic|openai>  LLM backend (default: ollama)
  --model <name>               Completion model
                               Defaults: ollama=qwen3:32b, openai=gpt-4o-mini,
                                         anthropic=claude-haiku-4-5-20251001
  --embed-model <name>         Embedding model
                               Defaults: ollama=nomic-embed-text,
                                         openai=text-embedding-3-small
  --scorer <embedding|judge|both>  Scoring strategy (default: both)
  --judge-top-n <n>            Run judge on top N segments only (default: 3)
  --ollama-url <url>           Ollama base URL (default: http://localhost:11434)
  --api-base-url <url>         Base URL for OpenAI-compatible providers
                               (e.g. http://localhost:1234/v1 for LM Studio)
  --out <path>                 Output file for JSON results (default: results.json)
  --quiet                      Suppress progress logs

ENVIRONMENT VARIABLES:
  ANTHROPIC_API_KEY            Required when --backend anthropic
  OPENAI_API_KEY               Required when --backend openai

REPORT OPTIONS:
  <file>               Path to results JSON (default: results.json)
  --no-explanations    Hide judge explanations

EXAMPLES:
  # Ollama (local) — two files
  context-inspector run \\
    --file sys=System=./system-prompt.md \\
    --file ctx=Context=./context.md \\
    --probe "What should I do next?" \\
    --out results.json

  # Ollama — granular ablation: split a large file by section
  context-inspector run \\
    --file memory=Memory=./memory.md \\
    --split memory \\
    --probe "What should I do next?" \\
    --scorer embedding

  # OpenAI — three files, judge-only scoring
  context-inspector run \\
    --file sys=System=./system.md \\
    --file rules=Rules=./rules.md \\
    --file ctx=Context=./context.md \\
    --probe "Summarize the current situation." \\
    --backend openai \\
    --scorer judge

  # OpenAI-compatible local server (LM Studio, Ollama OpenAI API, etc.)
  context-inspector run \\
    --file sys=System=./system.md \\
    --probe "What's the plan?" \\
    --backend openai \\
    --api-base-url http://localhost:1234/v1

  context-inspector report results.json
`.trim();

// ─── Arg parsing ──────────────────────────────────────────────────────────────

function parseArgs(argv: string[]): Record<string, string | string[] | boolean> {
  const args: Record<string, string | string[] | boolean> = {};
  const MULTI_FLAGS = new Set(["file", "split"]);
  let i = 0;
  while (i < argv.length) {
    const arg = argv[i];
    if (arg.startsWith("--")) {
      const key = arg.slice(2);
      const next = argv[i + 1];
      if (!next || next.startsWith("--")) {
        args[key] = true;
        i++;
      } else if (MULTI_FLAGS.has(key)) {
        const existing = args[key];
        args[key] = [...(Array.isArray(existing) ? existing : []), next];
        i += 2;
      } else {
        args[key] = next;
        i += 2;
      }
    } else {
      args["_positional"] = arg;
      i++;
    }
  }
  return args;
}

// ─── Report formatter ─────────────────────────────────────────────────────────

function renderBar(score: number, width = 20): string {
  const filled = Math.round(score * width);
  return "█".repeat(filled) + "░".repeat(width - filled);
}

function scoreLabel(score: number): string {
  if (score >= 0.6) return "CRITICAL";
  if (score >= 0.3) return "PARTIAL ";
  return "LOW     ";
}

function formatReport(result: RunResult, showExplanations: boolean): string {
  const lines: string[] = [];

  lines.push("═".repeat(70));
  lines.push("  Context Inspector — Ablation Report");
  lines.push("═".repeat(70));
  lines.push(`  Probe:   ${result.probePrompt}`);
  lines.push(`  Model:   ${result.model}`);
  lines.push(`  Scorer:  ${result.scorer}`);
  lines.push(`  Time:    ${result.timestamp}`);
  lines.push("─".repeat(70));
  lines.push("  SEGMENT RANKING (most impactful → least)");
  lines.push("─".repeat(70));

  for (let i = 0; i < result.results.length; i++) {
    const r = result.results[i];
    const bar = renderBar(r.behavioralScore);
    const label = scoreLabel(r.behavioralScore);
    lines.push(`  ${(i + 1).toString().padStart(2)}. ${r.label.padEnd(28)} ${bar} ${label} ${(r.behavioralScore * 100).toFixed(1)}%`);
    if (r.embeddingScore !== undefined) {
      lines.push(`      embedding delta:  ${(r.embeddingScore * 100).toFixed(1)}%`);
    }
    if (r.judgeScore !== undefined) {
      lines.push(`      judge score:      ${(r.judgeScore * 100).toFixed(1)}%`);
    }
    if (showExplanations && r.judgeExplanation) {
      lines.push(`      explanation:      ${r.judgeExplanation}`);
    }
    lines.push("");
  }

  lines.push("─".repeat(70));
  lines.push("  BASELINE RESPONSE (full context)");
  lines.push("─".repeat(70));
  const preview = result.originalResponse.slice(0, 400);
  lines.push(`  ${preview}${result.originalResponse.length > 400 ? "..." : ""}`);
  lines.push("═".repeat(70));

  return lines.join("\n");
}

// ─── Commands ─────────────────────────────────────────────────────────────────

async function cmdRun(argv: string[]): Promise<void> {
  const args = parseArgs(argv);

  const probe = args["probe"] as string | undefined;
  if (!probe) {
    console.error("Error: --probe is required");
    process.exit(1);
  }

  // Parse --file specs
  const splitIds = new Set(
    (args["split"] as string[] | undefined) ?? [],
  );

  const fileSpecs = ((args["file"] as string[] | undefined) ?? []).map((spec) => {
    const parts = spec.split("=");
    if (parts.length < 3) {
      console.error(`Invalid --file spec: "${spec}". Expected: id=label=path`);
      process.exit(1);
    }
    const id = parts[0];
    return {
      id,
      label: parts[1],
      path: parts.slice(2).join("="),
      split: splitIds.has(id),
    };
  });

  // Validate --split references a known file id
  for (const id of splitIds) {
    if (!fileSpecs.some((f) => f.id === id)) {
      console.error(`Error: --split "${id}" does not match any --file id`);
      process.exit(1);
    }
  }

  const segments = parseAgentContext({
    files: fileSpecs,
    conversationPath: args["conversation"] as string | undefined,
  });

  const scorer = (args["scorer"] as string | undefined) ?? "both";
  if (!["embedding", "judge", "both"].includes(scorer)) {
    console.error(`Invalid --scorer: ${scorer}. Use: embedding | judge | both`);
    process.exit(1);
  }

  const backend = (args["backend"] as string | undefined) ?? "ollama";
  if (!["ollama", "anthropic", "openai"].includes(backend)) {
    console.error(`Invalid --backend: ${backend}. Use: ollama | anthropic | openai`);
    process.exit(1);
  }

  const config: RunConfig = {
    segments,
    probePrompt: probe,
    backend: backend as "ollama" | "anthropic" | "openai",
    model: args["model"] as string | undefined,
    embedModel: args["embed-model"] as string | undefined,
    scorer: scorer as "embedding" | "judge" | "both",
    judgeTopN: parseInt((args["judge-top-n"] as string | undefined) ?? "3", 10),
    ollamaBaseUrl: args["ollama-url"] as string | undefined,
    apiBaseUrl: args["api-base-url"] as string | undefined,
  };

  const quiet = args["quiet"] === true;
  const onProgress = quiet ? undefined : (msg: string) => console.log(msg);

  console.log(`Running ablation on ${segments.length} segments...`);
  const backendInstance = createBackend(config);
  const result = await runAblation(config, backendInstance, onProgress);

  const outPath = (args["out"] as string | undefined) ?? "results.json";
  writeFileSync(outPath, JSON.stringify(result, null, 2));
  console.log(`\nResults saved to ${outPath}`);

  // Always show a quick summary
  console.log("\nTop segments by impact:");
  for (const r of result.results.slice(0, 5)) {
    console.log(`  ${(r.behavioralScore * 100).toFixed(1)}% — ${r.label}`);
  }
}

function cmdReport(argv: string[]): void {
  const args = parseArgs(argv);
  const filePath = (args["_positional"] as string | undefined) ?? "results.json";
  const showExplanations = args["no-explanations"] !== true;

  let result: RunResult;
  try {
    result = JSON.parse(readFileSync(filePath, "utf-8")) as RunResult;
  } catch (e) {
    console.error(`Could not read results file ${filePath}: ${(e as Error).message}`);
    process.exit(1);
  }

  console.log(formatReport(result, showExplanations));
}

// ─── Entrypoint ───────────────────────────────────────────────────────────────

const [, , command, ...rest] = process.argv;

if (!command || command === "--help" || command === "-h") {
  console.log(USAGE);
  process.exit(0);
}

if (command === "run") {
  cmdRun(rest).catch((e) => {
    console.error("Error:", (e as Error).message);
    process.exit(1);
  });
} else if (command === "report") {
  cmdReport(rest);
} else {
  console.error(`Unknown command: ${command}`);
  console.log(USAGE);
  process.exit(1);
}
