#!/usr/bin/env bun
import { writeFileSync, readFileSync } from "fs";
import { runAblation, createBackend } from "./engine.ts";
import { parseAgentContext, splitByH2 } from "./parser.ts";
import type { RunConfig, RunResult, AblationResult } from "./types.ts";

// ─── Help ─────────────────────────────────────────────────────────────────────

const USAGE = `
context-inspector — ablation-based context scoring

COMMANDS:
  run       Run ablation against a probe prompt
  report    Pretty-print results from a run output file

RUN OPTIONS:
  --memory <path>      Path to MEMORY.md
  --bridge <path>      Path to bridge.md
  --responsibilities <path>  Path to responsibilities file
  --claude-md <path>   Path to CLAUDE.md
  --conversation <path> Path to conversation JSON {messages:[{role,content}]}
  --file <id>=<label>=<path>  Extra file segment (repeatable)
  --split-memory       Split MEMORY.md by H2 headers (granular ablation)
  --probe <text>       Probe prompt to run against the context
  --backend <ollama|api>  LLM backend (default: ollama)
  --model <name>       Completion model (default: qwen3:32b)
  --embed-model <name> Embedding model (default: nomic-embed-text)
  --scorer <embedding|judge|both>  Scoring strategy (default: both)
  --judge-top-n <n>    Run judge on top N segments only (default: 3)
  --ollama-url <url>   Ollama base URL (default: http://localhost:11434)
  --out <path>         Output file for JSON results (default: results.json)
  --quiet              Suppress progress logs

REPORT OPTIONS:
  <file>               Path to results JSON (default: results.json)
  --no-explanations    Hide judge explanations

EXAMPLES:
  context-inspector run \\
    --memory ~/Code/dottie-weaver/identity/MEMORY.md \\
    --bridge ~/Code/dottie-weaver/identity/bridge.md \\
    --probe "What should I do next?" \\
    --scorer both \\
    --out results.json

  context-inspector report results.json
`.trim();

// ─── Arg parsing ──────────────────────────────────────────────────────────────

function parseArgs(argv: string[]): Record<string, string | string[] | boolean> {
  const args: Record<string, string | string[] | boolean> = {};
  let i = 0;
  while (i < argv.length) {
    const arg = argv[i];
    if (arg.startsWith("--")) {
      const key = arg.slice(2);
      const next = argv[i + 1];
      if (!next || next.startsWith("--")) {
        args[key] = true;
        i++;
      } else {
        if (key === "file") {
          const existing = args["file"];
          args["file"] = [
            ...(Array.isArray(existing) ? existing : []),
            next,
          ];
        } else {
          args[key] = next;
        }
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

  // Parse context files into segments
  const splitMemory = args["split-memory"] === true;
  const memoryPath = args["memory"] as string | undefined;
  const extraFiles = (args["file"] as string[] | undefined)?.map((spec) => {
    const parts = spec.split("=");
    if (parts.length < 3) {
      console.error(`Invalid --file spec: ${spec}. Use: id=label=path`);
      process.exit(1);
    }
    return { id: parts[0], label: parts[1], path: parts.slice(2).join("=") };
  });

  const segments = (() => {
    if (splitMemory && memoryPath) {
      const content = readFileSync(memoryPath, "utf-8");
      return [
        ...splitByH2(content, "memory"),
        ...parseAgentContext({
          bridgePath: args["bridge"] as string | undefined,
          responsibilitiesPath: args["responsibilities"] as string | undefined,
          claudeMdPath: args["claude-md"] as string | undefined,
          conversationPath: args["conversation"] as string | undefined,
          extraFiles,
        }),
      ];
    }
    return parseAgentContext({
      memoryPath,
      bridgePath: args["bridge"] as string | undefined,
      responsibilitiesPath: args["responsibilities"] as string | undefined,
      claudeMdPath: args["claude-md"] as string | undefined,
      conversationPath: args["conversation"] as string | undefined,
      extraFiles,
    });
  })();

  const scorer = (args["scorer"] as string | undefined) ?? "both";
  if (!["embedding", "judge", "both"].includes(scorer)) {
    console.error(`Invalid --scorer: ${scorer}. Use: embedding | judge | both`);
    process.exit(1);
  }

  const config: RunConfig = {
    segments,
    probePrompt: probe,
    backend: (args["backend"] as "ollama" | "api" | undefined) ?? "ollama",
    model: (args["model"] as string | undefined) ?? "qwen3:32b",
    embedModel: (args["embed-model"] as string | undefined) ?? "nomic-embed-text",
    scorer: scorer as "embedding" | "judge" | "both",
    judgeTopN: parseInt((args["judge-top-n"] as string | undefined) ?? "3", 10),
    ollamaBaseUrl: args["ollama-url"] as string | undefined,
  };

  const quiet = args["quiet"] === true;
  const onProgress = quiet ? undefined : (msg: string) => console.log(msg);

  console.log(`Running ablation on ${segments.length} segments...`);
  const backend = createBackend(config);
  const result = await runAblation(config, backend, onProgress);

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
