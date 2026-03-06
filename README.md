# context-inspector

Ablation-based context scoring for AI agents. Find out which parts of your prompt context actually drive behavior — and which are dead weight.

## Why

When you build an AI agent, you assemble a context window from many pieces: a system prompt, a memory file, a session log, tool results, conversation history. You don't know which of those pieces matter for any given task.

**context-inspector** answers that question empirically. It runs ablation tests — systematically removing one segment at a time, re-running the same probe prompt, and measuring how much the agent's behavior changed. Segments that cause large behavioral shifts when removed are load-bearing. Segments that cause no shift are safe to trim or ignore.

This is useful for:
- **Debugging**: understanding why an agent made a specific decision
- **Optimization**: finding context you can cut without behavioral cost
- **Auditing**: verifying that certain context files are actually influencing the agent
- **Development**: checking which sections of a large instructions file actually matter

## How it works

```
Full context → probe prompt → baseline response

For each segment:
  Remove segment → same probe → ablated response
  Score the delta (embedding cosine distance and/or LLM-as-judge)

Rank segments by behavioral impact score
```

Two scoring strategies, configurable:

- **Embedding** (fast): embeds both responses with a text embedding model, computes cosine similarity. High cosine distance = large behavioral shift = load-bearing segment. Runs in seconds.
- **Judge** (slow, explanatory): sends both responses to a completion model and asks it to rate the behavioral difference 1–10 and explain it in one sentence. Gives you the "why," not just the score.
- **Both** (default): embeddings score all segments fast; judge runs only on the top N (default: 3) to add explanations where they matter most.

## Prerequisites

### Option A: Ollama (local, default)

Install [Ollama](https://ollama.com) and pull the models you want to use:

```bash
# Embedding model (required for --scorer embedding or --scorer both)
ollama pull nomic-embed-text

# Completion model (any chat model works — this is the default)
ollama pull qwen3:32b
```

You can use any Ollama-compatible model. Pass `--model` and `--embed-model` to override the defaults.

### Option B: OpenAI

Set your API key:

```bash
export OPENAI_API_KEY=sk-...
```

Then use `--backend openai`. Default models: `gpt-4o-mini` for completions, `text-embedding-3-small` for embeddings.

This also works with any OpenAI-compatible provider (Together AI, Groq, Mistral, LM Studio, etc.) — set `--api-base-url` to point at the provider's endpoint.

### Option C: Anthropic

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Then use `--backend anthropic`. Note: Anthropic does not expose an embedding endpoint, so use `--scorer judge` when using this backend.

### Runtime

Requires [Bun](https://bun.sh):

```bash
curl -fsSL https://bun.sh/install | bash
```

## Quick start

```bash
git clone https://github.com/scoutos-labs/context-inspector.git
cd context-inspector
bun install
```

Run against your own context files:

```bash
bun src/cli.ts run \
  --file sys=System=./my-system.md \
  --file mem=Memory=./my-memory.md \
  --probe "What should I do next?" \
  --out results.json

bun src/cli.ts report results.json
```

Example output:

```
══════════════════════════════════════════════════════════════════════
  Context Inspector — Ablation Report
══════════════════════════════════════════════════════════════════════
  Probe:   What should I do next?
  Model:   qwen3:32b
  Scorer:  both

  SEGMENT RANKING (most impactful → least)
──────────────────────────────────────────────────────────────────────
   1. session-state.md             ████████████░░░░░░░░ CRITICAL  61.2%
      embedding delta:  58.1%
      judge score:      70.0%
      explanation:      Removing session state caused the agent to propose
                        starting from scratch rather than continuing in-progress work.

   2. memory.md                    ████░░░░░░░░░░░░░░░░ PARTIAL   21.4%
      embedding delta:  20.9%
      judge score:      22.0%

   3. system-prompt.md             ██░░░░░░░░░░░░░░░░░░ LOW        9.8%
      embedding delta:  9.8%
```

## Usage

```
context-inspector run [options]
context-inspector report [file]
```

### Input flags

| Flag | Description |
|------|-------------|
| `--file <id>=<label>=<path>` | Load a file as a named context segment (repeatable) |
| `--split <id>` | Split this segment by `##` headers for section-level ablation (repeatable) |
| `--conversation <path>` | JSON file: `{ messages: [{role, content}] }` |

`--file` is the primary input mechanism. Pass any number of files, each with a short identifier, a display label, and a path.

`--split` takes a file id and splits that file into one segment per `##` section instead of treating it as a single block. Useful when a large file has many sections and you want to know which sections matter.

### Scoring flags

| Flag | Description |
|------|-------------|
| `--scorer embedding\|judge\|both` | Scoring strategy (default: `both`) |
| `--judge-top-n <n>` | Run judge only on top N segments (default: 3) |
| `--probe <text>` | Probe prompt (required) |

### Backend flags

| Flag | Description |
|------|-------------|
| `--backend ollama\|anthropic\|openai` | LLM backend (default: `ollama`) |
| `--model <name>` | Completion model override |
| `--embed-model <name>` | Embedding model override (Ollama and OpenAI) |
| `--ollama-url <url>` | Ollama base URL (default: `http://localhost:11434`) |
| `--api-base-url <url>` | Base URL for OpenAI-compatible providers |

Default models by backend:

| Backend | Completion | Embedding |
|---------|-----------|-----------|
| `ollama` | `qwen3:32b` | `nomic-embed-text` |
| `openai` | `gpt-4o-mini` | `text-embedding-3-small` |
| `anthropic` | `claude-haiku-4-5-20251001` | _(not supported)_ |

### Output flags

| Flag | Description |
|------|-------------|
| `--out <path>` | Save results JSON (default: `results.json`) |
| `--quiet` | Suppress progress logs |

## Examples

```bash
# Granular ablation of a large instructions file (section by section)
bun src/cli.ts run \
  --file mem=Memory=./memory.md \
  --split mem \
  --probe "What is the current project status?" \
  --scorer embedding

# Anthropic backend, judge-only scoring
bun src/cli.ts run \
  --file sys=System=./system.md \
  --file ctx=Context=./context.md \
  --probe "What should I do next?" \
  --backend anthropic \
  --scorer judge

# OpenAI-compatible local server (LM Studio, vLLM, etc.)
bun src/cli.ts run \
  --file sys=System=./system.md \
  --file ctx=Context=./context.md \
  --probe "What should I do next?" \
  --backend openai \
  --api-base-url http://localhost:1234/v1

# Multiple files with a custom Ollama instance and model
bun src/cli.ts run \
  --file a=Instructions=./instructions.md \
  --file b=Session=./session.md \
  --file c=Tools=./tools.md \
  --probe "How do I handle this situation?" \
  --ollama-url http://my-ollama-host:11434 \
  --model llama3.1:70b \
  --out my-results.json
```

## Running tests

```bash
bun test
```

All tests run against a mock backend — no Ollama or API key required.

## License

MIT
