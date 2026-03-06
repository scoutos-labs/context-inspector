# context-inspector

Ablation-based context scoring for AI agents. Find out which parts of your prompt context actually drive behavior — and which are dead weight.

## Why

When you build an AI agent, you assemble a context window from many pieces: a system prompt, a memory file, a session log, tool results, conversation history. You don't know which of those pieces matter for any given task.

**context-inspector** answers that question empirically. It runs ablation tests — systematically removing one segment at a time, re-running the same probe prompt, and measuring how much the agent's behavior changed. Segments that cause large behavioral shifts when removed are load-bearing. Segments that cause no shift are safe to trim or ignore.

This is useful for:
- **Debugging**: understanding why an agent made a specific decision
- **Optimization**: finding context you can cut without behavioral cost
- **Auditing**: verifying that certain context files are actually influencing the agent
- **Development**: checking which sections of a large memory or instructions file actually matter

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
ollama pull qwen2.5:32b
```

You can use any Ollama-compatible model. Pass `--model` and `--embed-model` to override the defaults.

### Option B: Anthropic API

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Then use `--backend api`. Note: the API backend does not support embeddings, so use `--scorer judge` with it.

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
  --file sys=system-prompt=./my-system.md \
  --file mem=memory=./my-memory.md \
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
  Model:   qwen2.5:32b
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
| `--memory <path>` | Convenience alias: load a file labeled "MEMORY.md" |
| `--bridge <path>` | Convenience alias: load a file labeled "bridge.md" |
| `--responsibilities <path>` | Convenience alias: load a file labeled "responsibilities" |
| `--claude-md <path>` | Convenience alias: load a file labeled "CLAUDE.md" |
| `--conversation <path>` | JSON file: `{ messages: [{role, content}] }` |
| `--file <id>=<label>=<path>` | Load any file as a named segment (repeatable) |
| `--split-memory` | Split the `--memory` file by `##` headers for section-level ablation |

The `--file` flag is the generic way to load any context file. The named flags (`--memory`, `--bridge`, etc.) are convenience shortcuts for common agent context file naming conventions.

### Scoring flags

| Flag | Description |
|------|-------------|
| `--scorer embedding\|judge\|both` | Scoring strategy (default: `both`) |
| `--judge-top-n <n>` | Run judge only on top N segments (default: 3) |
| `--probe <text>` | Probe prompt (required) |

### Backend flags

| Flag | Description |
|------|-------------|
| `--backend ollama\|api` | LLM backend (default: `ollama`) |
| `--model <name>` | Completion model (default: `qwen2.5:32b` for Ollama, `claude-haiku-4-5-20251001` for API) |
| `--embed-model <name>` | Embedding model, Ollama only (default: `nomic-embed-text`) |
| `--ollama-url <url>` | Ollama base URL (default: `http://localhost:11434`) |

### Output flags

| Flag | Description |
|------|-------------|
| `--out <path>` | Save results JSON (default: `results.json`) |
| `--quiet` | Suppress progress logs |

## Examples

```bash
# Granular ablation of a large memory file (section by section)
bun src/cli.ts run \
  --split-memory \
  --memory ~/my-agent/memory.md \
  --probe "What is the current project status?" \
  --scorer embedding

# API backend, judge-only
bun src/cli.ts run \
  --file sys=system-prompt=./system.md \
  --file ctx=context=./context.md \
  --probe "What should I do next?" \
  --backend api \
  --scorer judge

# Multiple files with a custom Ollama instance
bun src/cli.ts run \
  --file a=instructions=./instructions.md \
  --file b=session=./session.md \
  --file c=tools=./tools.md \
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
