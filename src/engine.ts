import type {
  Segment,
  AblationResult,
  RunConfig,
  RunResult,
  LLMBackend,
} from "./types.ts";

// ─── Math ────────────────────────────────────────────────────────────────────

function cosine(a: number[], b: number[]): number {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom === 0 ? 0 : dot / denom;
}

// ─── Backends ─────────────────────────────────────────────────────────────────

export class OllamaBackend implements LLMBackend {
  private baseUrl: string;
  private model: string;
  private embedModel: string;

  constructor(opts: { baseUrl?: string; model?: string; embedModel?: string }) {
    this.baseUrl = opts.baseUrl ?? "http://localhost:11434";
    this.model = opts.model ?? "qwen3:32b";
    this.embedModel = opts.embedModel ?? "nomic-embed-text";
  }

  async complete(system: string, prompt: string): Promise<string> {
    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: this.model,
        stream: false,
        options: { temperature: 0 },
        messages: [
          { role: "system", content: system },
          { role: "user", content: prompt },
        ],
      }),
    });
    if (!res.ok) throw new Error(`Ollama complete failed: ${res.status} ${await res.text()}`);
    const data = await res.json() as { message: { content: string } };
    return data.message.content.trim();
  }

  async embed(text: string): Promise<number[]> {
    const res = await fetch(`${this.baseUrl}/api/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: this.embedModel, prompt: text }),
    });
    if (!res.ok) throw new Error(`Ollama embed failed: ${res.status} ${await res.text()}`);
    const data = await res.json() as { embedding: number[] };
    return data.embedding;
  }
}

export class AnthropicBackend implements LLMBackend {
  private apiKey: string;
  private model: string;

  constructor(opts: { apiKey: string; model?: string }) {
    this.apiKey = opts.apiKey;
    this.model = opts.model ?? "claude-haiku-4-5-20251001";
  }

  async complete(system: string, prompt: string): Promise<string> {
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": this.apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: this.model,
        max_tokens: 1024,
        system,
        messages: [{ role: "user", content: prompt }],
      }),
    });
    if (!res.ok) throw new Error(`Anthropic complete failed: ${res.status} ${await res.text()}`);
    const data = await res.json() as { content: Array<{ text: string }> };
    return data.content[0].text.trim();
  }

  async embed(_text: string): Promise<number[]> {
    // Anthropic doesn't expose an embedding endpoint — caller should not use this
    throw new Error("AnthropicBackend does not support embeddings. Use scorer='judge' with API backend.");
  }
}

// ─── Scorers ─────────────────────────────────────────────────────────────────

async function scoreByEmbedding(
  originalResponse: string,
  ablatedResponse: string,
  backend: LLMBackend,
): Promise<number> {
  const [embA, embB] = await Promise.all([
    backend.embed(originalResponse),
    backend.embed(ablatedResponse),
  ]);
  const similarity = cosine(embA, embB);
  return 1 - similarity; // high delta = segment mattered
}

async function scoreByJudge(
  originalResponse: string,
  ablatedResponse: string,
  backend: LLMBackend,
): Promise<{ score: number; explanation: string }> {
  const system = `You are a behavioral difference evaluator. You will be shown two agent responses to the same prompt.
Your job is to rate how behaviorally different they are on a scale of 0–10:
0 = identical behavior / no meaningful difference
10 = completely different behavior, different actions, different conclusions

Focus on: what the agent decides to do, what it concludes, what actions it proposes.
Ignore: minor wording differences, formatting, length.

Respond ONLY in this JSON format:
{"score": <number 0-10>, "explanation": "<one sentence>"}`;

  const prompt = `Response A (original context):
---
${originalResponse}
---

Response B (one context segment removed):
---
${ablatedResponse}
---

Rate the behavioral difference.`;

  const raw = await backend.complete(system, prompt);

  // Parse JSON from response, tolerant of markdown fences
  const jsonMatch = raw.match(/\{[\s\S]*?\}/);
  if (!jsonMatch) {
    console.warn("Judge returned non-JSON:", raw.slice(0, 100));
    return { score: 0, explanation: "parse error" };
  }
  const parsed = JSON.parse(jsonMatch[0]) as { score: number; explanation: string };
  return {
    score: Math.min(10, Math.max(0, parsed.score)) / 10,
    explanation: parsed.explanation ?? "",
  };
}

// ─── Context composition ──────────────────────────────────────────────────────

function composeContext(segments: Segment[]): string {
  return segments
    .map((s) => `## ${s.label}\n\n${s.content}`)
    .join("\n\n---\n\n");
}

// ─── Main ablation runner ─────────────────────────────────────────────────────

export async function runAblation(
  config: RunConfig,
  backend: LLMBackend,
  onProgress?: (msg: string) => void,
): Promise<RunResult> {
  const log = onProgress ?? (() => {});
  const { segments, probePrompt, scorer, judgeTopN } = config;

  // Step 1: baseline response (full context)
  log("Running baseline (full context)...");
  const fullSystem = composeContext(segments);
  const originalResponse = await backend.complete(fullSystem, probePrompt);
  log(`Baseline done. (${originalResponse.length} chars)`);

  // Step 2: ablate each segment
  const results: AblationResult[] = [];

  for (const segment of segments) {
    log(`Ablating: ${segment.label}...`);
    const reduced = segments.filter((s) => s.id !== segment.id);
    const reducedSystem = composeContext(reduced);
    const ablatedResponse = await backend.complete(reducedSystem, probePrompt);

    const result: AblationResult = {
      segmentId: segment.id,
      label: segment.label,
      behavioralScore: 0,
      originalResponse,
      ablatedResponse,
    };

    // Embedding score (fast, always runs if scorer includes it)
    if (scorer === "embedding" || scorer === "both") {
      try {
        result.embeddingScore = await scoreByEmbedding(originalResponse, ablatedResponse, backend);
      } catch (e) {
        log(`  Embedding failed for ${segment.label}: ${(e as Error).message}`);
      }
    }

    result.behavioralScore = result.embeddingScore ?? 0;
    results.push(result);
    log(`  ✓ ${segment.label} → embedding score: ${(result.embeddingScore ?? 0).toFixed(3)}`);
  }

  // Step 3: LLM-as-judge on top N by embedding score (if scorer includes judge)
  if (scorer === "judge" || scorer === "both") {
    // Sort by embedding score to find top N candidates
    const sorted = [...results].sort(
      (a, b) => (b.embeddingScore ?? b.behavioralScore) - (a.embeddingScore ?? a.behavioralScore),
    );
    const topN = scorer === "judge" ? results : sorted.slice(0, judgeTopN);

    for (const r of topN) {
      log(`  Judge scoring: ${r.label}...`);
      try {
        const { score, explanation } = await scoreByJudge(
          r.originalResponse,
          r.ablatedResponse,
          backend,
        );
        r.judgeScore = score;
        r.judgeExplanation = explanation;
        // When both scorers run, blend: embed 70%, judge 30%
        if (scorer === "both" && r.embeddingScore !== undefined) {
          r.behavioralScore = r.embeddingScore * 0.7 + score * 0.3;
        } else {
          r.behavioralScore = score;
        }
        log(`  ✓ ${r.label} → judge: ${score.toFixed(3)} — ${explanation}`);
      } catch (e) {
        log(`  Judge failed for ${r.label}: ${(e as Error).message}`);
      }
    }
  }

  // Final ranking
  results.sort((a, b) => b.behavioralScore - a.behavioralScore);
  const rankedSegments = results.map((r) => r.segmentId);

  return {
    probePrompt,
    model: config.model ?? "qwen3:32b",
    scorer,
    originalResponse,
    results,
    rankedSegments,
    timestamp: new Date().toISOString(),
  };
}

// ─── Factory ──────────────────────────────────────────────────────────────────

export function createBackend(config: RunConfig): LLMBackend {
  if (config.backend === "ollama") {
    return new OllamaBackend({
      baseUrl: config.ollamaBaseUrl,
      model: config.model,
      embedModel: config.embedModel,
    });
  }
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("ANTHROPIC_API_KEY not set");
  return new AnthropicBackend({ apiKey, model: config.model });
}
