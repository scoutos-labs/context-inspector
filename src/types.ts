// Shared interfaces for Context Inspector

export interface Segment {
  id: string;    // e.g. "memory", "bridge", "system-prompt"
  label: string; // display name
  content: string;
}

export interface AblationResult {
  segmentId: string;
  label: string;
  behavioralScore: number; // 0–1; 0 = dead weight, 1 = critical
  embeddingScore?: number; // cosine delta (if embedding scorer ran)
  judgeScore?: number;     // 0–10 normalized to 0–1 (if judge scorer ran)
  judgeExplanation?: string;
  originalResponse: string;
  ablatedResponse: string;
}

export interface RunConfig {
  segments: Segment[];
  probePrompt: string;
  backend: "ollama" | "anthropic" | "openai";
  model?: string;          // completion model
  embedModel?: string;     // embedding model (Ollama or OpenAI)
  scorer: "embedding" | "judge" | "both";
  judgeTopN: number;       // only run judge on top N segments by embedding score
  ollamaBaseUrl?: string;  // defaults to http://localhost:11434
  apiBaseUrl?: string;     // base URL for OpenAI-compatible providers
}

export interface RunResult {
  probePrompt: string;
  model: string;
  scorer: string;
  originalResponse: string;
  results: AblationResult[];
  rankedSegments: string[]; // segmentIds, most impactful first
  timestamp: string;
}

// Backend abstraction — engine is backend-agnostic
export interface LLMBackend {
  complete(system: string, prompt: string): Promise<string>;
  embed(text: string): Promise<number[]>;
}
