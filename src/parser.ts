import { readFileSync } from "fs";
import type { Segment } from "./types.ts";

export interface ParseOptions {
  memoryPath?: string;
  bridgePath?: string;
  responsibilitiesPath?: string;
  claudeMdPath?: string;
  conversationPath?: string;    // JSON file with {messages: [{role, content}]}
  extraFiles?: Array<{ id: string; label: string; path: string }>;
}

function readFile(path: string): string {
  try {
    return readFileSync(path, "utf-8").trim();
  } catch (e) {
    throw new Error(`Could not read file ${path}: ${(e as Error).message}`);
  }
}

export function parseAgentContext(opts: ParseOptions): Segment[] {
  const segments: Segment[] = [];

  if (opts.memoryPath) {
    segments.push({
      id: "memory",
      label: "MEMORY.md",
      content: readFile(opts.memoryPath),
    });
  }

  if (opts.bridgePath) {
    segments.push({
      id: "bridge",
      label: "bridge.md",
      content: readFile(opts.bridgePath),
    });
  }

  if (opts.responsibilitiesPath) {
    segments.push({
      id: "responsibilities",
      label: "responsibilities",
      content: readFile(opts.responsibilitiesPath),
    });
  }

  if (opts.claudeMdPath) {
    segments.push({
      id: "claude_md",
      label: "CLAUDE.md",
      content: readFile(opts.claudeMdPath),
    });
  }

  if (opts.conversationPath) {
    const raw = readFile(opts.conversationPath);
    const parsed = JSON.parse(raw) as { messages: Array<{ role: string; content: string }> };
    const conversation = parsed.messages
      .map((m) => `[${m.role.toUpperCase()}]: ${m.content}`)
      .join("\n\n");
    segments.push({
      id: "conversation",
      label: "conversation history",
      content: conversation,
    });
  }

  for (const extra of opts.extraFiles ?? []) {
    segments.push({
      id: extra.id,
      label: extra.label,
      content: readFile(extra.path),
    });
  }

  if (segments.length === 0) {
    throw new Error("No context files provided. Use --memory, --bridge, etc.");
  }

  return segments;
}

// Split a single large file by markdown H2 headers (## Section Name)
export function splitByH2(content: string, fileId: string): Segment[] {
  const lines = content.split("\n");
  const segments: Segment[] = [];
  let currentLabel = "preamble";
  let currentLines: string[] = [];

  for (const line of lines) {
    const h2Match = line.match(/^##\s+(.+)/);
    if (h2Match) {
      if (currentLines.length > 0) {
        segments.push({
          id: `${fileId}__${currentLabel.toLowerCase().replace(/\s+/g, "_")}`,
          label: `${fileId}: ${currentLabel}`,
          content: currentLines.join("\n").trim(),
        });
      }
      currentLabel = h2Match[1];
      currentLines = [];
    } else {
      currentLines.push(line);
    }
  }

  if (currentLines.length > 0) {
    segments.push({
      id: `${fileId}__${currentLabel.toLowerCase().replace(/\s+/g, "_")}`,
      label: `${fileId}: ${currentLabel}`,
      content: currentLines.join("\n").trim(),
    });
  }

  return segments.filter((s) => s.content.length > 10);
}
