import { readFileSync } from "fs";
import type { Segment } from "./types.ts";

export interface FileSpec {
  id: string;
  label: string;
  path: string;
  split?: boolean; // if true, split by H2 headers instead of treating as a single segment
}

export interface ParseOptions {
  files?: FileSpec[];
  conversationPath?: string; // JSON file with {messages: [{role, content}]}
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

  for (const file of opts.files ?? []) {
    const content = readFile(file.path);
    if (file.split) {
      segments.push(...splitByH2(content, file.id, file.label));
    } else {
      segments.push({ id: file.id, label: file.label, content });
    }
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

  if (segments.length === 0) {
    throw new Error("No context files provided. Use --file id=label=path.");
  }

  return segments;
}

// Split a single file by markdown H2 headers (## Section Name).
// Each section becomes its own segment for granular ablation.
export function splitByH2(content: string, fileId: string, fileLabel?: string): Segment[] {
  const lines = content.split("\n");
  const segments: Segment[] = [];
  let currentLabel = "preamble";
  let currentLines: string[] = [];

  for (const line of lines) {
    const h2Match = line.match(/^##\s+(.+)/);
    if (h2Match) {
      if (currentLines.length > 0) {
        const prefix = fileLabel ?? fileId;
        segments.push({
          id: `${fileId}__${currentLabel.toLowerCase().replace(/\s+/g, "_")}`,
          label: `${prefix}: ${currentLabel}`,
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
    const prefix = fileLabel ?? fileId;
    segments.push({
      id: `${fileId}__${currentLabel.toLowerCase().replace(/\s+/g, "_")}`,
      label: `${prefix}: ${currentLabel}`,
      content: currentLines.join("\n").trim(),
    });
  }

  return segments.filter((s) => s.content.length > 10);
}
