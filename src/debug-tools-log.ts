import { appendFileSync, closeSync, constants as fsConstants, fstatSync, openSync } from "node:fs"
import { homedir, tmpdir } from "node:os"
import { join } from "node:path"

const seenNativeToolPairs = new Set<string>()
let warnedDebugLog = false

const EMAIL_RE = /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g
const BEARER_RE = /Bearer\s+\S+/gi
const SK_TOKEN_RE = /\bsk-[A-Za-z0-9_-]{16,}/g
const GH_TOKEN_RE = /\bgh[pousr]_[A-Za-z0-9]{20,}/g
const AIZA_TOKEN_RE = /\bAIza[0-9A-Za-z_-]{20,}/g
const YA29_TOKEN_RE = /\bya29\.[0-9A-Za-z_-]+/g
const BARE_TOKEN_RE = /[A-Za-z0-9_-]{32,}/g

function redactString(value: string): string {
  const home = homedir()
  const result = home && home !== "/" ? value.replaceAll(home, "~") : value
  return result
    .replace(EMAIL_RE, "<email>")
    .replace(BEARER_RE, "<redacted>")
    .replace(SK_TOKEN_RE, "<redacted>")
    .replace(GH_TOKEN_RE, "<redacted>")
    .replace(AIZA_TOKEN_RE, "<redacted>")
    .replace(YA29_TOKEN_RE, "<redacted>")
    .replace(BARE_TOKEN_RE, "<redacted>")
}

/** Recursively redact sensitive strings (home paths, emails, tokens) in a debug value. */
export function redactDebugValue(value: unknown): unknown {
  if (typeof value === "string") return redactString(value)
  if (Array.isArray(value)) return value.map((item) => redactDebugValue(item))
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {}
    for (const [key, item] of Object.entries(value)) {
      out[key] = redactDebugValue(item)
    }
    return out
  }
  return value
}

/** Append one redacted tool-shape diagnostic line when ANTIGRAVITY_ACP_DEBUG_TOOLS=1. */
export function logNativeTool(name: string, kind: string | undefined, rawInput: unknown, mappedTo: string): void {
  if (process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS !== "1") return
  const key = JSON.stringify([name, kind])
  if (seenNativeToolPairs.has(key)) return
  seenNativeToolPairs.add(key)
  try {
    const redacted = redactDebugValue({ name, kind: kind ?? null, rawInput, mappedTo })
    const entry = JSON.stringify(redacted,
      (_key, value) => typeof value === "string" ? value.slice(0, 300) : value)
    const file = process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS_FILE || join(tmpdir(), "antigravity-acp-toolcalls.jsonl")
    const fd = openSync(file,
      fsConstants.O_WRONLY | fsConstants.O_APPEND | fsConstants.O_CREAT |
        fsConstants.O_NOFOLLOW | fsConstants.O_NONBLOCK, 0o600)
    try {
      const stat = fstatSync(fd)
      if (stat.isFile() && stat.uid === process.getuid?.() && stat.nlink === 1 && (stat.mode & 0o077) === 0) {
        if (!warnedDebugLog) {
          warnedDebugLog = true
          console.warn(`antigravity-acp: tool debug log enabled -> ${file}; it may contain command and file snippets`)
        }
        appendFileSync(fd, `${entry}\n`)
      }
    } finally {
      closeSync(fd)
    }
  } catch {
    // Diagnostics must not interrupt the tool stream.
  }
}
