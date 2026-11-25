import type {
  LanguageModelV3Prompt,
  LanguageModelV3ToolResultOutput,
} from "@ai-sdk/provider"
import type { AcpContentBlock } from "./acp-client.js"

export const ACP_PROVIDER_ID = "antigravity-acp"

export const UNSAFE_INTERNAL_SIGNATURE = /\bsession_move\b|tools\.opencode\./i

export const API_ERROR_ABORT_PATTERN =
  /API error \(attempt \d+\)|RESOURCE_EXHAUSTED|Individual quota reached|\(code 429\)|\(code 403\)|PERMISSION_DENIED/i

const ALLOWED_INSTRUCTION_KEYS = new Set([
  "core/instructions",
  "core/skill-guidance",
  "core/mcp-guidance",
  "core/codemode",
])

function truncateText(text: string, maxLen = 6000): string {
  if (text.length <= maxLen) return text
  return `${text.slice(0, maxLen)}\n... [truncated ${text.length - maxLen} chars]`
}

function stringifyCompact(value: unknown, maxLen = 2000): string {
  if (typeof value === "string") return truncateText(value, maxLen)
  try {
    return truncateText(JSON.stringify(value), maxLen)
  } catch {
    return truncateText(String(value), maxLen)
  }
}

function summarizeToolResultValue(result: any): string {
  if (!result || typeof result !== "object") return stringifyCompact(result)
  if (result.type === "content" && Array.isArray(result.value)) {
    return result.value
      .map((item: any) =>
        item?.type === "text"
          ? String(item.text ?? "")
          : `[file ${item?.mime ?? ""} ${item?.name ?? ""}]`,
      )
      .join("\n")
  }
  if ("value" in result) return stringifyCompact(result.value, 6000)
  return stringifyCompact(result, 6000)
}

/**
 * Rewrites Antigravity providerExecuted (hosted) tool calls/results into plain text blocks
 * when a non-Antigravity model runs next, preventing Anthropic/OpenAI protocol serializers
 * from rejecting unfamiliar server-executed tool names.
 */
export function sanitizeForNonAcpModel(messages: any[]): void {
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]
    if (!msg || !Array.isArray(msg.content)) continue

    let changed = false
    const nextContent: any[] = []
    for (const part of msg.content) {
      if (!part || typeof part !== "object") {
        nextContent.push(part)
        continue
      }

      if (part.type === "compaction") {
        changed = true
        nextContent.push({
          type: "text",
          text: part.text
            ? `[Prior conversation summary]\n${String(part.text)}`
            : "[Prior conversation checkpoint]",
        })
        continue
      }

      if (part.type === "effort") {
        changed = true
        continue
      }

      if (msg.role === "assistant") {
        const toolName = String(part.name ?? part.toolName ?? "")
        const isAcpHosted =
          (part.type === "tool-call" || part.type === "tool-result") &&
          part.providerExecuted === true &&
          (Boolean(part.providerMetadata?.[ACP_PROVIDER_ID]) ||
            part.providerOptions?.[ACP_PROVIDER_ID]?.acp === true ||
            toolName.startsWith("acp_"))

        if (isAcpHosted) {
          changed = true
          if (part.type === "tool-call") {
            nextContent.push({
              type: "text",
              text: `[Antigravity tool call \`${toolName}\`: ${stringifyCompact(part.input, 1200)}]`,
            })
          } else {
            nextContent.push({
              type: "text",
              text: `[Antigravity tool \`${toolName}\` result: ${summarizeToolResultValue(part.result)}]`,
            })
          }
          continue
        }
      }

      nextContent.push(part)
    }

    if (changed) {
      const merged = mergeAdjacentTextParts(nextContent)
      messages[i] =
        merged.length > 0
          ? { ...msg, content: merged }
          : {
              ...msg,
              content: [{
                type: "text",
                text: msg.role === "assistant" ? "[Antigravity turn completed]" : "[Checkpoint]",
              }],
            }
    }
  }
}

/**
 * Rewrites native compaction parts, effort parts, and provider media refs before
 * OpenCode's AI SDK adapter serializes history into LanguageModelV3Prompt.
 */
export function sanitizeForAcpModel(messages: any[]): void {
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]
    if (!msg || !Array.isArray(msg.content)) continue

    let changed = false
    const nextContent: any[] = []
    for (const part of msg.content) {
      if (!part || typeof part !== "object") {
        nextContent.push(part)
        continue
      }
      if (part.type === "compaction") {
        changed = true
        nextContent.push({
          type: "text",
          text: part.text
            ? `[Prior conversation summary]\n${String(part.text)}`
            : "[Prior conversation checkpoint]",
        })
        continue
      }
      if (part.type === "effort") {
        changed = true
        continue
      }
      if (part.type === "media" && part.media?.source?.type === "ref") {
        changed = true
        nextContent.push({
          type: "text",
          text: `[Provider media ref ${String(part.media.source.provider ?? "")}:${String(part.media.source.id ?? "")}]`,
        })
        continue
      }
      nextContent.push(part)
    }

    if (changed) {
      const merged = mergeAdjacentTextParts(nextContent)
      messages[i] =
        msg.role === "assistant" && merged.length === 0
          ? { ...msg, content: [{ type: "text", text: "[Prior assistant step]" }] }
          : { ...msg, content: merged }
    }
  }
}

function mergeAdjacentTextParts(parts: any[]): any[] {
  const out: any[] = []
  for (const part of parts) {
    const prev = out[out.length - 1]
    if (
      part?.type === "text" &&
      prev?.type === "text" &&
      !part.providerMetadata &&
      !prev.providerMetadata
    ) {
      out[out.length - 1] = {
        ...prev,
        text: `${prev.text}\n${part.text}`,
      }
    } else {
      out.push(part)
    }
  }
  return out
}

export function extractToolResultsFromPrompt(
  prompt: LanguageModelV3Prompt,
  expectedCallIds: ReadonlySet<string>,
): Map<string, { text: string; isError: boolean }> {
  const results = new Map<string, { text: string; isError: boolean }>()
  for (let i = prompt.length - 1; i >= 0; i--) {
    const msg = prompt[i]
    if (msg.role !== "tool" && msg.role !== "assistant") continue
    for (const part of msg.content) {
      if (part.type !== "tool-result") continue
      if (!expectedCallIds.has(part.toolCallId) || results.has(part.toolCallId)) continue
      results.set(part.toolCallId, formatToolResultOutput(part.output))
    }
    if (results.size >= expectedCallIds.size) break
  }
  return results
}

function formatToolResultOutput(output: LanguageModelV3ToolResultOutput): {
  text: string
  isError: boolean
} {
  switch (output.type) {
    case "text":
      return { text: output.value, isError: false }
    case "json":
      return { text: JSON.stringify(output.value), isError: false }
    case "error-text":
      return { text: output.value, isError: true }
    case "error-json":
      return { text: JSON.stringify(output.value), isError: true }
    case "execution-denied":
      return {
        text: output.reason ? `Tool execution denied: ${output.reason}` : "Tool execution denied",
        isError: true,
      }
    case "content": {
      const text = output.value
        .map((c) => {
          if (c.type === "text") return c.text
          if (c.type === "file-data" || c.type === "image-data") {
            const fname = "filename" in c ? c.filename : undefined
            return `[Attachment ${c.mediaType}${fname ? `: ${fname}` : ""}]`
          }
          if (c.type === "file-url" || c.type === "image-url") {
            return `[Attachment URL: ${c.url}]`
          }
          return "[Attachment]"
        })
        .join("\n")
      return { text, isError: false }
    }
    default:
      return { text: stringifyCompact(output), isError: false }
  }
}

function hasAcpMarker(msg: LanguageModelV3Prompt[number]): boolean {
  if (msg.role !== "assistant") return false
  const msgOpts = (msg as any).providerOptions
  if (msgOpts && ACP_PROVIDER_ID in msgOpts) return true
  const msgMeta = (msg as any).providerMetadata
  if (msgMeta && ACP_PROVIDER_ID in msgMeta) return true
  return (msg.content as any[]).some((part) => {
    if (!part || typeof part !== "object") return false
    const partOpts = (part as any).providerOptions
    if (partOpts && ACP_PROVIDER_ID in partOpts) return true
    const partMeta = (part as any).providerMetadata
    return !!(partMeta && ACP_PROVIDER_ID in partMeta)
  })
}

function formatMessageForCatchUp(msg: LanguageModelV3Prompt[number]): string {
  if (msg.role === "system") return ""
  const chunks: string[] = []
  for (const part of msg.content) {
    if (part.type === "text") {
      if (part.text.trim()) chunks.push(part.text)
    } else if (part.type === "file") {
      chunks.push(`[file ${part.mediaType}${part.filename ? ` ${part.filename}` : ""}]`)
    } else if (part.type === "tool-call") {
      chunks.push(`[tool-call ${part.toolName}(${stringifyCompact(part.input, 800)})]`)
    } else if (part.type === "tool-result") {
      const formatted = formatToolResultOutput(part.output)
      chunks.push(
        `[tool-result ${part.toolName}${formatted.isError ? " ERROR" : ""}: ${truncateText(formatted.text, 2000)}]`,
      )
    }
  }
  if (chunks.length === 0) return ""
  return `[${msg.role.toUpperCase()}]: ${chunks.join("\n")}`
}

function fileDataToBase64(data: Uint8Array | string | URL): string | undefined {
  if (data instanceof Uint8Array) {
    return Buffer.from(data).toString("base64")
  }
  if (typeof data === "string") {
    // data: URIs carry inline base64 after the marker. The wrapper regex alone
    // is not trusted: the captured payload must itself be valid non-empty
    // base64 (no whitespace, padded length, base64 alphabet only).
    if (data.startsWith("data:")) {
      const match = /^data:[^;]+;base64,(.*)$/s.exec(data)
      if (!match) return undefined
      const payload = match[1]
      if (!payload || payload.length === 0) return undefined
      if (/\s/.test(payload)) return undefined
      if (payload.length % 4 !== 0) return undefined
      if (!/^[A-Za-z0-9+/]*={0,2}$/.test(payload)) return undefined
      return payload
    }
    // Raw inline base64 is also accepted (no URL/path/whitespace, valid
    // alphabet, padded length). Anything else — URLs, filesystem paths,
    // plain strings, refs — falls back to an "[Attached file]" placeholder
    // and is never sent as image bytes.
    if (isInlineBase64(data)) return data
    return undefined
  }
  return undefined
}

/**
 * Conservative inline-base64 test: standard alphabet only, padded length, a
 * minimum size so short ids/tokens never masquerade as images, and no URL,
 * scheme, or whitespace markers anywhere in the string.
 *
 * Real base64-encoded JPEGs start with `/9j/` — a leading `/` alone must NOT
 * reject them. Only known filesystem path prefixes (`/tmp/`, `/var/`,
 * `/home/`, `/Users/`, `/etc/`, `/usr/`, `/opt/`, `./`, `../`, `~/`) plus
 * backslashes still reject; everything else starting with `/` that matches
 * the base64 alphabet is accepted as inline image bytes.
 */
const KNOWN_PATH_PREFIXES = [
  "/tmp/",
  "/var/",
  "/home/",
  "/Users/",
  "/etc/",
  "/usr/",
  "/opt/",
  "./",
  "../",
  "~/",
] as const

function isInlineBase64(data: string): boolean {
  if (data.length < 32 || data.length % 4 !== 0) return false
  if (/\s/.test(data)) return false
  if (data.includes("://")) return false
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(data)) return false
  if (data.includes("\\")) return false
  for (const prefix of KNOWN_PATH_PREFIXES) {
    if (data.startsWith(prefix)) return false
  }
  // Unix-path-shaped strings like `/workspace/assets/screenshots/abcdef` use
  // only path-safe chars and slashes but are not image bytes. A leading `/`
  // with 2+ path-like segments (`[A-Za-z0-9_.-]+`) and no base64-only markers
  // (`=` padding or `+`) is rejected — except real JPEG base64, which starts
  // with `/9j/` and must keep working.
  if (data.startsWith("/") && !data.startsWith("/9j/")) {
    const rest = data.slice(1)
    if (rest.includes("/")) {
      const segments = data.split("/").filter(Boolean)
      if (
        segments.length >= 2 &&
        segments.every((seg) => /^[A-Za-z0-9_.\-]+$/.test(seg)) &&
        !/[+=]/.test(data)
      ) {
        return false
      }
    }
  }
  return /^[A-Za-z0-9+/]*={0,2}$/.test(data)
}

/**
 * Legacy cross-model catch-up builder (pre-Thin mode).
 * @deprecated Strict Thin ACP Mode must use {@link buildThinAcpPromptBlocks} —
 * non-ACP turns are never replayed upstream (see `hasNonAcpAssistantHistory`).
 * Kept only for backward compatibility; not used by the bridge.
 */
export function buildAcpPromptBlocks(
  prompt: LanguageModelV3Prompt,
  options?: { readonly forceFullHistory?: boolean },
): AcpContentBlock[] {
  const nonSystem = prompt.filter((m) => m.role !== "system")
  if (nonSystem.length === 0) {
    return [{ type: "text", text: "Continue." }]
  }

  let lastAcpIdx = -1
  if (!options?.forceFullHistory) {
    for (let i = nonSystem.length - 1; i >= 0; i--) {
      if (hasAcpMarker(nonSystem[i])) {
        lastAcpIdx = i
        break
      }
    }
  }

  const unseen = nonSystem.slice(lastAcpIdx + 1)
  if (unseen.length === 0) {
    const tail = nonSystem[nonSystem.length - 1]
    const fallback = formatMessageForCatchUp(tail)
    return [{ type: "text", text: fallback || "Continue." }]
  }

  // Last user message is the active turn trigger; everything in `unseen` before it came from other models
  let lastUserIdx = unseen.length - 1
  while (lastUserIdx >= 0 && unseen[lastUserIdx].role !== "user") {
    lastUserIdx--
  }

  const priorMsgs = lastUserIdx > 0 ? unseen.slice(0, lastUserIdx) : lastUserIdx < 0 ? unseen : []
  const currentUserMsg = lastUserIdx >= 0 ? unseen[lastUserIdx] : undefined

  const blocks: AcpContentBlock[] = []
  const catchUpLines = priorMsgs.map(formatMessageForCatchUp).filter(Boolean)
  if (catchUpLines.length > 0) {
    blocks.push({
      type: "text",
      text: `<conversation_context_from_other_models>\n${catchUpLines.join("\n\n")}\n</conversation_context_from_other_models>\n`,
    })
  }

  if (currentUserMsg && currentUserMsg.role === "user") {
    for (const part of currentUserMsg.content) {
      if (part.type === "text") {
        if (part.text.length > 0) {
          blocks.push({ type: "text", text: part.text })
        }
      } else if (part.type === "file") {
        const b64 = fileDataToBase64(part.data)
        if (part.mediaType.startsWith("image/") && b64) {
          blocks.push({ type: "image", data: b64, mimeType: part.mediaType })
        } else {
          blocks.push({
            type: "text",
            text: `[Attached file: ${part.filename ?? "file"} (${part.mediaType})]`,
          })
        }
      }
    }
  }

  if (blocks.length === 0) {
    blocks.push({ type: "text", text: "Continue." })
  }
  return blocks
}

/**
 * Thin-client prompt builder (ToS-compliant).
 * Sends ONLY the raw latest human prompt to `agy acp` with zero injected
 * OpenCode system prompt, zero `<workspace_tools_bridge>` preamble, and zero
 * OpenCode tool/skill serialization. Antigravity owns its own session history
 * via `session/new` / `session/resume` — never replay non-ACP turns.
 */
export function buildThinAcpPromptBlocks(prompt: LanguageModelV3Prompt): AcpContentBlock[] {
  const nonSystem = prompt.filter((m) => m.role !== "system")
  if (nonSystem.length === 0) {
    return [{ type: "text", text: "Continue." }]
  }
  // Latest human user message is the only turn trigger.
  let lastUserIdx = nonSystem.length - 1
  while (lastUserIdx >= 0 && nonSystem[lastUserIdx].role !== "user") {
    lastUserIdx--
  }
  const currentUserMsg = lastUserIdx >= 0 ? nonSystem[lastUserIdx] : undefined
  const blocks: AcpContentBlock[] = []
  if (currentUserMsg && currentUserMsg.role === "user") {
    for (const part of currentUserMsg.content) {
      if (part.type === "text") {
        if (part.text.length > 0) {
          blocks.push({ type: "text", text: part.text })
        }
      } else if (part.type === "file") {
        const b64 = fileDataToBase64(part.data)
        if (part.mediaType.startsWith("image/") && b64) {
          blocks.push({ type: "image", data: b64, mimeType: part.mediaType })
        } else {
          blocks.push({
            type: "text",
            text: `[Attached file: ${part.filename ?? "file"} (${part.mediaType})]`,
          })
        }
      }
    }
  }
  if (blocks.length === 0) {
    blocks.push({ type: "text", text: "Continue." })
  }
  return blocks
}

/**
 * Detects OpenCode session history produced by non-ACP models.
 * In Thin ACP Mode Antigravity owns its own session history, so a thread that
 * already contains non-ACP assistant turns must not be replayed into `agy acp`
 * — the user should start a fresh session or run `/agy-reset`.
 */
export function hasNonAcpAssistantHistory(prompt: LanguageModelV3Prompt): boolean {
  for (const msg of prompt) {
    if (msg.role !== "assistant" || !Array.isArray(msg.content)) continue
    if (msg.content.length === 0) continue
    if (!hasAcpMarker(msg)) return true
  }
  return false
}

/**
 * Local plugin status/cooldown replies (circuit-open, restart-skip, missing
 * CLI) are emitted by `createImmediateTextStream` with a
 * `{ [ACP_PROVIDER_ID]: { localStatus: true } }` marker. They are neither ACP
 * turns nor foreign model turns and must be skipped when scanning history.
 */
export function isLocalStatusMessage(msg: LanguageModelV3Prompt[number]): boolean {
  const check = (holder: any): boolean => {
    if (!holder || typeof holder !== "object") return false
    const meta = holder.providerMetadata?.[ACP_PROVIDER_ID] ?? holder.providerOptions?.[ACP_PROVIDER_ID]
    return !!meta && typeof meta === "object" && (meta as any).localStatus === true
  }
  if (check(msg)) return true
  if (check((msg as any).providerOptions)) return true
  if (check((msg as any).providerMetadata)) return true
  const content = (msg as any).content
  if (Array.isArray(content)) {
    for (const part of content) {
      if (!part || typeof part !== "object") continue
      if (check(part)) return true
    }
  }
  return false
}

/**
 * T3-Code-style `create_with_handoff` detector.
 * Returns true only when there are UNSEEN non-ACP assistant turns since the
 * last ACP turn (i.e. the user switched to a normal model and back).
 * Normal ACP-to-ACP flow (only ACP assistant turns + latest user) returns false.
 *
 * Local plugin status replies (`localStatus: true`) are skipped entirely:
 * they are neither a completed ACP handoff checkpoint nor a foreign model turn.
 */
export function hasUnseenNonAcpAssistantHistory(prompt: LanguageModelV3Prompt): boolean {
  const nonSystem = prompt.filter((m) => m.role !== "system")
  if (nonSystem.length === 0) return false
  let lastAcpIdx = -1
  for (let i = nonSystem.length - 1; i >= 0; i--) {
    const msg = nonSystem[i] as any
    if (isLocalStatusMessage(msg)) continue
    if (hasAcpMarker(nonSystem[i])) {
      lastAcpIdx = i
      break
    }
  }
  const unseen = nonSystem.slice(lastAcpIdx + 1)
  for (const msg of unseen) {
    if (isLocalStatusMessage(msg as any)) continue
    if (msg.role !== "assistant" || !Array.isArray(msg.content)) continue
    if (msg.content.length === 0) continue
    if (!hasAcpMarker(msg)) return true
  }
  return false
}

export const TEXT_ONLY_HANDOFF_HEADER =
  "Historical conversation context (plain text only; not a new request; native tool/reasoning state is not replayed):"

/** Appended when a single over-budget history line must be cut to fit. */
const HANDOFF_TRUNCATION_MARKER = "\n... [truncated]"

function extractHistoryText(msg: LanguageModelV3Prompt[number]): string {
  const chunks: string[] = []
  for (const part of msg.content) {
    if (!part || typeof part !== "object") continue
    if (part.type === "text" && typeof (part as any).text === "string" && (part as any).text.trim()) {
      chunks.push((part as any).text.trim())
    }
    // Deliberately dropped: file, image, tool-call, tool-result, reasoning,
    // and any other non-text parts (no tool/reasoning replay, no attachments).
  }
  return chunks.join("\n").trim()
}

/**
 * T3-Code-style `create_with_handoff` prompt builder (v0.2.0).
 * Plain-text-only handoff: ONLY `[Historical user]` / `[Historical assistant]`
 * text turns (capped at `maxChars`, default 16,000), dropping all reasoning,
 * tool calls, tool results, schemas, todos, and approvals — followed by the
 * latest raw user prompt + attachments. Zero `<workspace_tools_bridge>`,
 * zero OpenCode system prompt, zero tool schema injection.
 */
export function buildTextOnlyHandoffPromptBlocks(
  prompt: LanguageModelV3Prompt,
  options?: { readonly maxChars?: number },
): AcpContentBlock[] {
  const maxChars = options?.maxChars ?? 16_000
  const nonSystem = prompt.filter((m) => m.role !== "system")
  if (nonSystem.length === 0) {
    return [{ type: "text", text: "Continue." }]
  }
  let lastUserIdx = nonSystem.length - 1
  while (lastUserIdx >= 0 && nonSystem[lastUserIdx].role !== "user") {
    lastUserIdx--
  }
  const currentUserMsg = lastUserIdx >= 0 ? nonSystem[lastUserIdx] : undefined
  const priorMsgs = lastUserIdx > 0 ? nonSystem.slice(0, lastUserIdx) : []

  const lines: string[] = []
  for (const msg of priorMsgs) {
    if (msg.role !== "user" && msg.role !== "assistant") continue
    // Local plugin status/cooldown notices are harness output, not conversation
    // history — never send them upstream as `[Historical assistant]` lines.
    if (isLocalStatusMessage(msg as any)) continue
    const text = extractHistoryText(msg)
    if (!text) continue
    lines.push(msg.role === "user" ? `[Historical user]: ${text}` : `[Historical assistant]: ${text}`)
  }

  // Newest-first budget: drop oldest lines first so the most recent context survives.
  // The header line is framing overhead reserved up front (the "\n\n"
  // separators and trailing newline are already counted by the +2 per line
  // below), so the entire historical block (excluding the latest user prompt)
  // never exceeds maxChars. The truncation marker length is reserved too.
  const framingOverhead = TEXT_ONLY_HANDOFF_HEADER.length
  const budget = Math.max(0, maxChars - framingOverhead)
  let total = lines.reduce((sum, l) => sum + l.length + 2, 0)
  while (lines.length > 1 && total > budget) {
    const dropped = lines.shift()!
    total -= dropped.length + 2
  }
  if (lines.length === 1 && lines[0].length + 2 > budget) {
    if (budget <= HANDOFF_TRUNCATION_MARKER.length + 2) {
      lines.length = 0
    } else {
      lines[0] = `${lines[0].slice(0, budget - HANDOFF_TRUNCATION_MARKER.length - 2)}${HANDOFF_TRUNCATION_MARKER}`
    }
  }

  const blocks: AcpContentBlock[] = []
  if (lines.length > 0) {
    blocks.push({
      type: "text",
      text: `${TEXT_ONLY_HANDOFF_HEADER}\n${lines.join("\n\n")}\n`,
    })
  }

  if (currentUserMsg && currentUserMsg.role === "user") {
    for (const part of currentUserMsg.content) {
      if (part.type === "text") {
        if (part.text.length > 0) {
          blocks.push({ type: "text", text: part.text })
        }
      } else if (part.type === "file") {
        const b64 = fileDataToBase64(part.data)
        if (part.mediaType.startsWith("image/") && b64) {
          blocks.push({ type: "image", data: b64, mimeType: part.mediaType })
        } else {
          blocks.push({
            type: "text",
            text: `[Attached file: ${part.filename ?? "file"} (${part.mediaType})]`,
          })
        }
      }
    }
  }

  if (blocks.length === 0) {
    blocks.push({ type: "text", text: "Continue." })
  }
  return blocks
}

/**
 * Removes any line matching internal OpenCode signatures from arbitrary text (prompts or tool metadata).
 */
export function stripUnsafeInternalLines(text: string): string {
  return text
    .split(/\r?\n/)
    .filter((line) => !UNSAFE_INTERNAL_SIGNATURE.test(line))
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
}

/**
 * Strips OpenCode's internal `opencode` Code Mode namespace (`session_move`, worktrees, etc.)
 * without modifying or obfuscating strings.
 */
export function stripInternalOpencodeNamespace(codemodeText: string): string {
  const lines = codemodeText.split(/\r?\n/)
  const kept: string[] = []
  let skippingNamespace = false

  for (const line of lines) {
    if (skippingNamespace) {
      if (line.startsWith("  ") || line.startsWith("\t")) {
        continue
      }
      skippingNamespace = false
    }

    if (/^- opencode(?:\b|[(:])/.test(line)) {
      skippingNamespace = true
      continue
    }

    if (UNSAFE_INTERNAL_SIGNATURE.test(line)) {
      continue
    }

    kept.push(line)
  }

  return kept
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
}

/**
 * Extracts only user-owned workspace context (`AGENTS.md`, skills, MCP guidance, and
 * non-internal Code Mode tool namespaces) from OpenCode's `system` parts.
 */
export function extractSafeAcpContext(systemParts: ReadonlyArray<any>): string {
  const sections: string[] = []

  for (const part of systemParts) {
    const entries = part?.metadata?.experimental?.instructions
    if (!Array.isArray(entries)) continue

    for (const entry of entries) {
      const key = typeof entry?.key === "string" ? entry.key : ""
      if (!ALLOWED_INSTRUCTION_KEYS.has(key)) continue
      const rawText = typeof entry?.text === "string" ? entry.text : ""
      if (!rawText.trim()) continue

      const sectionText =
        key === "core/codemode" ? stripInternalOpencodeNamespace(rawText) : rawText.trim()
      if (sectionText) {
        sections.push(sectionText)
      }
    }
  }

  if (sections.length === 0) return ""

  return stripUnsafeInternalLines(sections.join("\n\n"))
}

/**
 * Detects an OpenCode synthetic auto-resume prompt following a halted API error turn so the
 * bridge does not automatically replay an erroring request on server restart.
 */
export function isSyntheticRestartAfterError(prompt: LanguageModelV3Prompt): boolean {
  const nonSystem = prompt.filter((m) => m.role !== "system")
  if (nonSystem.length === 0) return false

  const last = nonSystem[nonSystem.length - 1]
  if (last.role !== "user") return false

  const lastText = last.content
    .filter(
      (p): p is { type: "text"; text: string } =>
        p.type === "text" && typeof p.text === "string",
    )
    .map((p) => p.text)
    .join("\n")

  if (!lastText.includes("The server restarted while you were working.")) return false

  for (let i = nonSystem.length - 2; i >= 0; i--) {
    const msg = nonSystem[i]
    if (msg.role !== "assistant") continue
    const assistantText = msg.content
      .filter(
        (p): p is { type: "text"; text: string } =>
          p.type === "text" && typeof p.text === "string",
      )
      .map((p) => p.text)
      .join("\n")
    return (
      assistantText.includes("Antigravity ACP halted") ||
      API_ERROR_ABORT_PATTERN.test(assistantText)
    )
  }

  return false
}
