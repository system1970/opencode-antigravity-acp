import { afterEach, beforeEach, describe, expect, test } from "bun:test"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import { Effect } from "effect"
import {
  AgyNotFoundError,
  FALLBACK_MODELS,
  buildAgySpawnEnv,
  buildModelCatalog,
  discoverAcpModels,
  resolveAcpAuthMethod,
  resolveAgyBinary,
  type AcpDiscoveredModel,
  type AcpSessionUpdate,
} from "../src/acp-client.js"
import {
  API_ERROR_ABORT_PATTERN,
  TEXT_ONLY_HANDOFF_HEADER,
  UNSAFE_INTERNAL_SIGNATURE,
  buildTextOnlyHandoffPromptBlocks,
  buildThinAcpPromptBlocks,
  extractSafeAcpContext,
  hasNonAcpAssistantHistory,
  hasUnseenNonAcpAssistantHistory,
  isSyntheticRestartAfterError,
  sanitizeForAcpModel,
  sanitizeForNonAcpModel,
  stripUnsafeInternalLines,
} from "../src/history-sanitizer.js"
import { McpRelayServer } from "../src/mcp-relay.js"
import { elicitationToQuestions, parseQuestionToolAnswers } from "../src/question-bridge.js"
import {
  SessionBridgeManager,
  quotaBucketForModel,
  resetSharedCatalogForTests,
} from "../src/session-bridge.js"

const bridgeEnv = [
  "ANTIGRAVITY_ACP_NATIVE_TOOLS",
  "ANTIGRAVITY_ACP_DEBUG_TOOLS",
  "ANTIGRAVITY_ACP_DEBUG_TOOLS_FILE",
  "ANTIGRAVITY_ACP_PREAMBLE",
] as const
let previousBridgeEnv: (string | undefined)[]
beforeEach(() => {
  previousBridgeEnv = bridgeEnv.map((key) => process.env[key])
  for (const key of bridgeEnv) delete process.env[key]
  process.env.ANTIGRAVITY_ACP_PREAMBLE = "off"
})
afterEach(() => {
  for (const [index, key] of bridgeEnv.entries()) {
    const value = previousBridgeEnv[index]
    if (value === undefined) delete process.env[key]
    else process.env[key] = value
  }
})

function makeMockCtx(initialStorage?: Record<string, unknown>) {
  const store = new Map<string, unknown>(Object.entries(initialStorage ?? {}))
  return {
    location: { directory: "/tmp/acp-test-dir" },
    storage: {
      get: (k: string) => Effect.succeed(store.get(k)),
      set: (k: string, v: unknown) => Effect.sync(() => void store.set(k, v)),
      remove: (k: string) => Effect.sync(() => void store.delete(k)),
    },
    session: {
      get: () => Effect.succeed({ location: { directory: "/tmp/acp-test-dir" } }),
      update: () => Effect.void,
    },
    permission: {
      assert: () => Effect.void,
    },
  } as any
}

async function readAllParts(stream: ReadableStream<any>) {
  const reader = stream.getReader()
  const parts: any[] = []
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    parts.push(value)
  }
  return parts
}

async function waitFor(cond: () => boolean, ms = 5000): Promise<void> {
  const t0 = Date.now()
  while (!cond()) {
    if (Date.now() - t0 > ms) throw new Error("timed out waiting for background warm")
    await new Promise((r) => setTimeout(r, 10))
  }
}

describe("resolveAgyBinary & missing binary handling", () => {
  test("respects AGY_BIN env override", () => {
    expect(resolveAgyBinary({ AGY_BIN: " /custom/path/agy " })).toBe("/custom/path/agy")
  })

  test("finds executable agy on PATH when present", () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "agy-bin-test-"))
    try {
      const fakeBin = path.join(tmpDir, process.platform === "win32" ? "agy.exe" : "agy")
      fs.writeFileSync(fakeBin, "#!/bin/sh\nexit 0\n")
      fs.chmodSync(fakeBin, 0o755)
      expect(resolveAgyBinary({ PATH: tmpDir })).toBe(fakeBin)
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true })
    }
  })

  test("discoverAcpModels throws AgyNotFoundError on missing binary", async () => {
    await expect(
      discoverAcpModels("/tmp", { binaryPath: "/nonexistent/path/agy-missing-bin" }),
    ).rejects.toBeInstanceOf(AgyNotFoundError)
  })
})

describe("McpRelayServer", () => {
  test("supports start -> stop -> start, concurrent stop, auth, and tool scrubbing", async () => {
    const relay = new McpRelayServer({
      getTools: () => [
        {
          name: "mcp_oc_execute",
          description: "Run JS code.\ntools.opencode.session_move({})\nSafe tool line.",
          inputSchema: { type: "object", properties: { code: { type: "string" } } },
        },
      ],
      callTool: async () => ({ content: [{ type: "text", text: "ok" }] }),
    })

    expect(() => relay.urlForSession("s1")).toThrow(/McpRelayServer has not started/)

    await relay.start()
    const url1 = relay.urlForSession("s1")
    expect(url1).toMatch(/^http:\/\/127\.0\.0\.1:\d+\/mcp\/s1$/)

    const unauth = await fetch(url1, { method: "POST", body: "{}" })
    expect(unauth.status).toBe(401)

    const listRes = await fetch(url1, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${relay.token}`,
      },
      body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/list", params: {} }),
    })
    expect(listRes.status).toBe(200)
    const listJson = (await listRes.json()) as any
    const desc = listJson.result.tools[0].description as string
    expect(UNSAFE_INTERNAL_SIGNATURE.test(desc)).toBe(false)
    expect(desc).toContain("Safe tool line.")

    await relay.stop()
    expect(() => relay.urlForSession("s1")).toThrow(/McpRelayServer has not started/)

    await relay.start()
    const url2 = relay.urlForSession("s1")
    expect(url2).toMatch(/^http:\/\/127\.0\.0\.1:\d+\/mcp\/s1$/)
    await relay.stop()

    const startP = relay.start()
    const stopP = relay.stop()
    await Promise.allSettled([startP, stopP])
    expect(() => relay.urlForSession("s1")).toThrow(/McpRelayServer has not started/)
  })
})

describe("SessionBridgeManager native ACP tool results", () => {
  async function streamToolUpdates(updates: readonly AcpSessionUpdate[]) {
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: ({ onUpdate }) => ({
        isClosed: false,
        initialize: async () => ({}),
        authenticate: async () => ({}),
        newSession: async () => ({ sessionId: "acp-native-tools" }),
        setModel: async () => {},
        setMode: async () => {},
        notifyCancel: () => {},
        dispose: async () => {},
        prompt: async (sessionId: string) => {
          for (const update of updates) onUpdate?.(sessionId, update)
          return { stopReason: "end_turn" }
        },
      } as any),
    })
    try {
      const lm = bridge.createLanguageModel("gemini-4-neon")
      const { stream } = await lm.doStream({
        prompt: [{ role: "user", content: [{ type: "text", text: "Check the tool result" }] }],
        headers: { "x-opencode-session-id": "ses-native-tools" },
      })
      return await readAllParts(stream)
    } finally {
      await bridge.disposeAll()
    }
  }

  test("keeps the announced tool name on terminal updates with an omitted or changed name", async () => {
    const parts = await streamToolUpdates([
      { sessionUpdate: "tool_call", toolCallId: "read", name: "read_file", status: "pending", rawInput: { Path: "/repo/a" } },
      { sessionUpdate: "tool_call", toolCallId: "run", name: "run_command", status: "in_progress", title: "Print directory", rawInput: { CommandLine: "pwd", WaitMsBeforeAsync: 1000 } },
      { sessionUpdate: "tool_call_update", toolCallId: "read", status: "completed", rawOutput: "file text" },
      { sessionUpdate: "tool_call_update", toolCallId: "run", name: "read_file", status: "failed", rawOutput: "command failed" },
    ])

    const calls = parts.filter((p) => p.type === "tool-call")
    expect(calls.map((p) => [p.toolCallId, p.toolName, JSON.parse(p.input), p.providerExecuted])).toEqual([
      ["read", "read", { filePath: "/repo/a" }, true],
      ["run", "bash", { description: "Print directory", command: "pwd" }, true],
    ])
    expect(calls.every((p) => p.providerMetadata?.["antigravity-acp"]?.acp === true)).toBe(true)
    expect(parts.filter((p) => p.type === "tool-input-start").every(
      (p) => p.providerMetadata?.["antigravity-acp"]?.acp === true,
    )).toBe(true)
    const results = parts.filter((p) => p.type === "tool-result")
    expect(results.map((p) => [p.toolCallId, p.toolName, p.result, p.isError])).toEqual([
      ["read", "read", "file text", false],
      ["run", "bash", "command failed", true],
    ])
  })

  test("does not emit mapped native parts for relayed MCP updates", async () => {
    const parts = await streamToolUpdates([
      { sessionUpdate: "tool_call", toolCallId: "relay", name: "mcp_oc_execute", kind: "execute" },
      { sessionUpdate: "tool_call_update", toolCallId: "relay", name: "run_command", status: "completed" },
    ])
    expect(parts.filter((p) => ["tool-input-start", "tool-call", "tool-result"].includes(p.type))).toEqual([])
  })

  test("restores legacy display names and raw inputs with native tools off", async () => {
    process.env.ANTIGRAVITY_ACP_NATIVE_TOOLS = "off"
    const rawInput = { CommandLine: "pwd", WaitMsBeforeAsync: 1000 }
    const parts = await streamToolUpdates([
      { sessionUpdate: "tool_call", toolCallId: "run", name: "run_command", kind: "execute", title: "Directory", rawInput },
      { sessionUpdate: "tool_call_update", toolCallId: "run", status: "completed" },
    ])
    expect(parts.find((p) => p.type === "tool-call")).toMatchObject({
      toolName: "acp_run_command", input: JSON.stringify(rawInput), providerExecuted: true,
    })
    expect(parts.find((p) => p.type === "tool-result")?.toolName).toBe("acp_run_command")
  })

  test.skipIf(process.platform === "win32")("logs the first tool shape only after opt-in, with a private file and truncated strings", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "acp-tool-log-"))
    const file = path.join(dir, "calls.jsonl")
    process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS_FILE = file
    const rawInput = { sample: "x ".repeat(175), short: "value" }
    const updates = [
      { sessionUpdate: "tool_call", toolCallId: "a", name: "diagnostic_tool", kind: "other", rawInput },
      { sessionUpdate: "tool_call", toolCallId: "b", name: "diagnostic_tool", kind: "other", rawInput: { sample: "not first" } },
      { sessionUpdate: "tool_call", toolCallId: "c", name: "diagnostic_tool", kind: "execute", rawInput: { CommandLine: "pwd" } },
    ] as const
    try {
      await streamToolUpdates(updates)
      expect(fs.existsSync(file)).toBe(false)
      process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS = "true"
      await streamToolUpdates(updates)
      expect(fs.existsSync(file)).toBe(false)
      process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS = "1"
      await streamToolUpdates(updates)
      const lines = fs.readFileSync(file, "utf8").trimEnd().split("\n")
      expect(lines.map((line) => JSON.parse(line))).toEqual([
        { name: "diagnostic_tool", kind: "other", rawInput: { ...rawInput, sample: "x ".repeat(150) }, mappedTo: "acp_diagnostic_tool" },
        { name: "diagnostic_tool", kind: "execute", rawInput: { CommandLine: "pwd" }, mappedTo: "bash" },
      ])
      expect(fs.statSync(file).mode & 0o777).toBe(0o600)
    } finally {
      fs.rmSync(dir, { recursive: true, force: true })
    }
  })

  test.skipIf(process.platform === "win32")("refuses a symlink or a shared file without interrupting the tool stream", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "acp-tool-log-"))
    const file = path.join(dir, "calls.jsonl")
    const link = path.join(dir, "link.jsonl")
    process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS = "1"
    try {
      fs.writeFileSync(file, "unchanged", { mode: 0o600 })
      fs.symlinkSync(file, link)
      process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS_FILE = link
      await streamToolUpdates([{ sessionUpdate: "tool_call", toolCallId: "a", name: "linked_log_shape", status: "completed" }])
      expect(fs.readFileSync(file, "utf8")).toBe("unchanged")
      fs.chmodSync(file, 0o644)
      process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS_FILE = file
      const parts = await streamToolUpdates([{ sessionUpdate: "tool_call", toolCallId: "a", name: "shared_log_shape", status: "completed" }])
      expect(fs.readFileSync(file, "utf8")).toBe("unchanged")
      expect(parts.find((p) => p.type === "tool-result")?.toolName).toBe("acp_shared_log_shape")
    } finally {
      fs.rmSync(dir, { recursive: true, force: true })
    }
  })

  test("marks an unfinished tool as unknown with its original name when the turn ends", async () => {
    const parts = await streamToolUpdates([
      { sessionUpdate: "tool_call", toolCallId: "run", name: "run_command", status: "pending", rawInput: { CommandLine: "pwd" } },
      { sessionUpdate: "tool_call_update", toolCallId: "run", status: "in_progress", kind: "execute" },
    ])

    const results = parts.filter((p) => p.type === "tool-result")
    expect(results).toHaveLength(1)
    expect(results[0]).toMatchObject({
      toolCallId: "run",
      toolName: "bash",
      result: { status: "unknown" },
      isError: true,
    })
    expect(parts.at(-1)?.finishReason?.unified).toBe("stop")
  })

  test("uses the result when the initial tool call already reports a terminal status", async () => {
    const parts = await streamToolUpdates([
      { sessionUpdate: "tool_call", toolCallId: "read", kind: "read", status: "completed", rawInput: { Path: "/repo/a" }, rawOutput: "file text" },
    ])

    const results = parts.filter((p) => p.type === "tool-result")
    expect(results).toHaveLength(1)
    expect(results[0]).toMatchObject({
      toolCallId: "read",
      toolName: "read",
      result: "file text",
      isError: false,
    })
  })
})

describe("SessionBridgeManager lifecycle & cached discovery", () => {
  afterEach(() => {
    resetSharedCatalogForTests()
  })

  test("post-dispose cached LanguageModelV3 returns non-retryable plugin_disposed stop stream", async () => {
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir")
    const cachedLm = bridge.createLanguageModel("gemini-4-neon")
    expect(bridge.getState("sess-1").cooldownUntil).toBeNull()

    await bridge.disposeAll()

    const res = await cachedLm.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
      headers: { "x-opencode-session-id": "sess-1" },
    } as any)
    const parts = await readAllParts(res.stream)
    expect(parts.some((p) => p.type === "error")).toBe(false)
    const finish = parts.find((p) => p.type === "finish")
    expect(finish?.finishReason?.unified).toBe("stop")
    expect(finish?.finishReason?.raw).toBe("plugin_disposed")
  })

  test("mid-turn disposeAll() cleanly terminates stream with plugin_unloaded", async () => {
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir")
    const sessionObj = (bridge as any).getOrCreateSession("sess-active")
    let resolvedMcp!: (res: any) => void
    const mcpPromise = new Promise((resolve) => {
      resolvedMcp = resolve
    })
    const turnObj: any = {
      bufferedParts: [],
      partSeq: 0,
      activeNativeTools: new Map(),
      relayedUpdateIds: new Set(),
      pendingMcpCalls: new Map([
        [
          "call-1",
          {
            toolCallId: "call-1",
            mcpName: "mcp_oc_execute",
            openCodeToolName: "mcp_oc_execute",
            args: {},
            emittedToOpenCode: false,
            resolved: false,
            resolve: resolvedMcp,
          },
        ],
      ]),
      dispatchedCallIds: new Set(),
      promptSettled: false,
    }
    const activeStream = (bridge as any).createBoundStream(sessionObj, turnObj, undefined, () => {})
    sessionObj.activeTurn = turnObj
    ;(bridge as any).handleAcpUpdate(sessionObj, {
      sessionUpdate: "tool_call",
      toolCallId: "native-1",
      name: "run_command",
      status: "in_progress",
      rawInput: { CommandLine: "pwd" },
    })

    const streamReaderPromise = readAllParts(activeStream)
    await bridge.disposeAll()
    const midTurnParts = await streamReaderPromise
    const mcpResult = (await mcpPromise) as any
    expect(mcpResult.isError).toBe(true)
    expect(midTurnParts.some((p) => p.type === "error")).toBe(false)
    expect(midTurnParts.find((p) => p.type === "tool-result")).toMatchObject({
      toolCallId: "native-1",
      toolName: "bash",
      result: { status: "unknown" },
      isError: true,
    })
    const midFinish = midTurnParts.find((p) => p.type === "finish")
    expect(midFinish?.finishReason?.unified).toBe("stop")
    expect(midFinish?.finishReason?.raw).toBe("plugin_unloaded")
  })

  test("surfaces AgyNotFoundError as non-retryable agy_unavailable stream", async () => {
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir")
    ;(bridge as any).ensureAcpSession = async () => {
      throw new AgyNotFoundError("/missing/agy")
    }
    const lm = bridge.createLanguageModel("gemini-4-neon")
    const res = await lm.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
      headers: { "x-opencode-session-id": "sess-missing" },
    } as any)
    const parts = await readAllParts(res.stream)
    expect(parts.some((p) => p.type === "error")).toBe(false)
    const text = parts
      .filter((p) => p.type === "text-delta")
      .map((p) => p.delta)
      .join("")
    expect(text).toContain("Antigravity CLI")
    const finish = parts.find((p) => p.type === "finish")
    expect(finish?.finishReason?.raw).toBe("agy_unavailable")
    await bridge.disposeAll()
  })

  test("exposes cooldownUntil when circuit breaker is tripped", async () => {
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir")
    ;(bridge as any).tripCircuitBreaker("RESOURCE_EXHAUSTED (code 429)")
    const st = bridge.getState("s3")
    expect(typeof st.cooldownUntil).toBe("number")
    expect(st.cooldownUntil!).toBeGreaterThan(Date.now())
    await bridge.disposeAll()
  })

  test("deduplicates live discovery and loads fresh catalog from storage cache", async () => {
    const liveModels: AcpDiscoveredModel[] = [
      {
        id: "gemini-4-carbon",
        name: "Gemini 4 Carbon",
        family: "gemini-4-carbon",
        defaultAcpModelId: "gemini-4-carbon-high",
        variants: [{ id: "high", acpModelId: "gemini-4-carbon-high", thoughtLevel: "high" }],
      },
    ]

    let discoverCalls = 0
    let reloadNotifications = 0
    const ctx = makeMockCtx()
    const discoverModels = async () => {
      discoverCalls++
      await new Promise((r) => setTimeout(r, 15))
      return liveModels
    }

    const b1 = new SessionBridgeManager(ctx, "/tmp/w1", undefined, {
      onCatalogChanged: () => reloadNotifications++,
      discoverModels,
    })
    const b2 = new SessionBridgeManager(ctx, "/tmp/w2", undefined, {
      onCatalogChanged: () => reloadNotifications++,
      discoverModels,
    })

    const c1 = await b1.loadCachedModels()
    const c2 = await b2.loadCachedModels()
    expect(c1.stale).toBe(true)
    expect(c2.stale).toBe(true)

    const [changed1, changed2] = await Promise.all([
      b1.refreshModelsInBackground(),
      b2.refreshModelsInBackground(),
    ])
    expect(discoverCalls).toBe(1)
    expect(changed1).toBe(true)
    expect(changed2).toBe(true)
    expect(reloadNotifications).toBe(2)
    expect(b1.getModels().map((m) => m.id)).toEqual(["gemini-4-carbon"])

    // Simulate fresh server restart where memory cache is cleared but storage persists
    resetSharedCatalogForTests()
    let warmDiscoverCalls = 0
    const b3 = new SessionBridgeManager(ctx, "/tmp/w1", undefined, {
      discoverModels: async () => {
        warmDiscoverCalls++
        return liveModels
      },
    })
    const c3 = await b3.loadCachedModels()
    expect(c3.stale).toBe(false)
    expect(c3.changed).toBe(true)
    expect(b3.getModels().map((m) => m.id)).toEqual(["gemini-4-carbon"])
    expect(warmDiscoverCalls).toBe(0)

    await Promise.all([b1.disposeAll(), b2.disposeAll(), b3.disposeAll()])
  })
})

describe("Model catalog & family visibility", () => {
  test("FALLBACK_MODELS has 9 distinct families and all 23 (model, variant) targets", () => {
    expect(FALLBACK_MODELS).toHaveLength(9)
    const families = new Set<string>()
    let totalTargets = 0
    for (const m of FALLBACK_MODELS) {
      expect(m.name).not.toContain("(ACP)")
      expect(m.family).toBe(m.id)
      expect(families.has(m.family!)).toBe(false)
      families.add(m.family!)
      totalTargets += m.variants.length > 0 ? m.variants.length : 1
    }
    expect(totalTargets).toBe(23)
    expect(FALLBACK_MODELS.find((m) => m.id === "gemini-pro-orchestrator-neon")?.name).toBe(
      "Orchestrator (Neon)",
    )
    expect(FALLBACK_MODELS.find((m) => m.id === "gemini-pro-orchestrator-g1-pro")?.name).toBe(
      "Orchestrator (Argon)",
    )
  })

  test("buildModelCatalog strips (ACP), discovers extra suffixes and standalone models", () => {
    const cat1 = buildModelCatalog({
      models: {
        availableModels: [{ modelId: "gpt-oss-120b-medium", name: "GPT-OSS 120B (Medium) (ACP)" }],
      },
      configOptions: [
        { id: "model", options: [{ value: "gpt-oss-120b-medium", name: "GPT-OSS 120B (Medium) (ACP)" }] },
      ],
    })
    expect(cat1[0].name).toBe("GPT-OSS 120B (Medium)")
    expect(cat1[0].family).toBe("gpt-oss-120b-medium")

    const cat2 = buildModelCatalog({
      models: {
        availableModels: [
          { modelId: "gemini-4-neon-low", name: "Gemini 4 Neon (Low)" },
          { modelId: "gemini-4-neon-minimal", name: "Gemini 4 Neon (Minimal)" },
          { modelId: "gemini-4-carbon", name: "Gemini 4 Carbon" },
        ],
      },
      configOptions: [
        { id: "model", options: [{ value: "gemini-4-neon", name: "Gemini 4 Neon" }] },
        { id: "thought_level", options: [{ value: "low" }, { value: "high" }] },
      ],
    })
    expect(cat2).toHaveLength(2)
    expect(cat2[0].variants.map((v) => v.id)).toEqual(["low", "minimal"])
    expect(cat2[0].family).toBe("gemini-4-neon")
    expect(cat2[1].id).toBe("gemini-4-carbon")
    expect(cat2[1].family).toBe("gemini-4-carbon")
  })
})

describe("History sanitizer & cross-model context handoff", () => {
  test("extractSafeAcpContext strips opencode internal signatures and keeps safe context", () => {
    const safeCtx = extractSafeAcpContext([
      {
        metadata: {
          experimental: {
            instructions: [
              { key: "core/instructions", text: "Repo AGENTS.md rule" },
              { key: "internal/opencode-system", text: "SHOULD BE DROPPED tools.opencode.session_move" },
              {
                key: "core/codemode",
                text: [
                  "- opencode: Internal controls",
                  "  opencode.session_move({ id })",
                  "  await tools.opencode.session_move()",
                  "- external_ns: Safe workspace tool",
                  "  external_ns.open({ url })",
                ].join("\n"),
              },
            ],
          },
        },
      },
    ])
    expect(safeCtx).toContain("Repo AGENTS.md rule")
    expect(safeCtx).toContain("external_ns")
    expect(UNSAFE_INTERNAL_SIGNATURE.test(safeCtx)).toBe(false)
    expect(safeCtx).not.toContain("SHOULD BE DROPPED")

    expect(stripUnsafeInternalLines("keep\ntools.opencode.session_move()\nsession_move\nok")).toBe(
      "keep\nok",
    )
    expect(API_ERROR_ABORT_PATTERN.test("⚠ API error (attempt 1): RESOURCE_EXHAUSTED (code 429)")).toBe(
      true,
    )
    expect(API_ERROR_ABORT_PATTERN.test("Normal response chunk")).toBe(false)
  })

  test("thin prompt blocks send ONLY the raw latest human prompt (zero replay)", () => {
    const thinBlocks = buildThinAcpPromptBlocks(
      [
        { role: "user", content: [{ type: "text", text: "Secret codeword is ALPHA_9" }] },
        { role: "assistant", content: [{ type: "text", text: "Noted codeword ALPHA_9." }] },
        { role: "user", content: [{ type: "text", text: "Now switch to Antigravity and repeat it" }] },
      ] as any,
    )
    expect(thinBlocks).toHaveLength(1)
    expect((thinBlocks[0] as any).text).toBe("Now switch to Antigravity and repeat it")
    expect(JSON.stringify(thinBlocks)).not.toContain("ALPHA_9")
    expect(JSON.stringify(thinBlocks)).not.toContain("conversation_context_from_other_models")
    expect(JSON.stringify(thinBlocks)).not.toContain("workspace_tools_bridge")
    expect(JSON.stringify(thinBlocks)).not.toContain("workspace_context")

    const thinAfterAcp = buildThinAcpPromptBlocks(
      [
        { role: "user", content: [{ type: "text", text: "First ACP prompt" }] },
        {
          role: "assistant",
          content: [{ type: "text", text: "ACP already knows this" }],
          providerOptions: { "antigravity-acp": { acp: true } },
        },
        { role: "user", content: [{ type: "text", text: "Back to ACP now" }] },
      ] as any,
    )
    expect(thinAfterAcp).toHaveLength(1)
    expect((thinAfterAcp[0] as any).text).toBe("Back to ACP now")

    const thinImages = buildThinAcpPromptBlocks(
      [
        {
          role: "user",
          content: [
            { type: "text", text: "Look at this" },
            { type: "file", data: `data:image/png;base64,${Buffer.from("imgbytes").toString("base64")}`, mediaType: "image/png", filename: "shot.png" },
          ],
        },
      ] as any,
    )
    expect(thinImages).toHaveLength(2)
    expect((thinImages[0] as any).text).toBe("Look at this")
    expect((thinImages[1] as any).type).toBe("image")

    // Non-data: strings are NOT inline base64 — they fall back to a placeholder.
    const thinPlainStringFile = buildThinAcpPromptBlocks(
      [
        {
          role: "user",
          content: [
            { type: "text", text: "Look at this" },
            { type: "file", data: Buffer.from("imgbytes").toString("base64"), mediaType: "image/png", filename: "shot.png" },
          ],
        },
      ] as any,
    )
    expect(thinPlainStringFile).toHaveLength(2)
    expect((thinPlainStringFile[1] as any).type).toBe("text")
    expect((thinPlainStringFile[1] as any).text).toContain("[Attached file: shot.png (image/png)]")
  })

  test("hasNonAcpAssistantHistory locks threads with foreign model turns", () => {
    expect(hasNonAcpAssistantHistory([{ role: "user", content: [{ type: "text", text: "hi" }] }] as any)).toBe(false)
    expect(
      hasNonAcpAssistantHistory([
        { role: "user", content: [{ type: "text", text: "hi" }] },
        { role: "assistant", content: [{ type: "text", text: "foreign model reply" }] },
      ] as any),
    ).toBe(true)
    expect(
      hasNonAcpAssistantHistory([
        { role: "user", content: [{ type: "text", text: "hi" }] },
        {
          role: "assistant",
          content: [{ type: "text", text: "ACP reply" }],
          providerOptions: { "antigravity-acp": { acp: true } },
        },
        { role: "user", content: [{ type: "text", text: "follow-up" }] },
      ] as any),
    ).toBe(false)
  })

  test("resolveAcpAuthMethod honors override, GEMINI_API_KEY, and oauth-personal default", () => {
    expect(resolveAcpAuthMethod({ ANTIGRAVITY_ACP_AUTH_METHOD: "agent-platform" } as any)).toBe("agent-platform")
    expect(resolveAcpAuthMethod({ GEMINI_API_KEY: "key-123" } as any)).toBe("gemini-api-key")
    expect(
      resolveAcpAuthMethod({
        ANTIGRAVITY_ACP_AUTH_METHOD: "gemini-api-key",
        GEMINI_API_KEY: "key-123",
      } as any),
    ).toBe("gemini-api-key")
    expect(resolveAcpAuthMethod({} as any)).toBe("oauth-personal")
  })

  test("thin-mode bridge defaults to Code Mode and hands off non-ACP history without a thread lock", async () => {
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir")
    const initial = bridge.getState("thin-sess")
    expect(initial.codeModeEnabled).toBe(true)
    expect(initial.relayedToolCount).toBe(0)

    bridge.updateContextSnapshot("thin-sess", {
      systemText: "",
      agent: "build",
      tools: { read: { description: "r", input: { type: "object" } } },
    })
    expect(bridge.getState("thin-sess").relayedToolCount).toBe(0)

    bridge.updateContextSnapshot("thin-sess", {
      systemText: "",
      agent: "build",
      tools: { mytool: { description: "r", input: { type: "object" } } },
    })
    expect(bridge.getState("thin-sess").relayedToolCount).toBe(1)

    const disabled = await bridge.setCodeMode("thin-sess", false)
    expect(disabled.codeModeEnabled).toBe(false)
    expect(disabled.relayedToolCount).toBe(0)
    const enabled = await bridge.setCodeMode("thin-sess", true)
    expect(enabled.codeModeEnabled).toBe(true)
    expect(enabled.relayedToolCount).toBe(1)

    // v0.2.0: non-ACP history no longer returns thin_history_locked. It resets
    // to a fresh agy session and sends a plain-text-only handoff instead.
    // Mock the ACP transport so no real `agy` binary is spawned.
    let resetCalls = 0
    const origSoftReset = (bridge as any).softResetForHandoff.bind(bridge)
    ;(bridge as any).softResetForHandoff = async (sid: string) => {
      resetCalls++
      return origSoftReset(sid)
    }
    let capturedPrompt: any[] | undefined
    ;(bridge as any).ensureAcpSession = async (s: any) => {
      const fakeProc = {
        setModel: async () => {},
        setConfigOption: async () => {},
        prompt: async (_sid: string, prompt: any[]) => {
          capturedPrompt = prompt
          return { stopReason: "end_turn" }
        },
      }
      return { proc: fakeProc, acpSessionId: "acp-fresh", freshSession: true }
    }

    const lm = bridge.createLanguageModel("gemini-4-neon")
    const res = await lm.doStream({
      prompt: [
        { role: "user", content: [{ type: "text", text: "hello" }] },
        { role: "assistant", content: [{ type: "text", text: "foreign reply" }] },
        { role: "user", content: [{ type: "text", text: "continue in ACP" }] },
      ],
      headers: { "x-opencode-session-id": "thin-sess" },
    } as any)
    const parts = await readAllParts(res.stream)
    expect(parts.some((p) => p.type === "error")).toBe(false)
    const text = parts.filter((p) => p.type === "text-delta").map((p) => p.delta).join("")
    // Handoff prompt itself isn't echoed; the turn just completes cleanly.
    expect(parts.find((p) => p.type === "finish")?.finishReason?.unified).toBe("stop")
    expect(resetCalls).toBeGreaterThan(0)
    expect(capturedPrompt).toBeDefined()
    const handoffText = (capturedPrompt as any[]).find((b) => b.type === "text")?.text ?? ""
    expect(handoffText).toContain("Historical conversation context")
    expect(handoffText).toContain("foreign reply")
    expect(JSON.stringify(capturedPrompt)).not.toContain("thin_history_locked")
    expect(bridge.hasSession("thin-sess")).toBe(true)
    await bridge.disposeAll()
  })

  test("buildTextOnlyHandoffPromptBlocks caps at 16k and drops tool/reasoning parts", () => {
    const blocks = buildTextOnlyHandoffPromptBlocks(
      [
        { role: "user", content: [{ type: "text", text: "first question" }] },
        {
          role: "assistant",
          content: [
            { type: "text", text: "foreign answer with tool work" },
            { type: "tool-call", toolName: "bash", input: "{}" },
            { type: "tool-result", toolCallId: "c1", toolName: "bash", output: { type: "text", value: "secret-output" } },
          ],
        } as any,
        {
          role: "assistant",
          content: [{ type: "reasoning", text: "should never leak" }] as any,
        } as any,
        { role: "user", content: [{ type: "text", text: "second question" }] },
        { role: "assistant", content: [{ type: "text", text: "second foreign reply" }] },
        { role: "user", content: [{ type: "text", text: "continue in ACP now" }] },
      ] as any,
      { maxChars: 16_000 },
    )
    const joined = JSON.stringify(blocks)
    expect(blocks[0].type).toBe("text")
    expect((blocks[0] as any).text).toContain(TEXT_ONLY_HANDOFF_HEADER)
    expect((blocks[0] as any).text).toContain("[Historical user]")
    expect((blocks[0] as any).text).toContain("[Historical assistant]")
    expect((blocks[0] as any).text).toContain("foreign answer with tool work")
    // Tool/reasoning internals are dropped, never replayed.
    expect(joined).not.toContain("secret-output")
    expect(joined).not.toContain("should never leak")
    expect(joined).not.toContain("tool-call")
    expect(joined).not.toContain("workspace_tools_bridge")
    expect(joined).not.toContain("workspace_context")
    // Latest raw prompt is appended last.
    expect((blocks[blocks.length - 1] as any).text).toBe("continue in ACP now")

    // 16k cap: oversized history keeps the newest turns and stays bounded.
    const bigText = "x".repeat(20_000)
    const capped = buildTextOnlyHandoffPromptBlocks(
      [
        { role: "user", content: [{ type: "text", text: "old question" }] },
        { role: "assistant", content: [{ type: "text", text: bigText }] },
        { role: "user", content: [{ type: "text", text: "continue in ACP now" }] },
      ] as any,
      { maxChars: 16_000 },
    )
    expect(((capped[0] as any).text as string).length).toBeLessThanOrEqual(16_000)
    expect((capped[capped.length - 1] as any).text).toBe("continue in ACP now")
  })

  test("hasUnseenNonAcpAssistantHistory only fires after a model switch", () => {
    // Normal ACP-to-ACP flow: no handoff.
    expect(
      hasUnseenNonAcpAssistantHistory([
        { role: "user", content: [{ type: "text", text: "First ACP prompt" }] },
        {
          role: "assistant",
          content: [{ type: "text", text: "ACP reply" }],
          providerOptions: { "antigravity-acp": { acp: true } },
        },
        { role: "user", content: [{ type: "text", text: "follow-up in ACP" }] },
      ] as any),
    ).toBe(false)
    // Muse Spark -> ACP, or ACP -> Muse Spark -> ACP: handoff.
    expect(
      hasUnseenNonAcpAssistantHistory([
        { role: "user", content: [{ type: "text", text: "hello" }] },
        { role: "assistant", content: [{ type: "text", text: "foreign reply" }] },
        { role: "user", content: [{ type: "text", text: "continue in ACP" }] },
      ] as any),
    ).toBe(true)
    expect(
      hasUnseenNonAcpAssistantHistory([
        { role: "user", content: [{ type: "text", text: "First ACP prompt" }] },
        {
          role: "assistant",
          content: [{ type: "text", text: "ACP reply" }],
          providerOptions: { "antigravity-acp": { acp: true } },
        },
        { role: "user", content: [{ type: "text", text: "ask Muse Spark" }] },
        { role: "assistant", content: [{ type: "text", text: "spark reply" }] },
        { role: "user", content: [{ type: "text", text: "back to ACP" }] },
      ] as any),
    ).toBe(true)
  })

  test("buildAgySpawnEnv scrubs ambient keys only for oauth-personal", () => {
    // Explicit oauth-personal with stray ambient keys: scrubbed (the collision case).
    const oauthEnv = buildAgySpawnEnv({
      ANTIGRAVITY_ACP_AUTH_METHOD: "oauth-personal",
      GEMINI_API_KEY: "k",
      GOOGLE_API_KEY: "g",
      GOOGLE_APPLICATION_CREDENTIALS: "/creds.json",
      OTHER: "keep",
    } as any)
    expect(oauthEnv.GEMINI_API_KEY).toBeUndefined()
    expect(oauthEnv.GOOGLE_API_KEY).toBeUndefined()
    expect(oauthEnv.GOOGLE_APPLICATION_CREDENTIALS).toBeUndefined()
    expect(oauthEnv.OTHER).toBe("keep")

    const keyEnv = buildAgySpawnEnv({
      ANTIGRAVITY_ACP_AUTH_METHOD: "gemini-api-key",
      GEMINI_API_KEY: "k",
      GOOGLE_API_KEY: "g",
    } as any)
    expect(keyEnv.GEMINI_API_KEY).toBe("k")
    expect(keyEnv.GOOGLE_API_KEY).toBe("g")

    const platformEnv = buildAgySpawnEnv({
      ANTIGRAVITY_ACP_AUTH_METHOD: "agent-platform",
      GOOGLE_APPLICATION_CREDENTIALS: "/creds.json",
    } as any)
    expect(platformEnv.GOOGLE_APPLICATION_CREDENTIALS).toBe("/creds.json")

    // oauth-business is also a browser OAuth flow: ambient keys are scrubbed too.
    const businessEnv = buildAgySpawnEnv({
      ANTIGRAVITY_ACP_AUTH_METHOD: "oauth-business",
      GEMINI_API_KEY: "k",
      GOOGLE_API_KEY: "g",
      GOOGLE_APPLICATION_CREDENTIALS: "/creds.json",
    } as any)
    expect(businessEnv.GEMINI_API_KEY).toBeUndefined()
    expect(businessEnv.GOOGLE_API_KEY).toBeUndefined()
    expect(businessEnv.GOOGLE_APPLICATION_CREDENTIALS).toBeUndefined()
  })

  test("freshSession fallback sends a text handoff for prior ACP turns, thin for brand-new threads", async () => {
    // Even when all prior turns were ACP-marked (no model switch), a fresh
    // session/new after a failed resume owns no history — hand it off as text
    // so the earlier conversation isn't silently lost.
    const acpPrompt = [
      { role: "user", content: [{ type: "text", text: "First ACP question" }] },
      {
        role: "assistant",
        content: [{ type: "text", text: "ACP answer one" }],
        providerOptions: { "antigravity-acp": { acp: true } },
      },
      { role: "user", content: [{ type: "text", text: "Follow-up in ACP" }] },
    ] as any

    async function runWithFresh(freshSession: boolean) {
      const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir")
      let capturedPrompt: any[] | undefined
      ;(bridge as any).ensureAcpSession = async (s: any) => {
        const fakeProc = {
          setModel: async () => {},
          setConfigOption: async () => {},
          prompt: async (_sid: string, prompt: any[]) => {
            capturedPrompt = prompt
            return { stopReason: "end_turn" }
          },
        }
        return { proc: fakeProc, acpSessionId: "acp-fresh", freshSession }
      }
      const lm = bridge.createLanguageModel("gemini-4-neon")
      const res = await lm.doStream({
        prompt: acpPrompt,
        headers: { "x-opencode-session-id": "fresh-sess" },
      } as any)
      const parts = await readAllParts(res.stream)
      expect(parts.some((p) => p.type === "error")).toBe(false)
      expect(parts.find((p) => p.type === "finish")?.finishReason?.unified).toBe("stop")
      await bridge.disposeAll()
      return capturedPrompt as any[] | undefined
    }

    const handoffPrompt = await runWithFresh(true)
    expect(handoffPrompt).toBeDefined()
    const handoffHead = handoffPrompt!.find((b) => b.type === "text")?.text ?? ""
    expect(handoffHead).toContain(TEXT_ONLY_HANDOFF_HEADER)
    expect(handoffHead).toContain("ACP answer one")
    expect((handoffPrompt![handoffPrompt!.length - 1] as any).text).toBe("Follow-up in ACP")

    // Brand-new conversation (single user turn, nothing prior): raw thin prompt.
    const thinOnly = await (async () => {
      const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir")
      let capturedPrompt: any[] | undefined
      ;(bridge as any).ensureAcpSession = async (s: any) => {
        const fakeProc = {
          setModel: async () => {},
          setConfigOption: async () => {},
          prompt: async (_sid: string, prompt: any[]) => {
            capturedPrompt = prompt
            return { stopReason: "end_turn" }
          },
        }
        return { proc: fakeProc, acpSessionId: "acp-new", freshSession: true }
      }
      const lm = bridge.createLanguageModel("gemini-4-neon")
      const res = await lm.doStream({
        prompt: [{ role: "user", content: [{ type: "text", text: "Hello fresh thread" }] }],
        headers: { "x-opencode-session-id": "brand-new" },
      } as any)
      await readAllParts(res.stream)
      await bridge.disposeAll()
      return capturedPrompt as any[] | undefined
    })()
    expect(thinOnly).toHaveLength(1)
    expect((thinOnly![0] as any).text).toBe("Hello fresh thread")
  })

  test("concurrent starters join one in-flight start; invalidation yields exactly one retry install", async () => {
    const factoryCalls: any[] = []
    let openFirstInit!: () => void
    const makeFakeProc = () => {
      const callIndex = factoryCalls.length
      const state = { disposed: false, closed: false }
      let resolveFirstInit!: () => void
      const firstGate = new Promise<void>((r) => {
        resolveFirstInit = r
      })
      if (callIndex === 0) openFirstInit = () => resolveFirstInit()
      const fake: any = {
        get disposed() {
          return state.disposed
        },
        get isClosed() {
          return state.closed
        },
        initialize: () => (callIndex === 0 ? firstGate : Promise.resolve()),
        authenticate: async () => ({}),
        newSession: async () => ({ sessionId: `acp-fan-${callIndex}` }),
        resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
        setMode: async () => {},
        setConfigOption: async () => {},
        closeSession: async () => {},
        notifyCancel: () => {},
        cancel: async () => {},
        dispose: async () => {
          state.disposed = true
          state.closed = true
        },
      }
      factoryCalls.push(fake)
      return fake
    }
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: () => makeFakeProc(),
    })
    const s = (bridge as any).getOrCreateSession("fan-sess")

    // Two concurrent starters while the first initialize is gated: single spawn.
    const p1 = (bridge as any).ensureAcpSession(s)
    const p2 = (bridge as any).ensureAcpSession(s)
    await new Promise((r) => setTimeout(r, 20))
    expect(factoryCalls.length).toBe(1)

    // Invalidate mid-flight via a real reset (bumps generation, awaits the lock).
    const pr = bridge.resetSession("fan-sess")
    expect((bridge as any).getOrCreateSession("fan-sess").startGeneration).toBe(1)
    openFirstInit()
    await pr
    const [r1, r2] = await Promise.all([p1, p2])

    // Exactly one retry install; the stale proc was disposed, never installed.
    expect(factoryCalls.length).toBe(2)
    expect(factoryCalls[0].disposed).toBe(true)
    expect(factoryCalls[1].disposed).toBe(false)
    expect(r1.proc).toBe(factoryCalls[1])
    expect(r2.proc).toBe(factoryCalls[1])
    expect(r1.acpSessionId).toBe("acp-fan-1")
    expect((bridge as any).getOrCreateSession("fan-sess").proc).toBe(factoryCalls[1])
    // No orphan procs: only the installed one remains tracked.
    expect((bridge as any).activeProcs.size).toBe(1)
    expect((bridge as any).activeProcs.has(factoryCalls[1])).toBe(true)
    await bridge.disposeAll()
  })

  test("setCodeMode during startup invalidates the stale start and re-ensures with the new mode", async () => {
    const factoryCalls: any[][] = []
    let openFirstInitialize!: () => void
    const makeFakeProc = () => {
      const callIndex = factoryCalls.length
      const state = { disposed: false, newSessionServers: [] as any[], closed: false }
      let resolveFirstInit!: () => void
      const firstGate = new Promise<void>((r) => {
        resolveFirstInit = r
      })
      if (callIndex === 0) openFirstInitialize = () => resolveFirstInit()
      const fake: any = {
        newSessionServers: state.newSessionServers,
        get disposed() {
          return state.disposed
        },
        get isClosed() {
          return state.closed
        },
        // First start parks at initialize so the mode flip lands mid-startup;
        // retries resolve immediately.
        initialize: () => (callIndex === 0 ? firstGate : Promise.resolve()),
        authenticate: async () => ({}),
        newSession: async (params: any) => {
          state.newSessionServers.push(params.mcpServers)
          return { sessionId: `acp-${callIndex}` }
        },
        resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
        setMode: async () => {},
        setConfigOption: async () => {},
        closeSession: async () => {},
        notifyCancel: () => {},
        cancel: async () => {},
        dispose: async () => {
          state.disposed = true
        },
      }
      factoryCalls.push([fake])
      return fake
    }

    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: () => makeFakeProc(),
    })
    const s = (bridge as any).getOrCreateSession("race-sess")
    expect(s.startGeneration).toBe(0)

    // Kick off a startup with Code Mode off; it parks at initialize.
    const startP = (bridge as any).ensureAcpSession(s)
    await new Promise((r) => setTimeout(r, 20))
    expect(factoryCalls.length).toBe(1)

    // Flip the mode mid-startup: reset invalidates the in-flight start...
    const modeP = bridge.setCodeMode("race-sess", true)
    expect((bridge as any).getOrCreateSession("race-sess").startGeneration).toBe(1)
    // ...then let the stale start proceed; it must dispose itself, not install.
    openFirstInitialize()
    await modeP
    // The original ensure transparently retried and installed with the new mode.
    const ensured = await startP
    expect(factoryCalls[0][0].disposed).toBe(true)
    expect(factoryCalls.length).toBe(2)
    expect(ensured.acpSessionId).toBe("acp-1")
    expect(factoryCalls[1][0].disposed).toBe(false)
    // Retry installed with the NEW mode: single oc MCP server.
    expect(factoryCalls[1][0].newSessionServers).toHaveLength(1)
    expect(factoryCalls[1][0].newSessionServers[0]).toHaveLength(1)
    expect(factoryCalls[1][0].newSessionServers[0][0].name).toBe("oc")
    expect((bridge as any).getOrCreateSession("race-sess").proc).toBe(factoryCalls[1][0])
    expect(bridge.getState("race-sess").codeModeEnabled).toBe(true)
    await bridge.disposeAll()
  })

  test("warmSession(sessionID) pre-warms so the first turn reuses it without a second spawn", async () => {
    const factoryCalls: any[] = []
    const makeFakeProc = () => {
      const state = { disposed: false, closed: false, prompts: [] as any[] }
      const fake: any = {
        get disposed() {
          return state.disposed
        },
        get isClosed() {
          return state.closed
        },
        initialize: async () => ({}),
        authenticate: async () => ({}),
        newSession: async () => ({ sessionId: "acp-warm-1" }),
        resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
        setModel: async () => {},
        setMode: async () => {},
        setConfigOption: async () => {},
        closeSession: async () => {},
        notifyCancel: () => {},
        cancel: async () => {},
        prompt: async (_sid: string, prompt: any[]) => {
          state.prompts.push(prompt)
          return { stopReason: "end_turn" }
        },
        dispose: async () => {
          state.disposed = true
          state.closed = true
        },
        _prompts: state.prompts,
      }
      factoryCalls.push(fake)
      return fake
    }
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: () => makeFakeProc(),
    })

    const st = await bridge.warmSession("ses-warm-1")
    expect(st.sessionID).toBe("ses-warm-1")
    await waitFor(() => (bridge as any).getOrCreateSession("ses-warm-1").proc !== undefined)
    expect(factoryCalls.length).toBe(1)
    expect(bridge.getState("ses-warm-1").connected).toBe(true)

    // First real turn must reuse the warmed session — no second spawn.
    const lm = bridge.createLanguageModel("gemini-4-neon")
    const res = await lm.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "hello warmed ACP" }] }],
      headers: { "x-opencode-session-id": "ses-warm-1" },
    } as any)
    const parts = await readAllParts(res.stream)
    expect(parts.some((p) => p.type === "error")).toBe(false)
    expect(parts.find((p) => p.type === "finish")?.finishReason?.unified).toBe("stop")
    expect(factoryCalls.length).toBe(1)
    expect(factoryCalls[0]._prompts).toHaveLength(1)
    await bridge.disposeAll()
  })

  test("warmed empty tools/list refreshes on first real turn with same proc", async () => {
    const newSessionCalls: any[] = []
    let procInstance: any
    const makeFakeProc = () => {
      const state = { disposed: false, closed: false, prompts: [] as any[] }
      const fake: any = {
        get disposed() {
          return state.disposed
        },
        get isClosed() {
          return state.closed
        },
        initialize: async () => ({}),
        authenticate: async () => ({}),
        newSession: async (params: any) => {
          newSessionCalls.push(params)
          return { sessionId: `acp-sess-${newSessionCalls.length}` }
        },
        resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
        setModel: async () => {},
        setMode: async () => {},
        setConfigOption: async () => {},
        closeSession: async () => {},
        notifyCancel: () => {},
        cancel: async () => {},
        prompt: async (_sid: string, prompt: any[]) => {
          state.prompts.push(prompt)
          return { stopReason: "end_turn" }
        },
        dispose: async () => {
          state.disposed = true
          state.closed = true
        },
        _prompts: state.prompts,
      }
      procInstance = fake
      return fake
    }
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: () => makeFakeProc(),
    })

    // Warm creates the ACP session before contextSnapshot exists (empty tools).
    await bridge.warmSession("ses-tools-refresh")
    await waitFor(() => (bridge as any).getOrCreateSession("ses-tools-refresh").proc !== undefined)
    expect(newSessionCalls).toHaveLength(1)
    expect(newSessionCalls[0].mcpServers).toHaveLength(1)
    const warmedId = (bridge as any).getOrCreateSession("ses-tools-refresh").acpSessionId
    expect(warmedId).toBe("acp-sess-1")

    // First real turn populates the snapshot (context hook) with a real tool.
    // updateContextSnapshot runs before doStream in the real plugin (index.ts).
    bridge.updateContextSnapshot("ses-tools-refresh", {
      systemText: "",
      agent: "build",
      tools: { mytool: { description: "r", input: { type: "object" } } },
    })

    const lm = bridge.createLanguageModel("gemini-4-neon")
    const res = await lm.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "hello with tools" }] }],
      headers: { "x-opencode-session-id": "ses-tools-refresh" },
    } as any)
    const parts = await readAllParts(res.stream)
    expect(parts.some((p) => p.type === "error")).toBe(false)
    expect(parts.find((p) => p.type === "finish")?.finishReason?.unified).toBe("stop")
    // Same proc reused (no extra spawn) but ACP session refreshed for tools.
    expect(newSessionCalls).toHaveLength(2)
    const refreshedId = (bridge as any).getOrCreateSession("ses-tools-refresh").acpSessionId
    expect(refreshedId).toBe("acp-sess-2")
    expect(refreshedId).not.toBe(warmedId)
    expect((bridge as any).getOrCreateSession("ses-tools-refresh").proc).toBe(procInstance)
    await bridge.disposeAll()
  })

  test("warmSession('') parks a spare that the first real session claims; disposeAll cleans unclaimed spares", async () => {
    const factoryCalls: any[] = []
    const makeFakeProc = () => {
      const state = { disposed: false, closed: false, prompts: [] as any[] }
      const fake: any = {
        get disposed() {
          return state.disposed
        },
        get isClosed() {
          return state.closed
        },
        initialize: async () => ({}),
        authenticate: async () => ({}),
        newSession: async () => ({ sessionId: "acp-claimed-1" }),
        resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
        setMode: async () => {},
        setConfigOption: async () => {},
        closeSession: async () => {},
        notifyCancel: () => {},
        cancel: async () => {},
        prompt: async (_sid: string, prompt: any[]) => {
          state.prompts.push(prompt)
          return { stopReason: "end_turn" }
        },
        dispose: async () => {
          state.disposed = true
          state.closed = true
        },
        _prompts: state.prompts,
      }
      factoryCalls.push(fake)
      return fake
    }
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: () => makeFakeProc(),
    })

    await bridge.warmSession("")
    await waitFor(() => (bridge as any).warmSpare?.proc !== undefined)
    expect(factoryCalls.length).toBe(1)
    const spare = factoryCalls[0]
    expect(spare.disposed).toBe(false)

    // First real turn with a fresh ses_... id claims the spare — no new spawn.
    const lm = bridge.createLanguageModel("gemini-4-neon")
    const res = await lm.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "first message in new tab" }] }],
      headers: { "x-opencode-session-id": "ses-claim-1" },
    } as any)
    const parts = await readAllParts(res.stream)
    expect(parts.some((p) => p.type === "error")).toBe(false)
    expect(parts.find((p) => p.type === "finish")?.finishReason?.unified).toBe("stop")
    expect(factoryCalls.length).toBe(1)
    expect((bridge as any).warmSpare).toBeUndefined()
    expect((bridge as any).getOrCreateSession("ses-claim-1").proc).toBe(spare)
    expect(spare._prompts).toHaveLength(1)

    // Park another spare, then dispose: the unclaimed spare must be disposed, zero leaks.
    await bridge.warmSession("")
    await waitFor(() => (bridge as any).warmSpare?.proc !== undefined)
    expect(factoryCalls.length).toBe(2)
    const spare2 = factoryCalls[1]
    await bridge.disposeAll()
    expect(spare2.disposed).toBe(true)
    expect((bridge as any).warmSpare).toBeUndefined()
    expect((bridge as any).activeProcs.size).toBe(0)
  })

  test("warmSession stays quiet during circuit-breaker cooldown", async () => {
    let spawns = 0
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: () => {
        spawns++
        throw new Error("must not spawn during cooldown")
      },
    })
    ;(bridge as any).tripCircuitBreaker("RESOURCE_EXHAUSTED (code 429)")
    expect(bridge.isCircuitOpen()).toBe(true)
    await bridge.warmSession("ses-cool-1")
    await bridge.warmSession("")
    expect(spawns).toBe(0)
    await bridge.disposeAll()
  })

  test("warmSession does not reinstall after a reset invalidates it", async () => {
    const factoryCalls: any[] = []
    let openInit!: () => void
    const makeFakeProc = () => {
      const callIndex = factoryCalls.length
      const state = { disposed: false, closed: false }
      let resolveInit!: () => void
      const gate = new Promise<void>((r) => {
        resolveInit = r
      })
      if (callIndex === 0) openInit = () => resolveInit()
      const fake: any = {
        get disposed() {
          return state.disposed
        },
        get isClosed() {
          return state.closed
        },
        initialize: () => (callIndex === 0 ? gate : Promise.resolve()),
        authenticate: async () => ({}),
        newSession: async () => ({ sessionId: `acp-warm-${callIndex}` }),
        resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
        setMode: async () => {},
        setConfigOption: async () => {},
        closeSession: async () => {},
        notifyCancel: () => {},
        cancel: async () => {},
        dispose: async () => {
          state.disposed = true
          state.closed = true
        },
      }
      factoryCalls.push(fake)
      return fake
    }
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: () => makeFakeProc(),
    })

    // Kick off a background warm, then reset while it is still starting.
    const warmP = bridge.warmSession("ses-quiet-1")
    await waitFor(() => factoryCalls.length === 1)
    const resetP = bridge.resetSession("ses-quiet-1")
    expect((bridge as any).getOrCreateSession("ses-quiet-1").startGeneration).toBe(1)
    // Let the stale warm proceed: it must dispose itself and NOT reinstall.
    openInit()
    await resetP
    await warmP
    await new Promise((r) => setTimeout(r, 50))
    expect(factoryCalls.length).toBe(1)
    expect(factoryCalls[0].disposed).toBe(true)
    expect((bridge as any).getOrCreateSession("ses-quiet-1").proc).toBeUndefined()
    expect(bridge.getState("ses-quiet-1").connected).toBe(false)
    await bridge.disposeAll()
  })

  test("quotaBucketForModel maps models to their upstream quota pools", () => {
    expect(quotaBucketForModel("claude-opus-5-5")).toBe("claude-gpt")
    expect(quotaBucketForModel("claude-sonnet-5-5")).toBe("claude-gpt")
    expect(quotaBucketForModel("claude-opus-5-5-medium")).toBe("claude-gpt")
    expect(quotaBucketForModel("gpt-oss-120b-medium")).toBe("claude-gpt")
    expect(quotaBucketForModel("gemini-4-neon")).toBe("gemini")
    expect(quotaBucketForModel("gemini-4-hydrogen-high")).toBe("gemini")
    expect(quotaBucketForModel("gemini-pro-orchestrator-neon")).toBe("gemini")
    expect(quotaBucketForModel("gemini-3.8-flash")).toBe("gemini")
    expect(quotaBucketForModel("some-future-model")).toBe("other")
    expect(quotaBucketForModel("")).toBe("other")
  })

  test("quota trip on claude blocks its pool but preserves the session; gemini proceeds immediately", async () => {
    const factoryCalls: any[] = []
    let failNextPrompt: Error | undefined = new Error(
      "API error (attempt 1): RESOURCE_EXHAUSTED (code 429): Individual quota reached",
    )
    const makeFakeProc = () => {
      const state = { disposed: false, closed: false, prompts: [] as any[], setModels: [] as string[] }
      const fake: any = {
        get disposed() {
          return state.disposed
        },
        get isClosed() {
          return state.closed
        },
        initialize: async () => ({}),
        authenticate: async () => ({}),
        newSession: async () => ({ sessionId: "acp-quota-1" }),
        resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
        setModel: async (sid: string, modelId: string) => {
          state.setModels.push(modelId)
        },
        setMode: async () => {},
        setConfigOption: async () => {},
        closeSession: async () => {},
        notifyCancel: () => {},
        cancel: async () => {},
        prompt: async (_sid: string, prompt: any[]) => {
          state.prompts.push(prompt)
          if (failNextPrompt) {
            const err = failNextPrompt
            failNextPrompt = undefined
            throw err
          }
          return { stopReason: "end_turn" }
        },
        dispose: async () => {
          state.disposed = true
          state.closed = true
        },
        _state: state,
      }
      factoryCalls.push(fake)
      return fake
    }
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: () => makeFakeProc(),
    })

    // Turn 1 on claude-opus-5-5 hits pool quota.
    const lmClaude = bridge.createLanguageModel("claude-opus-5-5")
    const res1 = await lmClaude.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "hello claude" }] }],
      headers: { "x-opencode-session-id": "ses-quota-1" },
    } as any)
    const parts1 = await readAllParts(res1.stream)
    expect(parts1.some((p) => p.type === "error")).toBe(false)
    expect(parts1.find((p) => p.type === "finish")?.finishReason?.unified).toBe("stop")
    const text1 = parts1.filter((p) => p.type === "text-delta").map((p) => p.delta).join("")
    expect(text1).toContain("quota")
    expect(factoryCalls.length).toBe(1)

    // Proc, session id, and storage all survive a pool-quota halt.
    const s = (bridge as any).getOrCreateSession("ses-quota-1")
    expect(s.proc).toBe(factoryCalls[0])
    expect(s.proc.isClosed).toBe(false)
    expect(s.acpSessionId).toBe("acp-quota-1")
    expect(factoryCalls[0].disposed).toBe(false)
    expect(await (bridge as any).runStorageGet("ses-quota-1")).toBe("acp-quota-1")

    // Same-pool retry is gated without spawning or prompting.
    const promptsBefore = factoryCalls[0]._state.prompts.length
    const lmSonnet = bridge.createLanguageModel("claude-sonnet-5-5")
    const resBlocked = await lmSonnet.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "hello sonnet" }] }],
      headers: { "x-opencode-session-id": "ses-quota-1" },
    } as any)
    const blockedParts = await readAllParts(resBlocked.stream)
    expect(blockedParts.find((p) => p.type === "finish")?.finishReason?.raw).toBe("circuit_open")
    expect(factoryCalls.length).toBe(1)
    expect(factoryCalls[0]._state.prompts.length).toBe(promptsBefore)
    // Session-bucket cooldown is exposed for the GUI banner.
    expect(bridge.getState("ses-quota-1").cooldownUntil).toBeGreaterThan(Date.now())

    // Other pool proceeds immediately on the SAME live session.
    const lmGemini = bridge.createLanguageModel("gemini-4-hydrogen")
    const resOk = await lmGemini.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "hello gemini" }] }],
      headers: { "x-opencode-session-id": "ses-quota-1" },
    } as any)
    const okParts = await readAllParts(resOk.stream)
    expect(okParts.some((p) => p.type === "error")).toBe(false)
    expect(okParts.find((p) => p.type === "finish")?.finishReason?.unified).toBe("stop")
    expect(factoryCalls.length).toBe(1)
    expect((bridge as any).getOrCreateSession("ses-quota-1").proc).toBe(factoryCalls[0])
    expect(factoryCalls[0]._state.prompts.length).toBe(promptsBefore + 1)
    expect(factoryCalls[0]._state.setModels).toContain("gemini-4-hydrogen-high")
    await bridge.disposeAll()
  })

  test("fatal 403 trips the global breaker and tears the session down", async () => {
    const factoryCalls: any[] = []
    const makeFakeProc = () => {
      const state = { disposed: false, closed: false, prompts: [] as any[] }
      const fake: any = {
        get disposed() {
          return state.disposed
        },
        get isClosed() {
          return state.closed
        },
        initialize: async () => ({}),
        authenticate: async () => ({}),
        newSession: async () => ({ sessionId: "acp-fatal-1" }),
        resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
        setModel: async () => {},
        setMode: async () => {},
        setConfigOption: async () => {},
        closeSession: async () => {},
        notifyCancel: () => {},
        cancel: async () => {},
        killImmediate: () => {
          state.closed = true
        },
        prompt: async () => {
          throw new Error("API error (attempt 1): PERMISSION_DENIED (code 403): access denied")
        },
        dispose: async () => {
          state.disposed = true
          state.closed = true
        },
      }
      factoryCalls.push(fake)
      return fake
    }
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: () => makeFakeProc(),
    })

    const lm = bridge.createLanguageModel("gemini-4-neon")
    const res = await lm.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
      headers: { "x-opencode-session-id": "ses-fatal-1" },
    } as any)
    const parts = await readAllParts(res.stream)
    expect(parts.some((p) => p.type === "error")).toBe(false)
    expect(parts.find((p) => p.type === "finish")?.finishReason?.raw).toBe("api_error_halted")

    // Global path kills the proc, clears the session id + storage.
    const s = (bridge as any).getOrCreateSession("ses-fatal-1")
    expect(s.proc).toBeUndefined()
    expect(s.acpSessionId).toBeUndefined()
    expect(factoryCalls[0].isClosed).toBe(true)
    expect(await (bridge as any).runStorageGet("ses-fatal-1")).toBeUndefined()
    expect(bridge.getState("ses-fatal-1").cooldownUntil).toBeGreaterThan(Date.now())

    // Global cooldown blocks every pool — the turn never reaches prompt.
    const resBlocked = await bridge.createLanguageModel("claude-opus-5-5").doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "hello again" }] }],
      headers: { "x-opencode-session-id": "ses-fatal-1" },
    } as any)
    const blockedParts = await readAllParts(resBlocked.stream)
    expect(blockedParts.find((p) => p.type === "finish")?.finishReason?.raw).toBe("circuit_open")
    expect(factoryCalls.length).toBe(1)
    await bridge.disposeAll()
  })

  test("sanitizes synthetic restart and cross-provider parts safely", () => {
    expect(
      isSyntheticRestartAfterError([
        {
          role: "assistant",
          content: [{ type: "text", text: "Antigravity ACP halted immediately after 1st upstream error" }],
        },
        {
          role: "user",
          content: [{ type: "text", text: "The server restarted while you were working. Continue." }],
        },
      ] as any),
    ).toBe(true)

    const nonAcpMsgs = [
      {
        role: "assistant",
        content: [
          {
            type: "tool-call",
            name: "acp_bash",
            input: { command: "ls" },
            providerExecuted: true,
            providerMetadata: { "antigravity-acp": { acp: true } },
          },
        ],
      },
    ]
    sanitizeForNonAcpModel(nonAcpMsgs)
    expect(nonAcpMsgs[0].content[0].type).toBe("text")

    const acpMsgs = [
      {
        role: "assistant",
        content: [
          { type: "compaction", text: "Summary" },
          { type: "effort", value: "high" },
        ],
      },
    ]
    sanitizeForAcpModel(acpMsgs)
    expect(acpMsgs[0].content).toHaveLength(1)
    expect(acpMsgs[0].content[0].type).toBe("text")
  })
})

describe("CodeRabbit review findings: races, handoffs, scrubbing, budgets", () => {
  const SESSION_KEY_PREFIX = "antigravity-acp:session:"

  function makeIdleFakeProc(onPrompt?: (prompt: any[]) => Promise<any>) {
    const state = {
      disposed: false,
      closed: false,
      prompts: [] as any[],
      setModels: [] as string[],
    }
    const fake: any = {
      get disposed() {
        return state.disposed
      },
      get isClosed() {
        return state.closed
      },
      initialize: async () => ({}),
      authenticate: async () => ({}),
      newSession: async () => ({ sessionId: `acp-test-${Math.random().toString(36).slice(2, 8)}` }),
      resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
      setModel: async (_sid: string, modelId: string) => {
        state.setModels.push(modelId)
      },
      setMode: async () => {},
      setConfigOption: async () => {},
      closeSession: async () => {},
      notifyCancel: () => {},
      cancel: async () => {},
      prompt: async (_sid: string, prompt: any[]) => {
        state.prompts.push(prompt)
        if (onPrompt) return onPrompt(prompt)
        return { stopReason: "end_turn" }
      },
      dispose: async () => {
        state.disposed = true
        state.closed = true
      },
      _state: state,
    }
    return fake
  }

  test("finding 1: stale-start retry waits for reset cleanup before reading storage", async () => {
    const events: string[] = []
    const SESSION_ID = "ses-race-barrier"
    const SESSION_KEY = `${SESSION_KEY_PREFIX}${SESSION_ID}`
    let releaseRemove!: () => void
    const removeGate = new Promise<void>((r) => {
      releaseRemove = r
    })
    const ctx = makeMockCtx({ [SESSION_KEY]: "acp-stale" })
    const rawGet = ctx.storage.get.bind(ctx.storage)
    const rawRemove = ctx.storage.remove.bind(ctx.storage)
    ctx.storage.get = ((key: string) => {
      if (String(key) === SESSION_KEY) events.push(`get:${key}`)
      return rawGet(key)
    }) as any
    ctx.storage.remove = ((key: string) => {
      if (String(key) !== SESSION_KEY) return rawRemove(key)
      return Effect.promise(async () => {
        events.push(`remove-start:${key}`)
        await removeGate
        events.push(`remove-end:${key}`)
        await Effect.runPromise(rawRemove(key) as Effect.Effect<unknown>)
      }) as any
    }) as any

    const factoryCalls: any[] = []
    let openInit!: () => void
    const bridge = new SessionBridgeManager(ctx, "/tmp/acp-test-dir", undefined, {
      createProc: () => {
        const callIndex = factoryCalls.length
        const state = { disposed: false, closed: false }
        let resolveInit!: () => void
        const gate = new Promise<void>((r) => {
          resolveInit = r
        })
        if (callIndex === 0) openInit = () => resolveInit()
        const fake: any = {
          get disposed() {
            return state.disposed
          },
          get isClosed() {
            return state.closed
          },
          initialize: () => (callIndex === 0 ? gate : Promise.resolve()),
          authenticate: async () => ({}),
          newSession: async () => ({ sessionId: `acp-race-${callIndex}` }),
          resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
          setMode: async () => {},
          setConfigOption: async () => {},
          closeSession: async () => {},
          notifyCancel: () => {},
          cancel: async () => {},
          prompt: async () => ({ stopReason: "end_turn" }),
          dispose: async () => {
            state.disposed = true
            state.closed = true
          },
        }
        factoryCalls.push(fake)
        return fake
      },
    })

    // Start 1 parks at initialize; a reset invalidates it mid-flight.
    const s = (bridge as any).getOrCreateSession(SESSION_ID)
    const startP = (bridge as any).ensureAcpSession(s)
    await waitFor(() => factoryCalls.length === 1)
    const resetP = bridge.resetSession(SESSION_ID)
    expect((bridge as any).getOrCreateSession(SESSION_ID).startGeneration).toBe(1)
    openInit()
    await new Promise((r) => setTimeout(r, 50))
    // The stale start disposed itself; the retry is parked on the reset
    // barrier — no second spawn, and storage hasn't been read yet.
    expect(factoryCalls.length).toBe(1)
    expect(factoryCalls[0].disposed).toBe(true)
    expect(events.filter((e) => e === `get:${SESSION_KEY}`).length).toBe(0)
    expect(events).toContain(`remove-start:${SESSION_KEY}`)
    // Release cleanup; the retry may now read storage and spawn fresh.
    releaseRemove()
    const ensured = await startP
    await resetP
    expect(factoryCalls.length).toBe(2)
    expect(ensured.proc).toBe(factoryCalls[1])
    expect((bridge as any).getOrCreateSession(SESSION_ID).proc).toBe(factoryCalls[1])
    const readIdx = events.findIndex((e) => e === `get:${SESSION_KEY}`)
    const removeEndIdx = events.indexOf(`remove-end:${SESSION_KEY}`)
    expect(removeEndIdx).toBeGreaterThanOrEqual(0)
    expect(readIdx).toBeGreaterThan(removeEndIdx)
    await bridge.disposeAll()
  })

  test("finding 2: concurrent cooldown loads share one in-flight read", async () => {
    const getCalls: string[] = []
    let releaseReads!: () => void
    const readGate = new Promise<void>((r) => {
      releaseReads = r
    })
    const ctx = makeMockCtx({
      "antigravity-acp:cooldown:global": { until: Date.now() + 60_000, reason: "test-quota" },
    })
    const rawGet = ctx.storage.get.bind(ctx.storage)
    ctx.storage.get = ((key: string) => {
      getCalls.push(String(key))
      return Effect.promise(async () => {
        await readGate
        return await Effect.runPromise(rawGet(key))
      }) as any
    }) as any
    let spawns = 0
    const bridge = new SessionBridgeManager(ctx, "/tmp/acp-test-dir", undefined, {
      createProc: (() => {
        spawns++
        throw new Error("must not spawn during cooldown")
      }) as any,
    })
    const p1 = bridge.warmSession("ses-shared-a")
    const p2 = bridge.warmSession("ses-shared-b")
    await new Promise((r) => setTimeout(r, 50))
    // Both wait on the same in-flight load: no proc spawned while hydrating.
    expect(spawns).toBe(0)
    releaseReads()
    const [s1, s2] = await Promise.all([p1, p2])
    expect(s1.cooldownUntil).not.toBeNull()
    expect(s2.cooldownUntil).not.toBeNull()
    // Exactly one read pass over the cooldown keys (legacy + 4 buckets).
    expect(getCalls.filter((k) => k.startsWith("antigravity-acp:cooldown")).length).toBe(5)
    await bridge.disposeAll()
  })

  test("finding 3: idle warmed sessions are LRU-capped; active turns are never evicted", async () => {
    const factoryCalls: any[] = []
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: () => {
        const fake = makeIdleFakeProc()
        factoryCalls.push(fake)
        return fake as any
      },
    })
    for (const sid of ["w1", "w2", "w3"]) {
      await bridge.warmSession(sid)
      await waitFor(() => bridge.getState(sid).connected)
    }
    // Cap is 2: the oldest idle session was torn down.
    await waitFor(() => factoryCalls[0].disposed)
    expect(bridge.getState("w1").connected).toBe(false)
    expect(bridge.getState("w2").connected).toBe(true)
    expect(bridge.getState("w3").connected).toBe(true)
    // Pin w2 with an active turn: eviction must skip it.
    ;(bridge as any).getOrCreateSession("w2").activeTurn = { bufferedParts: [] } as any
    for (const sid of ["w4", "w5"]) {
      await bridge.warmSession(sid)
      await waitFor(() => bridge.getState(sid).connected)
    }
    await waitFor(() => factoryCalls[3].disposed)
    expect(bridge.getState("w2").connected).toBe(true)
    expect(factoryCalls[1].disposed).toBe(false)
    expect(bridge.getState("w5").connected).toBe(true)
    expect(bridge.getState("w4").connected).toBe(false)
    expect(factoryCalls[3].disposed).toBe(true)
    expect(factoryCalls[2].disposed).toBe(true)
    ;(bridge as any).getOrCreateSession("w2").activeTurn = undefined
    await bridge.disposeAll()
  })

  test("finding 4: warm-installed fresh session triggers text handoff on first turn", async () => {
    const SESSION_ID = "ses-handoff-warm"
    const ctx = makeMockCtx({ [`antigravity-acp:session:${SESSION_ID}`]: "acp-stale" })
    let captured: any[] | undefined
    const bridge = new SessionBridgeManager(ctx, "/tmp/acp-test-dir", undefined, {
      createProc: () => {
        const fake = makeIdleFakeProc((prompt) => {
          captured = prompt
          return Promise.resolve({ stopReason: "end_turn" })
        })
        fake.resumeSession = async () => {
          throw new Error("session gone")
        }
        return fake as any
      },
    })
    await bridge.warmSession(SESSION_ID)
    await waitFor(() => bridge.getState(SESSION_ID).connected)
    // Resume failed during warm → fresh install armed the handoff flag.
    expect((bridge as any).getOrCreateSession(SESSION_ID).pendingFreshHandoff).toBe(true)

    const acpPrompt = [
      { role: "user", content: [{ type: "text", text: "First ACP question" }] },
      {
        role: "assistant",
        content: [{ type: "text", text: "ACP answer one" }],
        providerOptions: { "antigravity-acp": { acp: true } },
      },
      { role: "user", content: [{ type: "text", text: "Back to ACP now" }] },
    ] as any
    const lm = bridge.createLanguageModel("gemini-4-neon")
    const res = await lm.doStream({ prompt: acpPrompt, headers: { "x-opencode-session-id": SESSION_ID } } as any)
    const parts = await readAllParts(res.stream)
    expect(parts.find((p) => p.type === "finish")?.finishReason?.unified).toBe("stop")
    const head = (captured ?? []).find((b) => b.type === "text")?.text ?? ""
    expect(head).toContain(TEXT_ONLY_HANDOFF_HEADER)
    expect(head).toContain("First ACP question")
    expect(head).toContain("ACP answer one")
    expect((captured ?? [])[(captured ?? []).length - 1].text).toBe("Back to ACP now")
    // The flag is consumed by the first real turn.
    expect((bridge as any).getOrCreateSession(SESSION_ID).pendingFreshHandoff).toBeFalsy()
    await bridge.disposeAll()

    // Control: resume succeeds → no flag, raw thin prompt.
    let captured2: any[] | undefined
    const ctx2 = makeMockCtx({ "antigravity-acp:session:ses-handoff-ok": "acp-live" })
    const bridge2 = new SessionBridgeManager(ctx2, "/tmp/acp-test-dir", undefined, {
      createProc: () => {
        const fake = makeIdleFakeProc((prompt) => {
          captured2 = prompt
          return Promise.resolve({ stopReason: "end_turn" })
        })
        return fake as any
      },
    })
    await bridge2.warmSession("ses-handoff-ok")
    await waitFor(() => bridge2.getState("ses-handoff-ok").connected)
    expect((bridge2 as any).getOrCreateSession("ses-handoff-ok").pendingFreshHandoff).toBeFalsy()
    const lm2 = bridge2.createLanguageModel("gemini-4-neon")
    const res2 = await lm2.doStream({
      prompt: acpPrompt,
      headers: { "x-opencode-session-id": "ses-handoff-ok" },
    } as any)
    await readAllParts(res2.stream)
    expect(captured2).toHaveLength(1)
    expect(JSON.stringify(captured2)).not.toContain(TEXT_ONLY_HANDOFF_HEADER)
    await bridge2.disposeAll()
  })

  test("finding 5: file parts accept data URIs and raw inline base64, reject URLs/paths/plain text", () => {
    const rawB64 = Buffer.from("x".repeat(48)).toString("base64")
    expect(rawB64.length % 4).toBe(0)
    const paddedB64 = Buffer.from("y".repeat(47)).toString("base64")
    expect(paddedB64).toContain("=")
    const blocks = buildThinAcpPromptBlocks(
      [
        {
          role: "user",
          content: [
            { type: "text", text: "look" },
            { type: "file", data: `data:image/png;base64,${rawB64}`, mediaType: "image/png", filename: "a.png" },
            { type: "file", data: rawB64, mediaType: "image/png", filename: "b.png" },
            { type: "file", data: paddedB64, mediaType: "image/png", filename: "b2.png" },
            { type: "file", data: "https://example.com/i.png", mediaType: "image/png", filename: "c.png" },
            { type: "file", data: "http://example.com/i.png", mediaType: "image/png", filename: "c2.png" },
            { type: "file", data: "/tmp/i.png", mediaType: "image/png", filename: "d.png" },
            { type: "file", data: "hello", mediaType: "text/plain", filename: "e.txt" },
          ],
        },
      ] as any,
    )
    const images = blocks.filter((b) => b.type === "image")
    expect(images).toHaveLength(3)
    expect(images.map((b: any) => b.mimeType)).toEqual(["image/png", "image/png", "image/png"])
    const texts = blocks
      .filter((b) => b.type === "text")
      .map((b) => (b as any).text)
      .join("\n")
    expect(texts).toContain("[Attached file: c.png")
    expect(texts).toContain("[Attached file: c2.png")
    expect(texts).toContain("[Attached file: d.png")
    expect(texts).toContain("[Attached file: e.txt")
    expect(JSON.stringify(blocks)).not.toContain("https://example.com")
  })

  test("finding 6: follow-up turn waits for a cancelled prompt to settle before sending a new prompt", async () => {
    const order: string[] = []
    let resolveFirst!: (v: any) => void
    const factoryCalls: any[] = []
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: () => {
        const fake: any = {
          get isClosed() {
            return false
          },
          initialize: async () => ({}),
          authenticate: async () => ({}),
          newSession: async () => ({ sessionId: "acp-settle-1" }),
          resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
          setModel: async () => {},
          setMode: async () => {},
          setConfigOption: async () => {},
          closeSession: async () => {},
          notifyCancel: () => {},
          cancel: async () => {},
          prompt: async () => {
            if (order.length === 0) {
              order.push("p1-sent")
              const v = await new Promise<any>((r) => {
                resolveFirst = r
              })
              order.push("p1-settled")
              return v
            }
            order.push("p2-sent")
            return { stopReason: "end_turn" }
          },
          dispose: async () => {},
        }
        factoryCalls.push(fake)
        return fake
      },
    })
    const lm = bridge.createLanguageModel("gemini-4-neon")
    const sid = "ses-settle-order"
    const r1 = await lm.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "one" }] }],
      headers: { "x-opencode-session-id": sid },
    } as any)
    await waitFor(() => order.length === 1)
    // Turn 2 starts while turn 1's prompt is still unsettled server-side.
    const p2 = lm.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "two" }] }],
      headers: { "x-opencode-session-id": sid },
    } as any)
    await new Promise((r) => setTimeout(r, 100))
    expect(order).toEqual(["p1-sent"])
    expect(factoryCalls.length).toBe(1)
    resolveFirst({ stopReason: "end_turn" })
    const r2 = await p2
    const parts2 = await readAllParts(r2.stream)
    expect(parts2.find((p) => p.type === "finish")?.finishReason?.unified).toBe("stop")
    expect(order).toEqual(["p1-sent", "p1-settled", "p2-sent"])
    expect(factoryCalls.length).toBe(1)
    // The superseded turn's stream terminates with an error part instead of
    // hanging forever (regression: interrupt used to orphan the controller).
    const r1parts = await readAllParts(r1.stream)
    expect(r1parts.some((p) => p.type === "error")).toBe(true)
    await bridge.disposeAll()
  })

  test("finding 6: wedged prompt is torn down after the settle timeout so the next turn starts fresh", async () => {
    const factoryCalls: any[] = []
    let promptCalls = 0
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: () => {
        const state = { disposed: false, closed: false }
        const fake: any = {
          get disposed() {
            return state.disposed
          },
          get isClosed() {
            return state.closed
          },
          initialize: async () => ({}),
          authenticate: async () => ({}),
          newSession: async () => ({ sessionId: `acp-hang-${factoryCalls.length}` }),
          resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
          setModel: async () => {},
          setMode: async () => {},
          setConfigOption: async () => {},
          closeSession: async () => {},
          notifyCancel: () => {},
          cancel: async () => {},
          prompt: async () => {
            promptCalls++
            if (promptCalls === 1) {
              // Only the very first prompt wedges forever: the replacement
              // proc must answer normally so the follow-up turn can finish.
              await new Promise(() => {})
            }
            return { stopReason: "end_turn" }
          },
          dispose: async () => {
            state.disposed = true
            state.closed = true
          },
        }
        factoryCalls.push(fake)
        return fake
      },
    })
    const lm = bridge.createLanguageModel("gemini-4-neon")
    const sid = "ses-settle-hang"
    const r1 = await lm.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "one" }] }],
      headers: { "x-opencode-session-id": sid },
    } as any)
    void readAllParts(r1.stream).catch(() => {})
    const t0 = Date.now()
    const r2 = await lm.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "two" }] }],
      headers: { "x-opencode-session-id": sid },
    } as any)
    const parts2 = await readAllParts(r2.stream)
    // Bounded wait (not forever): the wedged proc is torn down and replaced.
    expect(Date.now() - t0).toBeLessThan(10000)
    expect(parts2.find((p) => p.type === "finish")?.finishReason?.unified).toBe("stop")
    expect(factoryCalls.length).toBe(2)
    expect(factoryCalls[0].disposed).toBe(true)
    expect((bridge as any).getOrCreateSession(sid).proc).toBe(factoryCalls[1])
    expect(bridge.getState(sid).connected).toBe(true)
    await bridge.disposeAll()
  }, 15000)

  test("finding 7: local status streams carry localStatus marker and never suppress handoff", async () => {
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir")
    ;(bridge as any).tripCircuitBreaker("RESOURCE_EXHAUSTED (code 429)", "gemini")
    const lm = bridge.createLanguageModel("gemini-4-neon")
    const res = await lm.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
      headers: { "x-opencode-session-id": "ses-local-marker" },
    } as any)
    const parts = await readAllParts(res.stream)
    expect(parts.find((p) => p.type === "finish")?.finishReason?.raw).toBe("circuit_open")
    for (const p of parts) {
      if (p.type === "text-start" || p.type === "text-end" || p.type === "finish") {
        expect((p as any).providerMetadata?.["antigravity-acp"]?.localStatus).toBe(true)
      }
    }
    await bridge.disposeAll()
    // A history containing only a localStatus-marked cooldown notice is skipped:
    // neither a foreign handoff nor an ACP checkpoint.
    expect(
      hasUnseenNonAcpAssistantHistory([
        { role: "user", content: [{ type: "text", text: "hello" }] },
        {
          role: "assistant",
          content: [
            {
              type: "text",
              text: "Antigravity ACP cooldown active (12s remaining)",
              providerMetadata: { "antigravity-acp": { localStatus: true } },
            },
          ],
        },
        { role: "user", content: [{ type: "text", text: "now?" }] },
      ] as any),
    ).toBe(false)
  })

  test("finding 10: context snapshot rebuilds when descriptions or schemas change", async () => {
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir")
    await bridge.setCodeMode("sess-sig", true)
    const inputA = {
      description: "does A",
      input: { type: "object", properties: { x: { type: "string" } } },
    }
    bridge.updateContextSnapshot("sess-sig", {
      systemText: "",
      agent: "build",
      tools: { mytool: inputA },
    } as any)
    const snapA = (bridge as any).sessions.get("sess-sig").contextSnapshot
    expect(snapA.mcpDescriptors).toHaveLength(1)
    expect(snapA.mcpDescriptors[0].description).toBe("does A")
    // Same keys, new description → must rebuild (old signature ignored it).
    bridge.updateContextSnapshot("sess-sig", {
      systemText: "",
      agent: "build",
      tools: { mytool: { description: "does B", input: inputA.input } },
    } as any)
    const snapB = (bridge as any).sessions.get("sess-sig").contextSnapshot
    expect(snapB).not.toBe(snapA)
    expect(snapB.mcpDescriptors[0].description).toBe("does B")
    // Same keys + description, new schema → must rebuild.
    const inputC = { type: "object", properties: { y: { type: "number" } } }
    bridge.updateContextSnapshot("sess-sig", {
      systemText: "",
      agent: "build",
      tools: { mytool: { description: "does B", input: inputC } },
    } as any)
    const snapC = (bridge as any).sessions.get("sess-sig").contextSnapshot
    expect(snapC).not.toBe(snapB)
    expect(snapC.mcpDescriptors[0].inputSchema).toEqual(inputC)
    // Identical input → no rebuild (same snapshot object).
    bridge.updateContextSnapshot("sess-sig", {
      systemText: "",
      agent: "build",
      tools: { mytool: { description: "does B", input: inputC } },
    } as any)
    expect((bridge as any).sessions.get("sess-sig").contextSnapshot).toBe(snapC)
    await bridge.disposeAll()
  })

  test("finding 11: buildAgySpawnEnv scrubs case-insensitively on win32 only", () => {
    const src = {
      ANTIGRAVITY_ACP_AUTH_METHOD: "oauth-personal",
      gemini_api_key: "k",
      GOOGLE_api_KEY: "g",
      OTHER: "x",
    } as any
    const win = buildAgySpawnEnv(src, "win32")
    expect(win.gemini_api_key).toBeUndefined()
    expect(win["GOOGLE_api_KEY"]).toBeUndefined()
    expect(win.OTHER).toBe("x")
    const nix = buildAgySpawnEnv(src, "linux")
    // Exact match on Unix: mixed-case keys survive.
    expect(nix.gemini_api_key).toBe("k")
    expect(nix["GOOGLE_api_KEY"]).toBe("g")
    // Non-OAuth methods are untouched even on win32.
    const keyWin = buildAgySpawnEnv(
      { ANTIGRAVITY_ACP_AUTH_METHOD: "gemini-api-key", gemini_api_key: "k" } as any,
      "win32",
    )
    expect(keyWin.gemini_api_key).toBe("k")
  })

  test("finding 12: handoff block (excluding latest prompt) never exceeds maxChars", () => {
    const userText = "y".repeat(400)
    const longPrompt = "z".repeat(1000)
    const blocks = buildTextOnlyHandoffPromptBlocks(
      [
        { role: "user", content: [{ type: "text", text: userText }] },
        { role: "assistant", content: [{ type: "text", text: userText }] },
        { role: "user", content: [{ type: "text", text: longPrompt }] },
      ] as any,
      { maxChars: 500 },
    )
    const head = (blocks[0] as any).text as string
    expect(head.startsWith(TEXT_ONLY_HANDOFF_HEADER)).toBe(true)
    expect(head.length).toBeLessThanOrEqual(500)
    // Latest user prompt is exempt from the cap and intact.
    expect((blocks[blocks.length - 1] as any).text).toBe(longPrompt)
    // Single oversized line is truncated with the marker, still within budget.
    const huge = "q".repeat(2000)
    const capped = buildTextOnlyHandoffPromptBlocks(
      [
        { role: "user", content: [{ type: "text", text: "q?" }] },
        { role: "assistant", content: [{ type: "text", text: huge }] },
        { role: "user", content: [{ type: "text", text: "go" }] },
      ] as any,
      { maxChars: 300 },
    )
    const chead = (capped[0] as any).text as string
    expect(chead).toContain("\n... [truncated]")
    expect(chead.length).toBeLessThanOrEqual(300)
    expect((capped[capped.length - 1] as any).text).toBe("go")
    // Degenerate budget: no handoff block, just the live prompt.
    const tiny = buildTextOnlyHandoffPromptBlocks(
      [
        { role: "user", content: [{ type: "text", text: "q?" }] },
        { role: "assistant", content: [{ type: "text", text: huge }] },
        { role: "user", content: [{ type: "text", text: "go" }] },
      ] as any,
      { maxChars: 10 },
    )
    expect(tiny).toHaveLength(1)
    expect((tiny[0] as any).text).toBe("go")
  })
})

describe("Remaining fixes regression", () => {
  afterEach(() => {
    resetSharedCatalogForTests()
  })

  test("/9j/ JPEG base64 accepted, filesystem paths rejected", () => {
    // Real JPEG base64 starts with /9j/ — must be accepted as inline image bytes.
    const jpegB64 = "/9j/" + "A".repeat(28)
    expect(jpegB64.length).toBe(32)
    expect(jpegB64.length % 4).toBe(0)
    const blocks = buildThinAcpPromptBlocks(
      [
        {
          role: "user",
          content: [
            { type: "text", text: "look" },
            { type: "file", data: jpegB64, mediaType: "image/png", filename: "photo.jpg" },
            { type: "file", data: "/tmp/foo.png", mediaType: "image/png", filename: "a.png" },
            { type: "file", data: "/Users/a/b.png", mediaType: "image/png", filename: "b.png" },
            { type: "file", data: "./img.png", mediaType: "image/png", filename: "c.png" },
            { type: "file", data: "file:///tmp/x.png", mediaType: "image/png", filename: "d.png" },
            { type: "file", data: "https://example.com/i.png", mediaType: "image/png", filename: "e.png" },
          ],
        },
      ] as any,
    )
    const images = blocks.filter((b) => b.type === "image")
    expect(images).toHaveLength(1)
    expect((images[0] as any).data).toBe(jpegB64)
    const text = blocks
      .filter((b) => b.type === "text")
      .map((b) => (b as any).text)
      .join("\n")
    expect(text).toContain("[Attached file: a.png")
    expect(text).toContain("[Attached file: b.png")
    expect(text).toContain("[Attached file: c.png")
    expect(text).toContain("[Attached file: d.png")
    expect(text).toContain("[Attached file: e.png")
    expect(JSON.stringify(blocks)).not.toContain("/tmp/foo.png")
    expect(JSON.stringify(blocks)).not.toContain("https://example.com")
  })

  test("LRU eviction preserves persisted acpSessionId in storage", async () => {
    const factoryCalls: any[] = []
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: () => {
        const idx = factoryCalls.length
        const state = { disposed: false, closed: false }
        const fake: any = {
          get disposed() {
            return state.disposed
          },
          get isClosed() {
            return state.closed
          },
          initialize: async () => ({}),
          authenticate: async () => ({}),
          newSession: async () => ({ sessionId: `acp-lru-${idx}` }),
          resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
          setMode: async () => {},
          setConfigOption: async () => {},
          closeSession: async () => {},
          notifyCancel: () => {},
          cancel: async () => {},
          prompt: async () => ({ stopReason: "end_turn" }),
          dispose: async () => {
            state.disposed = true
            state.closed = true
          },
        }
        factoryCalls.push(fake)
        return fake
      },
    })
    for (const sid of ["lr1", "lr2", "lr3"]) {
      await bridge.warmSession(sid)
      await waitFor(() => bridge.getState(sid).connected)
    }
    await waitFor(() => factoryCalls[0].disposed)
    // Oldest proc disposed, session disconnected...
    expect(bridge.getState("lr1").connected).toBe(false)
    // ...but the persisted acpSessionId survives eviction (no resetSession wipe).
    const s1 = (bridge as any).getOrCreateSession("lr1")
    expect(s1.proc).toBeUndefined()
    expect(s1.acpSessionId).toBe("acp-lru-0")
    expect(await (bridge as any).runStorageGet("lr1")).toBe("acp-lru-0")
    expect(bridge.getState("lr2").connected).toBe(true)
    expect(bridge.getState("lr3").connected).toBe(true)
    await bridge.disposeAll()
  })

  test("foreground reservation protects against LRU eviction during setModel", async () => {
    const factoryCalls: any[] = []
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: () => {
        const idx = factoryCalls.length
        const state = { disposed: false, closed: false }
        const fake: any = {
          get disposed() {
            return state.disposed
          },
          get isClosed() {
            return state.closed
          },
          initialize: async () => ({}),
          authenticate: async () => ({}),
          newSession: async () => ({ sessionId: `acp-fg-${idx}` }),
          resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
          setMode: async () => {},
          setConfigOption: async () => {},
          setModel: async () => {
            await new Promise((r) => setTimeout(r, 50))
          },
          closeSession: async () => {},
          notifyCancel: () => {},
          cancel: async () => {},
          prompt: async () => ({ stopReason: "end_turn" }),
          dispose: async () => {
            state.disposed = true
            state.closed = true
          },
        }
        factoryCalls.push(fake)
        return fake
      },
    })
    await bridge.warmSession("fg1")
    await waitFor(() => bridge.getState("fg1").connected)
    await bridge.warmSession("fg2")
    await waitFor(() => bridge.getState("fg2").connected)
    // Simulate a doStream foreground reservation on the oldest session.
    const s1 = (bridge as any).getOrCreateSession("fg1")
    s1.foregroundCount = 1
    await bridge.warmSession("fg3")
    await waitFor(() => bridge.getState("fg3").connected)
    await new Promise((r) => setTimeout(r, 50))
    // fg1 reserved: eviction must skip it and take fg2 instead.
    expect(bridge.getState("fg1").connected).toBe(true)
    expect(factoryCalls[0].disposed).toBe(false)
    expect(bridge.getState("fg3").connected).toBe(true)
    await waitFor(() => factoryCalls[1].disposed)
    expect(bridge.getState("fg2").connected).toBe(false)
    // Storage for the evicted idle session is still preserved.
    expect(await (bridge as any).runStorageGet("fg2")).toBe("acp-fg-1")
    s1.foregroundCount = 0
    await bridge.disposeAll()
  })

  test("local cooldown notices create neither foreign handoff nor ACP checkpoint", async () => {
    const local = (text: string) => ({
      role: "assistant",
      content: [
        { type: "text", text, providerMetadata: { "antigravity-acp": { localStatus: true } } },
      ],
    })
    const acpReply = (text: string) => ({
      role: "assistant",
      content: [{ type: "text", text }],
      providerOptions: { "antigravity-acp": { acp: true } },
    })
    const foreign = (text: string) => ({
      role: "assistant",
      content: [{ type: "text", text }],
    })
    const user = (text: string) => ({ role: "user", content: [{ type: "text", text }] })

    // Local-only: skipped entirely → no handoff.
    expect(hasUnseenNonAcpAssistantHistory([user("hi"), local("cooldown 12s"), user("now?")] as any)).toBe(false)
    // ACP + local: local is not a foreign turn → no handoff.
    expect(
      hasUnseenNonAcpAssistantHistory([user("a"), acpReply("acp one"), local("cooldown"), user("b")] as any),
    ).toBe(false)
    // Foreign + local: real foreign turn still triggers handoff (local doesn't suppress it).
    expect(
      hasUnseenNonAcpAssistantHistory([user("a"), foreign("spark reply"), local("cooldown"), user("b")] as any),
    ).toBe(true)
    // Local must not count as an ACP checkpoint hiding an earlier foreign turn.
    expect(
      hasUnseenNonAcpAssistantHistory([user("a"), foreign("spark reply"), local("cooldown"), user("back")] as any),
    ).toBe(true)
  })
})

describe("CodeRabbit final 8 findings", () => {
  test("fileData rejects path-shaped and bad data-URI payloads, accepts /9j/ JPEG", () => {
    const blocks = buildThinAcpPromptBlocks(
      [
        {
          role: "user",
          content: [
            { type: "text", text: "look" },
            {
              type: "file",
              data: "/workspace/assets/screenshots/abcdef",
              mediaType: "image/png",
              filename: "path.png",
            },
            {
              type: "file",
              data: "data:image/png;base64,not base64!",
              mediaType: "image/png",
              filename: "bad.png",
            },
            { type: "file", data: `/9j/${"A".repeat(28)}`, mediaType: "image/png", filename: "photo.jpg" },
          ],
        },
      ] as any,
    )
    const images = blocks.filter((b) => b.type === "image")
    expect(images).toHaveLength(1)
    expect((images[0] as any).data).toBe(`/9j/${"A".repeat(28)}`)
    const text = blocks
      .filter((b) => b.type === "text")
      .map((b) => (b as any).text)
      .join("\n")
    expect(text).toContain("[Attached file: path.png")
    expect(text).toContain("[Attached file: bad.png")
    expect(JSON.stringify(blocks)).not.toContain("/workspace/assets/screenshots/abcdef")
    expect(JSON.stringify(blocks)).not.toContain("not base64!")
  })

  test("buildTextOnlyHandoff skips localStatus assistant lines", () => {
    const blocks = buildTextOnlyHandoffPromptBlocks(
      [
        { role: "user", content: [{ type: "text", text: "first q" }] },
        {
          role: "assistant",
          content: [
            {
              type: "text",
              text: "Antigravity ACP cooldown active (60s remaining)",
              providerMetadata: { "antigravity-acp": { localStatus: true } },
            },
          ],
        } as any,
        { role: "assistant", content: [{ type: "text", text: "real foreign answer" }] } as any,
        { role: "user", content: [{ type: "text", text: "continue in ACP" }] },
      ] as any,
    )
    const head = (blocks[0] as any).text as string
    expect(head).toContain(TEXT_ONLY_HANDOFF_HEADER)
    expect(head).toContain("real foreign answer")
    expect(head).not.toContain("cooldown active")
    expect((blocks[blocks.length - 1] as any).text).toBe("continue in ACP")
  })

  test("cooldown recheck after settlingPrompt blocks tripped pool, other pools proceed", async () => {
    const promptCalls: string[] = []
    const makeProc = () => {
      const fake: any = {
        get isClosed() {
          return false
        },
        initialize: async () => ({}),
        authenticate: async () => ({}),
        newSession: async () => ({ sessionId: "acp-recheck-1" }),
        resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
        setModel: async () => {},
        setMode: async () => {},
        setConfigOption: async () => {},
        closeSession: async () => {},
        notifyCancel: () => {},
        cancel: async () => {},
        prompt: async () => {
          promptCalls.push("prompt")
          return { stopReason: "end_turn" }
        },
        dispose: async () => {},
      }
      return fake
    }
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: () => makeProc() as any,
    })
    const sid = "ses-recheck-1"
    const s = (bridge as any).getOrCreateSession(sid)
    // Install a live proc so ensureAcpSession joins immediately.
    const proc = makeProc()
    s.proc = proc
    s.acpSessionId = "acp-recheck-1"
    ;(bridge as any).activeProcs.add(proc)
    // Controlled settling barrier: the follow-up must wait on it.
    let releaseSettling!: () => void
    const gate = new Promise<void>((r) => {
      releaseSettling = r
    })
    s.settlingPrompt = gate

    const lmGemini = bridge.createLanguageModel("gemini-4-neon")
    const p2 = lmGemini.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "follow-up gemini" }] }],
      headers: { "x-opencode-session-id": sid },
    } as any)
    await new Promise((r) => setTimeout(r, 50))
    // Still waiting on the barrier: no prompt yet.
    expect(promptCalls).toHaveLength(0)
    // The settling prompt trips the gemini pool breaker while we wait.
    ;(bridge as any).tripCircuitBreaker("RESOURCE_EXHAUSTED (code 429)", "gemini")
    releaseSettling()
    const r2 = await p2
    const parts2 = await readAllParts(r2.stream)
    expect(parts2.find((p) => p.type === "finish")?.finishReason?.raw).toBe("circuit_open")
    expect(promptCalls).toHaveLength(0)

    // Unrelated pool proceeds on the same live session.
    const lmClaude = bridge.createLanguageModel("claude-opus-5-5")
    const r3 = await lmClaude.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "hello claude" }] }],
      headers: { "x-opencode-session-id": sid },
    } as any)
    const parts3 = await readAllParts(r3.stream)
    expect(parts3.find((p) => p.type === "finish")?.finishReason?.unified).toBe("stop")
    expect(promptCalls).toHaveLength(1)
    await bridge.disposeAll()
  })

  test("setModel reset revalidation retries setup instead of prompting disposed proc", async () => {
    const promptCalls: string[] = []
    let bridgeRef: any
    const factoryCalls: any[] = []
    const makeProc = (idx: number) => {
      const fake: any = {
        get isClosed() {
          return false
        },
        _idx: idx,
        initialize: async () => ({}),
        authenticate: async () => ({}),
        newSession: async () => ({ sessionId: `acp-reval-${idx}` }),
        resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
        setModel: async () => {
          if (idx === 0) {
            // Simulate a user reset landing mid-setModel: disposes proc0.
            await bridgeRef.resetSession("ses-reval-1")
          }
        },
        setMode: async () => {},
        setConfigOption: async () => {},
        closeSession: async () => {},
        notifyCancel: () => {},
        cancel: async () => {},
        prompt: async () => {
          promptCalls.push(`prompt-${idx}`)
          return { stopReason: "end_turn" }
        },
        dispose: async () => {},
      }
      factoryCalls.push(fake)
      return fake
    }
    let n = 0
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: () => makeProc(n++),
    })
    bridgeRef = bridge
    const lm = bridge.createLanguageModel("gemini-4-neon")
    const res = await lm.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
      headers: { "x-opencode-session-id": "ses-reval-1" },
    } as any)
    const parts = await readAllParts(res.stream)
    expect(parts.find((p) => p.type === "finish")?.finishReason?.unified).toBe("stop")
    // Retried setup: the disposed proc0 was never prompted; proc1 served the turn.
    expect(factoryCalls.length).toBe(2)
    expect(promptCalls).toEqual(["prompt-1"])
    await bridge.disposeAll()
  })

  test("warmup pendingFreshHandoff survives simulated restart before turn 1", async () => {
    const SESSION_ID = "ses-restart-handoff"
    const SESSION_KEY = `antigravity-acp:session:${SESSION_ID}`
    const store = new Map<string, unknown>([[SESSION_KEY, "acp-stale-restart"]])
    const ctx: any = {
      location: { directory: "/tmp/acp-test-dir" },
      storage: {
        get: (k: string) => Effect.succeed(store.get(k)),
        set: (k: string, v: unknown) => Effect.sync(() => void store.set(k, v)),
        remove: (k: string) => Effect.sync(() => void store.delete(k)),
      },
      session: {
        get: () => Effect.succeed({ location: { directory: "/tmp/acp-test-dir" } }),
        update: () => Effect.void,
      },
      permission: { assert: () => Effect.void },
    }
    const failingResume = () => ({
      initialize: async () => ({}),
      authenticate: async () => ({}),
      newSession: async () => ({ sessionId: "acp-fresh-restart" }),
      resumeSession: async () => {
        throw new Error("session gone")
      },
      setModel: async () => {},
      setMode: async () => {},
      setConfigOption: async () => {},
      closeSession: async () => {},
      notifyCancel: () => {},
      cancel: async () => {},
      prompt: async () => ({ stopReason: "end_turn" }),
      dispose: async () => {},
      get isClosed() {
        return false
      },
    })
    const bridge1 = new SessionBridgeManager(ctx, "/tmp/acp-test-dir", undefined, {
      createProc: () => failingResume() as any,
    })
    await bridge1.warmSession(SESSION_ID)
    await waitFor(() => (bridge1 as any).getOrCreateSession(SESSION_ID).proc !== undefined)
    expect((bridge1 as any).getOrCreateSession(SESSION_ID).pendingFreshHandoff).toBe(true)
    // Deferred persist: storage still holds the prior id, not the fresh one.
    expect(store.get(SESSION_KEY)).toBe("acp-stale-restart")
    await bridge1.disposeAll()

    // Simulated restart: fresh manager, same storage. First real turn must
    // still deliver prior history via handoff (resume fails again → fresh).
    let captured: any[] | undefined
    const bridge2 = new SessionBridgeManager(ctx, "/tmp/acp-test-dir", undefined, {
      createProc: () => {
        const base: any = failingResume()
        const origPrompt = base.prompt
        base.prompt = async (_sid: string, prompt: any[]) => {
          captured = prompt
          return origPrompt()
        }
        return base
      },
    })
    const lm = bridge2.createLanguageModel("gemini-4-neon")
    const acpPrompt = [
      { role: "user", content: [{ type: "text", text: "First ACP question" }] },
      {
        role: "assistant",
        content: [{ type: "text", text: "ACP answer one" }],
        providerOptions: { "antigravity-acp": { acp: true } },
      },
      { role: "user", content: [{ type: "text", text: "Back to ACP now" }] },
    ] as any
    const res = await lm.doStream({ prompt: acpPrompt, headers: { "x-opencode-session-id": SESSION_ID } } as any)
    const parts = await readAllParts(res.stream)
    expect(parts.find((p) => p.type === "finish")?.finishReason?.unified).toBe("stop")
    const head = (captured ?? []).find((b) => b.type === "text")?.text ?? ""
    expect(head).toContain(TEXT_ONLY_HANDOFF_HEADER)
    expect(head).toContain("ACP answer one")
    await bridge2.disposeAll()
  })
})

describe("Token/context window usage reporting", () => {
  afterEach(() => {
    resetSharedCatalogForTests()
  })

  function makeUsageBridge(promptImpl: (onUpdate: any) => (sid: string, prompt: any[]) => Promise<any>) {
    let onUpdateRef: any
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: ({ onUpdate }) => {
        onUpdateRef = onUpdate
        return {
          isClosed: false,
          initialize: async () => ({}),
          authenticate: async () => ({}),
          newSession: async () => ({ sessionId: "acp-usage-1" }),
          resumeSession: async (params: any) => ({ sessionId: params.sessionId }),
          setModel: async () => {},
          setMode: async () => {},
          setConfigOption: async () => {},
          closeSession: async () => {},
          notifyCancel: () => {},
          cancel: async () => {},
          prompt: promptImpl(onUpdate),
          dispose: async () => {},
        } as any
      },
    })
    return { bridge, getOnUpdate: () => onUpdateRef }
  }

  test("a) session/prompt usage wins over usage_update (used:0 and used=input+output)", async () => {
    const fullUsage = {
      inputTokens: 1000,
      outputTokens: 200,
      cachedReadTokens: 300,
      cachedWriteTokens: 100,
      thoughtTokens: 50,
      totalTokens: 1200,
    }
    for (const used of [0, 1200]) {
      const { bridge } = makeUsageBridge((onUpdate) => async (sid: string) => {
        onUpdate?.(sid, { sessionUpdate: "usage_update", used })
        return { stopReason: "end_turn", usage: { ...fullUsage } }
      })
      const lm = bridge.createLanguageModel("gemini-4-neon")
      const res = await lm.doStream({
        prompt: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
        headers: { "x-opencode-session-id": `ses-usage-a-${used}` },
      } as any)
      const parts = await readAllParts(res.stream)
      const finish = parts.find((p) => p.type === "finish")
      expect(finish).toBeDefined()
      expect(finish.usage.inputTokens.total).toBe(1000)
      expect(finish.usage.outputTokens.total).toBe(200)
      expect(finish.usage.outputTokens.text).toBe(150)
      expect(finish.usage.outputTokens.reasoning).toBe(50)
      expect(finish.usage.inputTokens.cacheRead).toBe(300)
      expect(finish.usage.inputTokens.cacheWrite).toBe(100)
      expect(finish.usage.inputTokens.noCache).toBe(600)
      await bridge.disposeAll()
    }

    // snake_case variant
    const { bridge } = makeUsageBridge((onUpdate) => async (sid: string) => {
      onUpdate?.(sid, { sessionUpdate: "usage_update", used: 0 })
      return {
        stopReason: "end_turn",
        usage: {
          input_tokens: 1000,
          output_tokens: 200,
          cached_read_tokens: 300,
          cached_write_tokens: 100,
          thought_tokens: 50,
          total_tokens: 1200,
        },
      }
    })
    const lm = bridge.createLanguageModel("gemini-4-neon")
    const res = await lm.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
      headers: { "x-opencode-session-id": "ses-usage-a-snake" },
    } as any)
    const parts = await readAllParts(res.stream)
    const finish = parts.find((p) => p.type === "finish")
    expect(finish.usage.inputTokens.total).toBe(1000)
    expect(finish.usage.outputTokens.text).toBe(150)
    expect(finish.usage.inputTokens.noCache).toBe(600)
    await bridge.disposeAll()
  })

  test("b) usage_update used as fallback when prompt omits usage; used:0 ignored", async () => {
    // Fallback: used > 0
    {
      const { bridge } = makeUsageBridge((onUpdate) => async (sid: string) => {
        onUpdate?.(sid, { sessionUpdate: "usage_update", used: 4567 })
        return { stopReason: "end_turn" }
      })
      const lm = bridge.createLanguageModel("gemini-4-neon")
      const res = await lm.doStream({
        prompt: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
        headers: { "x-opencode-session-id": "ses-usage-b-fallback" },
      } as any)
      const parts = await readAllParts(res.stream)
      const finish = parts.find((p) => p.type === "finish")
      expect(finish.usage.inputTokens.total).toBe(4567)
      await bridge.disposeAll()
    }
    // Ignored: used = 0
    {
      const { bridge } = makeUsageBridge((onUpdate) => async (sid: string) => {
        onUpdate?.(sid, { sessionUpdate: "usage_update", used: 0 })
        return { stopReason: "end_turn" }
      })
      const lm = bridge.createLanguageModel("gemini-4-neon")
      const res = await lm.doStream({
        prompt: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
        headers: { "x-opencode-session-id": "ses-usage-b-zero" },
      } as any)
      const parts = await readAllParts(res.stream)
      const finish = parts.find((p) => p.type === "finish")
      expect(finish.usage.inputTokens.total).toBeUndefined()
      await bridge.disposeAll()
    }
  })

  test("c) lastContextUsed cleared at start of new turn (no bleed into turn 2)", async () => {
    let calls = 0
    const { bridge } = makeUsageBridge((onUpdate) => async (sid: string) => {
      calls++
      if (calls === 1) {
        onUpdate?.(sid, { sessionUpdate: "usage_update", used: 9999 })
        return { stopReason: "end_turn" }
      }
      return { stopReason: "end_turn" }
    })
    const lm = bridge.createLanguageModel("gemini-4-neon")
    const sid = "ses-usage-c"
    const res1 = await lm.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "turn one" }] }],
      headers: { "x-opencode-session-id": sid },
    } as any)
    const parts1 = await readAllParts(res1.stream)
    expect(parts1.find((p) => p.type === "finish")?.usage.inputTokens.total).toBe(9999)

    const res2 = await lm.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "turn two" }] }],
      headers: { "x-opencode-session-id": sid },
    } as any)
    const parts2 = await readAllParts(res2.stream)
    expect(parts2.find((p) => p.type === "finish")?.usage.inputTokens.total).toBeUndefined()
    await bridge.disposeAll()
  })

  test("d) finishTurnCleanStop preserves lastContextUsed after interrupt/error", async () => {
    const { bridge } = makeUsageBridge((onUpdate) => async (sid: string) => {
      onUpdate?.(sid, { sessionUpdate: "usage_update", used: 7777 })
      throw new Error("API error (attempt 1): RESOURCE_EXHAUSTED (code 429): Individual quota reached")
    })
    const lm = bridge.createLanguageModel("gemini-4-neon")
    const res = await lm.doStream({
      prompt: [{ role: "user", content: [{ type: "text", text: "hello" }] }],
      headers: { "x-opencode-session-id": "ses-usage-d" },
    } as any)
    const parts = await readAllParts(res.stream)
    const finish = parts.find((p) => p.type === "finish")
    expect(finish?.finishReason?.raw).toBe("quota_cooldown")
    expect(finish?.usage.inputTokens.total).toBe(7777)
    await bridge.disposeAll()
  })
})

describe("Antigravity questions via OpenCode's question tool", () => {
  afterEach(() => {
    resetSharedCatalogForTests()
  })

  // Captured from `agy acp` 1.5.13 (single + multi-select), plus a free-form
  // question and Go's lexicographic key order.
  const agyForm = {
    mode: "form",
    message: "The agent has 3 questions.",
    requestedSchema: {
      type: "object",
      properties: {
        q1: {
          type: "string",
          title: "What is your favorite color?",
          oneOf: [
            { const: "1", title: "(Recommended) Blue" },
            { const: "2", title: "Red" },
          ],
        },
        q1_other: { type: "string", title: "Other", description: "Optional free-form answer to: What is your favorite color?" },
        q10_other: { type: "string", title: "Anything else?" },
        q2: {
          type: "array",
          title: "Which pets?",
          items: {
            anyOf: [
              { const: "1", title: "Cat" },
              { const: "2", title: "Dog, small" },
            ],
          },
        },
        q2_other: { type: "string", title: "Other", description: "Optional free-form answer to: Which pets?" },
      },
    },
  }

  test("maps agy ask_question forms to OpenCode questions and answers back", () => {
    const bridged = elicitationToQuestions(agyForm)!
    expect(bridged.questions).toEqual([
      {
        header: "Question 1",
        question: "What is your favorite color?",
        options: [
          { label: "(Recommended) Blue", description: "" },
          { label: "Red", description: "" },
        ],
      },
      {
        header: "Question 2",
        question: "Which pets?",
        options: [
          { label: "Cat", description: "" },
          { label: "Dog, small", description: "" },
        ],
        multiple: true,
      },
      { header: "Question 3", question: "Anything else?", options: [] },
    ])
    expect(bridged.respond([["Red"], ["Cat", "Dog, small", "a fish"], ["no"]])).toEqual({
      q1: "2",
      q2: ["1", "2"],
      q2_other: "a fish",
      q10_other: "no",
    })
    // OpenCode semantics: typed text replaces a single choice.
    expect(bridged.respond([["Purple"], [], []])).toEqual({ q1_other: "Purple" })
  })

  test("declines forms it cannot ask", () => {
    expect(elicitationToQuestions({ mode: "url", url: "https://example.com" })).toBeUndefined()
    expect(
      elicitationToQuestions({ requestedSchema: { type: "object", properties: { x: { type: "object" } } } }),
    ).toBeUndefined()
  })

  test("parses OpenCode question tool output with labels that contain commas", () => {
    const questions = elicitationToQuestions(agyForm)!.questions
    const text =
      'User has answered your questions: "What is your favorite color?"="Unanswered", ' +
      '"Which pets?"="Dog, small, a fish, really", "Anything else?"="yes, \\"quoted\\"". ' +
      "You can now continue with the user's answers in mind."
    expect(parseQuestionToolAnswers(text, questions)).toEqual([[], ["Dog, small", "a fish, really"], ['yes, \\"quoted\\"']])
    expect(parseQuestionToolAnswers("something else", questions)).toBeUndefined()
  })

  function makeQuestionBridge(
    withQuestionTool = true,
    beforeAsk?: (onUpdate: (sid: string, update: AcpSessionUpdate) => void, sid: string) => void,
  ) {
    const responses: any[] = []
    let releasePrompt!: () => void
    const promptDone = new Promise<void>((r) => (releasePrompt = r))
    const bridge = new SessionBridgeManager(makeMockCtx(), "/tmp/acp-test-dir", undefined, {
      createProc: ({ onUpdate, onElicitation }) =>
        ({
          isClosed: false,
          initialize: async () => ({}),
          authenticate: async () => ({}),
          newSession: async () => ({ sessionId: "acp-question" }),
          setModel: async () => {},
          setMode: async () => {},
          notifyCancel: () => {},
          cancel: async () => {},
          closeSession: async () => {},
          dispose: async () => {},
          prompt: async (sessionId: string) => {
            if (beforeAsk) beforeAsk(onUpdate!, sessionId)
            else
              onUpdate?.(sessionId, {
                sessionUpdate: "tool_call",
                toolCallId: "ask",
                kind: "other",
                status: "in_progress",
                rawInput: { questions: [{ question: "What is your favorite color?" }] },
              })
            responses.push(await onElicitation!(agyForm))
            onUpdate?.(sessionId, { sessionUpdate: "tool_call_update", toolCallId: "ask", status: "completed" })
            onUpdate?.(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "Thanks" } })
            releasePrompt()
            return { stopReason: "end_turn" }
          },
        }) as any,
    })
    const SESSION = "ses-question"
    bridge.updateContextSnapshot(SESSION, {
      systemText: "",
      agent: "build",
      tools: withQuestionTool ? { question: { description: "Ask", input: {} } } : {},
    })
    const lm = bridge.createLanguageModel("gemini-4-neon")
    const headers = { "x-opencode-session-id": SESSION }
    const user = { role: "user", content: [{ type: "text", text: "Ask me" }] }
    return { bridge, lm, headers, user, responses, promptDone }
  }

  test("asks through OpenCode's question tool and answers the elicitation", async () => {
    const { bridge, lm, headers, user, responses } = makeQuestionBridge()
    const first = await readAllParts((await lm.doStream({ prompt: [user], headers } as any)).stream)

    // agy's own open ask_question call is closed so the question can be dispatched.
    expect(first.find((p) => p.type === "tool-result")).toMatchObject({ toolCallId: "ask", isError: false })
    const call = first.find((p) => p.type === "tool-call" && p.toolName === "question")
    expect(JSON.parse(call.input).questions).toHaveLength(3)
    expect(first.at(-1)).toMatchObject({ type: "finish", finishReason: { unified: "tool-calls" } })

    const output =
      'User has answered your questions: "What is your favorite color?"="(Recommended) Blue", ' +
      '"Which pets?"="Cat", "Anything else?"="Unanswered". You can now continue with the user\'s answers in mind.'
    const second = await readAllParts(
      (
        await lm.doStream({
          prompt: [
            user,
            { role: "assistant", content: [{ type: "tool-call", toolCallId: call.toolCallId, toolName: "question", input: {} }] },
            {
              role: "tool",
              content: [{ type: "tool-result", toolCallId: call.toolCallId, toolName: "question", output: { type: "text", value: output } }],
            },
          ],
          headers,
        } as any)
      ).stream,
    )

    expect(responses).toEqual([{ action: "accept", content: { q1: "1", q2: ["1"] } }])
    // Updates for agy's settled ask_question call are ignored.
    expect(second.filter((p) => p.type === "tool-result")).toHaveLength(0)
    expect(second.find((p) => p.type === "text-delta")?.delta).toBe("Thanks")
    await bridge.disposeAll()
  })

  test("closes only agy's ask_question call and waits for other running tools", async () => {
    const { bridge, lm, headers, user } = makeQuestionBridge(true, (onUpdate, sid) => {
      onUpdate(sid, {
        sessionUpdate: "tool_call",
        toolCallId: "ask",
        kind: "other",
        status: "in_progress",
        rawInput: { questions: [{ question: "What is your favorite color?" }] },
      })
      // Opened after ask_question, so "newest open call" alone would pick it.
      onUpdate(sid, { sessionUpdate: "tool_call", toolCallId: "run", name: "run_command", status: "in_progress" })
      setTimeout(() => {
        onUpdate(sid, { sessionUpdate: "tool_call_update", toolCallId: "run", status: "completed", rawOutput: "done" })
      }, 100)
    })
    const parts = await readAllParts((await lm.doStream({ prompt: [user], headers } as any)).stream)
    const results = parts.filter((p) => p.type === "tool-result")
    expect(results.map((p) => [p.toolCallId, p.isError])).toEqual([
      ["ask", false],
      ["run", false],
    ])
    expect(results[1].result).toBe("done")
    // The question is dispatched only after the other tool finished.
    const questionIndex = parts.findIndex((p) => p.type === "tool-call" && p.toolName === "question")
    expect(questionIndex).toBeGreaterThan(parts.indexOf(results[1]))
    await bridge.handleInterrupt("ses-question")
    await bridge.disposeAll()
  })

  test("cancels when the user dismisses the question", async () => {
    const { bridge, lm, headers, user, responses } = makeQuestionBridge()
    const first = await readAllParts((await lm.doStream({ prompt: [user], headers } as any)).stream)
    const call = first.find((p) => p.type === "tool-call" && p.toolName === "question")
    await readAllParts(
      (
        await lm.doStream({
          prompt: [
            user,
            {
              role: "tool",
              content: [
                { type: "tool-result", toolCallId: call.toolCallId, toolName: "question", output: { type: "error-text", value: "dismissed" } },
              ],
            },
          ],
          headers,
        } as any)
      ).stream,
    )
    expect(responses).toEqual([{ action: "cancel" }])
    await bridge.disposeAll()
  })

  test("cancels without asking when the question tool is unavailable", async () => {
    const { bridge, lm, headers, user, responses, promptDone } = makeQuestionBridge(false)
    const parts = await readAllParts((await lm.doStream({ prompt: [user], headers } as any)).stream)
    await promptDone
    expect(parts.find((p) => p.type === "tool-call" && p.toolName === "question")).toBeUndefined()
    expect(responses).toEqual([{ action: "cancel" }])
    await bridge.disposeAll()
  })

  test("never settles an unrelated native tool when the elicitation arrives first", async () => {
    // Elicitation arrives before agy's ask_question tool_call update, so only
    // an unrelated running tool is visible. The bridge must not fabricate that
    // tool's result; the question is still asked.
    const { bridge, lm, headers, user, responses } = makeQuestionBridge(true, (onUpdate, sid) => {
      onUpdate(sid, { sessionUpdate: "tool_call", toolCallId: "run", name: "run_command", status: "in_progress" })
    })
    const parts = await readAllParts((await lm.doStream({ prompt: [user], headers } as any)).stream)
    expect(parts.find((p) => p.type === "tool-call" && p.toolName === "question")).toBeDefined()
    expect(parts.find((p) => p.type === "tool-result" && p.toolCallId === "run")).toBeUndefined()
    await bridge.disposeAll()
    // Never answered: the elicitation is cancelled, not invented.
    expect(responses).toEqual([{ action: "cancel" }])
  })

  test("parses answers when a question title contains quotes", async () => {
    const { parseQuestionToolAnswers } = await import("../src/question-bridge.js")
    const questions = [
      { header: "Q", question: 'What does "foo" mean?', options: [{ label: "A", description: "" }] },
      { header: "Q", question: "Second?", options: [{ label: "B", description: "" }] },
    ]
    const text =
      'User has answered your questions: "What does "foo" mean?"="A", "Second?"="B". ' +
      "You can now continue with the user's answers in mind."
    expect(parseQuestionToolAnswers(text, questions)).toEqual([["A"], ["B"]])
  })

  test("respond refuses to accept an empty answer set", async () => {
    const { elicitationToQuestions } = await import("../src/question-bridge.js")
    const bridged = elicitationToQuestions(agyForm)!
    expect(bridged.respond([[], [], []])).toBeUndefined()
  })
})
