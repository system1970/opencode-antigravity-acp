import { afterEach, beforeEach, expect, test } from "bun:test"
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import type { LanguageModelV3Prompt } from "@ai-sdk/provider"
import { Effect } from "effect"
import type { AcpContentBlock, AcpProcess, AcpProcessOptions } from "../src/acp-client.js"
import { ACP_PROVIDER_ID } from "../src/history-sanitizer.js"
import { readAcpPreamble } from "../src/preamble.js"
import { SessionBridgeManager } from "../src/session-bridge.js"

let previousSwitch: string | undefined
let previousFile: string | undefined
const sessionID = "oc-session"
const directory = "/workspace/project"
let home: string
let preamblePath: string

beforeEach(() => {
  previousSwitch = process.env.ANTIGRAVITY_ACP_PREAMBLE
  previousFile = process.env.ANTIGRAVITY_ACP_PREAMBLE_FILE
  home = mkdtempSync(join(tmpdir(), "acp-preamble-"))
  const configDir = join(home, ".config", "opencode")
  mkdirSync(configDir, { recursive: true })
  preamblePath = join(configDir, "antigravity-acp-preamble.md")
  delete process.env.ANTIGRAVITY_ACP_PREAMBLE
  delete process.env.ANTIGRAVITY_ACP_PREAMBLE_FILE
})

afterEach(() => {
  if (previousSwitch === undefined) delete process.env.ANTIGRAVITY_ACP_PREAMBLE
  else process.env.ANTIGRAVITY_ACP_PREAMBLE = previousSwitch
  if (previousFile === undefined) delete process.env.ANTIGRAVITY_ACP_PREAMBLE_FILE
  else process.env.ANTIGRAVITY_ACP_PREAMBLE_FILE = previousFile
  rmSync(home, { recursive: true })
})

test("replaces every placeholder literally, preserving Markdown", () => {
  writeFileSync(preamblePath, "# {{SESSION_ID}}\n{{DIRECTORY}}; {{SESSION_ID}}\n")
  expect(readAcpPreamble("id-$&", "/workspace/{{SESSION_ID}}", home)).toBe(
    "# id-$&\n/workspace/{{SESSION_ID}}; id-$&\n",
  )
})

test("caps the rendered preamble at 4000 characters after substitution", () => {
  writeFileSync(preamblePath, `{{DIRECTORY}}${"x".repeat(4000)}`)
  expect(readAcpPreamble(sessionID, directory, home)).toBe(`${directory}${"x".repeat(4000 - directory.length)}`)
})

test("missing, empty and blank files do not produce a preamble", () => {
  expect(readAcpPreamble(sessionID, directory, home)).toBeUndefined()
  writeFileSync(preamblePath, "")
  expect(readAcpPreamble(sessionID, directory, home)).toBeUndefined()
  writeFileSync(preamblePath, " \n\t")
  expect(readAcpPreamble(sessionID, directory, home)).toBeUndefined()
})

test("ignores directories and files larger than 64 KiB", () => {
  process.env.ANTIGRAVITY_ACP_PREAMBLE_FILE = home
  expect(readAcpPreamble(sessionID, directory, home)).toBeUndefined()
  process.env.ANTIGRAVITY_ACP_PREAMBLE_FILE = preamblePath
  writeFileSync(preamblePath, "x".repeat(64 * 1024 + 1))
  expect(readAcpPreamble(sessionID, directory, home)).toBeUndefined()
})

test("ANTIGRAVITY_ACP_PREAMBLE=off disables a populated file", () => {
  writeFileSync(preamblePath, "should not be sent")
  process.env.ANTIGRAVITY_ACP_PREAMBLE = "off"
  expect(readAcpPreamble(sessionID, directory, home)).toBeUndefined()
})

test("the preamble file override takes precedence over the default, but not off", () => {
  writeFileSync(preamblePath, "default")
  const override = join(home, "custom.md")
  process.env.ANTIGRAVITY_ACP_PREAMBLE_FILE = override
  expect(readAcpPreamble(sessionID, directory, home)).toBeUndefined()
  writeFileSync(override, "custom for {{SESSION_ID}}")
  expect(readAcpPreamble(sessionID, directory, home)).toBe(`custom for ${sessionID}`)
  process.env.ANTIGRAVITY_ACP_PREAMBLE = "off"
  expect(readAcpPreamble(sessionID, directory, home)).toBeUndefined()
})

test("strips internal control lines after rendering placeholders", () => {
  writeFileSync(preamblePath, "Keep this\n{{DIRECTORY}}\ntools.opencode.session_move()\nKeep this too")
  expect(readAcpPreamble(sessionID, "/workspace/session_move", home)).toBe("Keep this\nKeep this too")
})

const text = (value: string) => ({ type: "text", text: value } as const)
const user = (value: string) => ({ role: "user", content: [text(value)] } as const)
const assistant = (value: string, acp = false): LanguageModelV3Prompt[number] => ({
  role: "assistant",
  content: [text(value)],
  providerOptions: acp ? { [ACP_PROVIDER_ID]: { acp: true } } : undefined,
})

interface SharedBridgeState {
  storage?: Map<string, string>
  prompts?: { id: string; blocks: readonly AcpContentBlock[] }[]
  newSessions?: string[]
  resumeAttempts?: string[]
}

function mockBridge(onSetModel?: () => void, shared?: SharedBridgeState) {
  const storage = shared?.storage ?? new Map<string, string>()
  const prompts = shared?.prompts ?? []
  const newSessions = shared?.newSessions ?? []
  const resumeAttempts = shared?.resumeAttempts ?? []
  const preambleReads: string[] = []
  const warmInstalled = Promise.withResolvers<void>()
  let exit: () => void = () => {}
  let resumeWorks = true
  const bridge = new SessionBridgeManager({
    session: { get: () => Effect.succeed({ location: { directory } }) },
    storage: {
      get: (key: string) => Effect.sync(() => storage.get(key)),
      set: (key: string, value: string) => Effect.sync(() => storage.set(key, value)),
      remove: (key: string) => Effect.sync(() => storage.delete(key)),
    },
  } as never, "/fallback", (id) => {
    if (bridge.getState(id).connected) warmInstalled.resolve()
  }, {
    readPreamble: (id, cwd) => {
      preambleReads.push(id)
      return readAcpPreamble(id, cwd, home)
    },
    createProc: (options: AcpProcessOptions) => {
      const proc = {
        isClosed: false,
        initialize: async () => {},
        authenticate: async () => {},
        newSession: async ({ cwd }: { cwd: string }) => {
          expect(cwd).toBe(directory)
          const sessionId = `agy-${newSessions.length + 1}`
          newSessions.push(sessionId)
          return { sessionId }
        },
        resumeSession: async ({ sessionId }: { sessionId: string }) => {
          resumeAttempts.push(sessionId)
          if (!resumeWorks) throw new Error("resume unavailable")
          return { sessionId }
        },
        setMode: async () => {},
        setModel: async () => { onSetModel?.() },
        closeSession: async () => {},
        notifyCancel: () => {},
        dispose: async () => { proc.isClosed = true },
        prompt: async (id: string, blocks: readonly AcpContentBlock[]) => {
          prompts.push({ id, blocks })
          return { stopReason: "end_turn" }
        },
      }
      exit = () => {
        proc.isClosed = true
        options.onExit?.(1, null, "")
      }
      return proc as unknown as AcpProcess
    },
  })
  const model = bridge.createLanguageModel(bridge.getModels()[0].id)
  return {
    bridge, prompts, newSessions, resumeAttempts, preambleReads, warmInstalled: warmInstalled.promise,
    disconnect: (canResume: boolean) => { resumeWorks = canResume; exit() },
    send: (prompt: LanguageModelV3Prompt, abortSignal?: AbortSignal) => model.doGenerate({
      headers: { "x-opencode-session-id": sessionID },
      prompt,
      abortSignal,
    }),
  }
}

test("prepends only once per session/new, including warmed sessions and handoffs", async () => {
  writeFileSync(preamblePath, "{{SESSION_ID}} @ {{DIRECTORY}}")
  const { bridge, prompts, newSessions, send, warmInstalled, preambleReads } = mockBridge()
  const opening = text(`${sessionID} @ ${directory}`)
  try {
    await bridge.setCodeMode(sessionID, false)
    await bridge.warmSession(sessionID)
    await warmInstalled
    expect(preambleReads).toEqual([])
    await send([user("first")])
    await send([user("second")])
    await send([user("second"), assistant("Other model reply"), user("switch back")])
    await send([user("follow-up")])
    expect(newSessions).toEqual(["agy-1", "agy-2"])
    expect(prompts.slice(0, 2)).toEqual([
      { id: "agy-1", blocks: [opening, text("first")] },
      { id: "agy-1", blocks: [text("second")] },
    ])
    expect(prompts[2].id).toBe("agy-2")
    expect(prompts[2].blocks).toEqual([
      opening,
      expect.objectContaining({ text: expect.stringContaining("[Historical assistant]: Other model reply") }),
      text("switch back"),
    ])
    expect(prompts[3]).toEqual({ id: "agy-2", blocks: [text("follow-up")] })
  } finally {
    await bridge.disposeAll()
  }
})

test("an abort during setup does not read or consume the pending preamble", async () => {
  writeFileSync(preamblePath, "opening")
  const controller = new AbortController()
  const { bridge, prompts, send, preambleReads } = mockBridge(() => controller.abort())
  try {
    await bridge.setCodeMode(sessionID, false)
    await expect(send([user("cancelled")], controller.signal)).rejects.toMatchObject({ name: "AbortError" })
    expect(prompts).toEqual([])
    expect(preambleReads).toEqual([])
    await send([user("first dispatched")])
    expect(prompts).toEqual([{ id: "agy-1", blocks: [text("opening"), text("first dispatched")] }])
  } finally {
    await bridge.disposeAll()
  }
})

test("an abort while starting a replacement preserves its preamble and history handoff", async () => {
  writeFileSync(preamblePath, "opening")
  const controller = new AbortController()
  let abortOnSetModel = false
  const { bridge, disconnect, prompts, newSessions, send, preambleReads } = mockBridge(() => {
    if (abortOnSetModel) controller.abort()
  })
  const continued = [user("first"), assistant("Earlier ACP reply", true), user("next")]
  try {
    await bridge.setCodeMode(sessionID, false)
    await send([user("first")])
    disconnect(false)
    abortOnSetModel = true
    await expect(send(continued, controller.signal)).rejects.toMatchObject({ name: "AbortError" })
    expect(prompts).toHaveLength(1)
    expect(preambleReads).toHaveLength(1)

    await send(continued)
    expect(newSessions).toEqual(["agy-1", "agy-2"])
    expect(prompts[1]).toEqual({
      id: "agy-2",
      blocks: [
        text("opening"),
        expect.objectContaining({ text: expect.stringContaining("[Historical assistant]: Earlier ACP reply") }),
        text("next"),
      ],
    })
  } finally {
    await bridge.disposeAll()
  }
})

test("a same-process tool-set refresh sends a preamble only on the replacement's first prompt", async () => {
  writeFileSync(preamblePath, "opening")
  const { bridge, prompts, newSessions, send, warmInstalled } = mockBridge()
  try {
    await bridge.warmSession(sessionID)
    await warmInstalled
    bridge.updateContextSnapshot(sessionID, {
      systemText: "outer instructions",
      agent: "default",
      tools: { sample: { description: "Sample tool", input: { type: "object" } } },
    })
    await send([user("first")])
    await send([user("second")])
    expect(newSessions).toEqual(["agy-1", "agy-2"])
    expect(prompts).toEqual([
      { id: "agy-2", blocks: [text("opening"), text("first")] },
      { id: "agy-2", blocks: [text("second")] },
    ])
  } finally {
    await bridge.disposeAll()
  }
})

test("without a file the upstream prompt stays unchanged, even if a file appears later", async () => {
  const { bridge, prompts, send } = mockBridge()
  try {
    await bridge.setCodeMode(sessionID, false)
    await send([{ role: "system", content: "outer instructions" }, user("first")])
    writeFileSync(preamblePath, "too late for this session")
    await send([user("second")])
    expect(prompts).toEqual([
      { id: "agy-1", blocks: [text("first")] },
      { id: "agy-1", blocks: [text("second")] },
    ])
  } finally {
    await bridge.disposeAll()
  }
})

test("successful resume does not resend; failed resume prepends to the replacement session", async () => {
  writeFileSync(preamblePath, "preamble")
  const { bridge, prompts, newSessions, resumeAttempts, send, disconnect } = mockBridge()
  try {
    await bridge.setCodeMode(sessionID, false)
    await send([user("first")])
    disconnect(true)
    await send([user("resumed")])
    disconnect(false)
    await send([user("resumed"), assistant("Previous ACP reply", true), user("reconnected")])
    await send([user("next")])
    expect(resumeAttempts).toEqual(["agy-1", "agy-1"])
    expect(newSessions).toEqual(["agy-1", "agy-2"])
    expect(prompts.map(({ id, blocks }) => [id, blocks[0]])).toEqual([
      ["agy-1", text("preamble")],
      ["agy-1", text("resumed")],
      ["agy-2", text("preamble")],
      ["agy-2", text("next")],
    ])
    expect(prompts[2].blocks[1]).toEqual(
      expect.objectContaining({ text: expect.stringContaining("[Historical assistant]: Previous ACP reply") }),
    )
  } finally {
    await bridge.disposeAll()
  }
})

test("delivered then reload does not resend the preamble", async () => {
  writeFileSync(preamblePath, "preamble")
  const shared: Required<SharedBridgeState> = {
    storage: new Map<string, string>(),
    prompts: [],
    newSessions: [],
    resumeAttempts: [],
  }
  const first = mockBridge(undefined, shared)
  try {
    await first.bridge.setCodeMode(sessionID, false)
    await first.send([user("first")])
  } finally {
    await first.bridge.disposeAll()
  }

  const reloaded = mockBridge(undefined, shared)
  try {
    await reloaded.send([user("after reload")])
  } finally {
    await reloaded.bridge.disposeAll()
  }

  expect(shared.newSessions).toEqual(["agy-1"])
  expect(shared.resumeAttempts).toEqual(["agy-1"])
  expect(shared.prompts).toEqual([
    { id: "agy-1", blocks: [text("preamble"), text("first")] },
    { id: "agy-1", blocks: [text("after reload")] },
  ])
})

test("warm start, reload, then first prompt sends the preamble once", async () => {
  writeFileSync(preamblePath, "preamble")
  const shared: Required<SharedBridgeState> = {
    storage: new Map<string, string>(),
    prompts: [],
    newSessions: [],
    resumeAttempts: [],
  }
  const first = mockBridge(undefined, shared)
  try {
    await first.bridge.setCodeMode(sessionID, false)
    await first.bridge.warmSession(sessionID)
    await first.warmInstalled
  } finally {
    await first.bridge.disposeAll()
  }

  const reloaded = mockBridge(undefined, shared)
  try {
    await reloaded.send([user("first after reload")])
    await reloaded.send([user("second after reload")])
  } finally {
    await reloaded.bridge.disposeAll()
  }

  expect(shared.newSessions).toEqual(["agy-1"])
  expect(shared.resumeAttempts).toEqual(["agy-1"])
  expect(shared.prompts).toEqual([
    { id: "agy-1", blocks: [text("preamble"), text("first after reload")] },
    { id: "agy-1", blocks: [text("second after reload")] },
  ])
})

test("new acpSessionId after model switch sends the preamble once again across reload", async () => {
  writeFileSync(preamblePath, "preamble")
  const shared: Required<SharedBridgeState> = {
    storage: new Map<string, string>(),
    prompts: [],
    newSessions: [],
    resumeAttempts: [],
  }
  const first = mockBridge(undefined, shared)
  try {
    await first.bridge.setCodeMode(sessionID, false)
    await first.send([user("first")])
  } finally {
    await first.bridge.disposeAll()
  }

  const afterSwitch = mockBridge(undefined, shared)
  try {
    await afterSwitch.send([user("first"), assistant("Other model reply"), user("switched back")])
    await afterSwitch.send([user("follow-up")])
  } finally {
    await afterSwitch.bridge.disposeAll()
  }

  const reloadedAgain = mockBridge(undefined, shared)
  try {
    await reloadedAgain.send([user("after second reload")])
  } finally {
    await reloadedAgain.bridge.disposeAll()
  }

  expect(shared.newSessions).toEqual(["agy-1", "agy-2"])
  expect(shared.resumeAttempts).toEqual(["agy-2"])
  expect(shared.prompts[0]).toEqual({ id: "agy-1", blocks: [text("preamble"), text("first")] })
  expect(shared.prompts[1].id).toBe("agy-2")
  expect(shared.prompts[1].blocks).toEqual([
    text("preamble"),
    expect.objectContaining({ text: expect.stringContaining("[Historical assistant]: Other model reply") }),
    text("switched back"),
  ])
  expect(shared.prompts[2]).toEqual({ id: "agy-2", blocks: [text("follow-up")] })
  expect(shared.prompts[3]).toEqual({ id: "agy-2", blocks: [text("after second reload")] })
})

test("warm start and reload remain unaffected when preamble is off or file is missing", async () => {
  const runCase = async () => {
    const shared: Required<SharedBridgeState> = {
      storage: new Map<string, string>(),
      prompts: [],
      newSessions: [],
      resumeAttempts: [],
    }
    const first = mockBridge(undefined, shared)
    try {
      await first.bridge.setCodeMode(sessionID, false)
      await first.bridge.warmSession(sessionID)
      await first.warmInstalled
    } finally {
      await first.bridge.disposeAll()
    }
    const reloaded = mockBridge(undefined, shared)
    try {
      await reloaded.send([user("hello")])
    } finally {
      await reloaded.bridge.disposeAll()
    }
    expect(shared.resumeAttempts).toEqual(["agy-1"])
    return shared.prompts
  }

  // Missing file
  expect(await runCase()).toEqual([{ id: "agy-1", blocks: [text("hello")] }])

  // Populated file with ANTIGRAVITY_ACP_PREAMBLE=off
  writeFileSync(preamblePath, "preamble")
  process.env.ANTIGRAVITY_ACP_PREAMBLE = "off"
  expect(await runCase()).toEqual([{ id: "agy-1", blocks: [text("hello")] }])
})
