import { afterEach, beforeEach, expect, spyOn, test } from "bun:test"
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs"
import { homedir, tmpdir } from "node:os"
import { join } from "node:path"
import { logNativeTool, redactDebugValue } from "../src/debug-tools-log.js"

let prevDebug: string | undefined
let prevDebugFile: string | undefined
let tempDir: string | undefined

beforeEach(() => {
  prevDebug = process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS
  prevDebugFile = process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS_FILE
  delete process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS
  delete process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS_FILE
})

afterEach(() => {
  if (prevDebug === undefined) delete process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS
  else process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS = prevDebug
  if (prevDebugFile === undefined) delete process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS_FILE
  else process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS_FILE = prevDebugFile
  if (tempDir) {
    rmSync(tempDir, { recursive: true, force: true })
    tempDir = undefined
  }
})

test("redactDebugValue replaces home directory prefix with ~", () => {
  const home = homedir()
  expect(redactDebugValue(`${home}/Projects/repo/file.ts`)).toBe("~/Projects/repo/file.ts")
})

test("redactDebugValue replaces email addresses with <email>", () => {
  expect(redactDebugValue("Author: alice.smith+dev@example.com")).toBe("Author: <email>")
})

test("redactDebugValue redacts Bearer, sk-, gh*_, AIza, ya29., and bare 32+ char tokens", () => {
  expect(redactDebugValue("Authorization: Bearer abc.def.ghi")).toBe("Authorization: <redacted>")
  expect(redactDebugValue("bearer shorttoken")).toBe("<redacted>")
  expect(redactDebugValue("key=sk-1234567890abcdef")).toBe("key=<redacted>")
  expect(redactDebugValue("token ghp_12345678901234567890")).toBe("token <redacted>")
  expect(redactDebugValue("token ghs_12345678901234567890")).toBe("token <redacted>")
  expect(redactDebugValue("api=AIzaSyA12345678901234567890")).toBe("api=<redacted>")
  expect(redactDebugValue("oauth=ya29.a0AfH6SMC_token_value")).toBe("oauth=<redacted>")
  expect(redactDebugValue("bare=abcdefghijklmnopqrstuvwxyz012345")).toBe("bare=<redacted>")
  expect(redactDebugValue("short=abcdefghijklmnopqrstuvwxyz01234")).toBe("short=abcdefghijklmnopqrstuvwxyz01234")
})

test("redactDebugValue handles nested objects, arrays, and primitives", () => {
  const home = homedir()
  const input = {
    path: `${home}/.ssh/config`,
    list: ["user@example.org", { auth: "Bearer secret_123", count: 42, ok: true, empty: null }],
  }
  expect(redactDebugValue(input)).toEqual({
    path: "~/.ssh/config",
    list: ["<email>", { auth: "<redacted>", count: 42, ok: true, empty: null }],
  })
})

test("logNativeTool does nothing when ANTIGRAVITY_ACP_DEBUG_TOOLS is unset", () => {
  tempDir = mkdtempSync(join(tmpdir(), "acp-debug-unset-"))
  const logFile = join(tempDir, "toolcalls.jsonl")
  process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS_FILE = logFile
  const warnSpy = spyOn(console, "warn").mockImplementation(() => {})
  try {
    logNativeTool("unset_test_tool", "read", { Path: "/tmp/a" }, "read")
    expect(existsSync(logFile)).toBe(false)
    expect(warnSpy).not.toHaveBeenCalled()
  } finally {
    warnSpy.mockRestore()
  }
})

test("logNativeTool redacts before 300-char truncation and warns once per process", () => {
  tempDir = mkdtempSync(join(tmpdir(), "acp-debug-on-"))
  const logFile = join(tempDir, "toolcalls.jsonl")
  process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS = "1"
  process.env.ANTIGRAVITY_ACP_DEBUG_TOOLS_FILE = logFile

  const warnSpy = spyOn(console, "warn").mockImplementation(() => {})
  try {
    const prefix = "a ".repeat(148)
    const longWithTokenAtTail = `${prefix}sk-1234567890abcdef`
    logNativeTool("trunc_order_tool", "execute", { cmd: longWithTokenAtTail }, "bash")
    logNativeTool("second_tool", "read", { Path: `${homedir()}/notes.md` }, "read")

    expect(warnSpy).toHaveBeenCalledTimes(1)
    expect(warnSpy.mock.calls[0]?.[0]).toBe(
      `antigravity-acp: tool debug log enabled -> ${logFile}; it may contain command and file snippets`,
    )

    const stat = statSync(logFile)
    expect(stat.mode & 0o077).toBe(0)

    const lines = readFileSync(logFile, "utf8").trim().split("\n").map((line) => JSON.parse(line))
    expect(lines).toHaveLength(2)
    expect(lines[0].rawInput.cmd).toBe(`${prefix}<red`)
    expect(lines[1].rawInput.Path).toBe("~/notes.md")
  } finally {
    warnSpy.mockRestore()
  }
})
