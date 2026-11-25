import { afterEach, beforeEach, expect, test } from "bun:test"
import { mapNativeTool } from "../src/native-tool-map.js"

let previousNativeTools: string | undefined

beforeEach(() => {
  previousNativeTools = process.env.ANTIGRAVITY_ACP_NATIVE_TOOLS
  delete process.env.ANTIGRAVITY_ACP_NATIVE_TOOLS
})

afterEach(() => {
  if (previousNativeTools === undefined) delete process.env.ANTIGRAVITY_ACP_NATIVE_TOOLS
  else process.env.ANTIGRAVITY_ACP_NATIVE_TOOLS = previousNativeTools
})

test("real captured run_command maps to bash", () => {
  const raw = { CommandLine: "echo hello-from-gemini && pwd", WaitMsBeforeAsync: 5000, toolSummary: "Run echo and pwd" }
  expect(mapNativeTool("run_command", "execute", raw)).toEqual({
    toolName: "bash",
    input: { command: "echo hello-from-gemini && pwd", description: "Run echo and pwd" },
  })
})

test("real captured read_file maps to read", () => {
  const raw = { Path: "/tmp/x/hello.txt" }
  expect(mapNativeTool("read_file", "read", raw)).toEqual({
    toolName: "read",
    input: { filePath: "/tmp/x/hello.txt" },
  })
})

test("real captured edit_file maps to edit", () => {
  const raw = { Path: "/tmp/x/hello.txt", Target: "foo", Replacement: "bar" }
  expect(mapNativeTool("edit_file", "edit", raw)).toEqual({
    toolName: "edit",
    input: { filePath: "/tmp/x/hello.txt", oldString: "foo", newString: "bar" },
  })
})

test("edit_file missing required Path falls back to acp_edit_file with raw input unchanged", () => {
  const raw = { Target: "foo", Replacement: "bar" }
  const result = mapNativeTool("edit_file", "edit", raw)
  expect(result.toolName).toBe("acp_edit_file")
  expect(result.input).toBe(raw)
})
