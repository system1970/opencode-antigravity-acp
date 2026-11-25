import { afterEach, beforeEach, expect, test } from "bun:test"
import { ACP_PROVIDER_ID, sanitizeForNonAcpModel } from "../src/history-sanitizer.js"
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

const examples = [
  ["RUN_COMMAND", "other", { CommandLine: "pwd", Summary: "Current directory", Cwd: "/repo" },
    "bash", { command: "pwd", description: "Current directory", workdir: "/repo" }],
  ["run_command", "execute", { CommandLine: "pwd", WaitMsBeforeAsync: 1_000, toolSummary: "Print directory" },
    "bash", { command: "pwd", description: "Print directory" }],
  ["execute", undefined, { cmd: "pwd" }, "bash", { command: "pwd" }],
  ["read_file", "read", { AbsolutePath: "/repo/a", StartLine: 2, EndLine: 8 },
    "read", { filePath: "/repo/a", offset: 2, limit: 6 }],
  ["read_file", "read", { Path: "/repo/a" }, "read", { filePath: "/repo/a" }],
  ["view_file", undefined, { target_file: "/repo/a", start_line: 3 },
    "read", { filePath: "/repo/a", offset: 3 }],
  ["write_to_file", "edit", { TargetFile: "/repo/a", CodeContent: "" },
    "write", { filePath: "/repo/a", content: "" }],
  ["edit_file", "edit", { TargetFile: "/repo/a", TargetContent: "before", ReplacementContent: "" },
    "edit", { filePath: "/repo/a", oldString: "before", newString: "" }],
  ["edit_file", "edit", { Path: "/repo/a", Target: "before", Replacement: "after" },
    "edit", { filePath: "/repo/a", oldString: "before", newString: "after" }],
  ["replace_file_content", "edit", { path: "/repo/a", old_string: "a", new_string: "b" },
    "edit", { filePath: "/repo/a", oldString: "a", newString: "b" }],
  ["multi_replace_file_content", "edit", {
    target_file: "/repo/a", replacement_chunks: [{ target_content: "a", replacement_content: "b" }],
  }, "edit", { filePath: "/repo/a", oldString: "a", newString: "b" }],
  ["grep_search", "search", { Query: "needle", SearchPath: "/repo" },
    "grep", { pattern: "needle", path: "/repo" }],
  ["search", undefined, { query: "needle" }, "grep", { pattern: "needle" }],
  ["find_by_name", "search", { Pattern: "*.ts", SearchDirectory: "/repo" },
    "glob", { pattern: "*.ts", path: "/repo" }],
  ["list_dir", "read", { directoryPath: "/repo" }, "list", { path: "/repo" }],
  ["read_url_content", "fetch", { Url: "https://example.com" },
    "webfetch", { url: "https://example.com" }],
  ["search_web", "search", { url: "https://example.com" },
    "webfetch", { url: "https://example.com" }],
  ["unnamed", "EXECUTE", { command_line: "pwd", title: "Directory", cwd: "/repo" },
    "bash", { command: "pwd", description: "Directory", workdir: "/repo" }],
  ["unnamed", "read", { file_path: "/repo/a", startLine: 1, endLine: 5 },
    "read", { filePath: "/repo/a", offset: 1, limit: 4 }],
  ["unnamed", "edit", { targetFile: "/repo/a", code_content: "text" },
    "write", { filePath: "/repo/a", content: "text" }],
  ["unnamed", "edit", { path: "/repo/a", targetContent: "a", replacementContent: "b" },
    "edit", { filePath: "/repo/a", oldString: "a", newString: "b" }],
  ["unnamed", "search", { pattern: "needle", search_path: "/repo" },
    "grep", { pattern: "needle", path: "/repo" }],
  ["unnamed", "fetch", { URL: "https://example.com" },
    "webfetch", { url: "https://example.com" }],
] as const

for (const [name, kind, raw, toolName, input] of examples) {
  test(`${name} (${kind ?? "no kind"}) maps to ${toolName}`, () => {
    expect(mapNativeTool(name, kind, raw)).toEqual({ toolName, input })
  })
}

test("multi-edit displays only the first chunk when all required fields map", () => {
  const raw = {
    TargetFile: "/repo/a",
    ReplacementChunks: [
      { TargetContent: "a", ReplacementContent: "b" },
      { TargetContent: "c", ReplacementContent: "d" },
    ],
  }
  expect(mapNativeTool("multi_replace_file_content", "edit", raw)).toEqual({
    toolName: "edit", input: { filePath: "/repo/a", oldString: "a", newString: "b" },
  })
})

test("complete required fields omit raw input even with extra or partial optional fields", () => {
  const command = { command: "pwd", title: "Title", toolSummary: "Summary", extra: 1 }
  expect(mapNativeTool("run_command", "execute", command).input).toEqual({ command: "pwd", description: "Title" })
  expect(mapNativeTool("read_file", "read", { path: "/repo/a", EndLine: 8 }).input).toEqual({ filePath: "/repo/a" })
  expect(mapNativeTool("write_to_file", "edit", { Path: "/repo/a", extra: 1 }).input).toEqual({ filePath: "/repo/a" })
})

test("missing required fields fall back to acp_<name> with raw input unchanged", () => {
  const command = { WaitMsBeforeAsync: 1_000, toolSummary: "Print directory" }
  const edit = { Path: "/repo/a", Target: "before" }
  const search = { Query: "example" }
  expect(mapNativeTool("run_command", "execute", command)).toEqual({ toolName: "acp_run_command", input: command })
  expect(mapNativeTool("edit_file", "edit", edit)).toEqual({ toolName: "acp_edit_file", input: edit })
  expect(mapNativeTool("search_web", "fetch", search)).toEqual({ toolName: "acp_search_web", input: search })
})

test("unknown tool preserves original spelling and input identity", () => {
  const raw = ["opaque args"]
  const result = mapNativeTool("Other_Tool", "other", raw)
  expect(result.toolName).toBe("acp_Other_Tool")
  expect(result.input).toBe(raw)
})

test("disabling native tools preserves raw input for name and kind matches", () => {
  process.env.ANTIGRAVITY_ACP_NATIVE_TOOLS = "off"
  const raw = { CommandLine: "pwd" }
  for (const name of ["RUN_COMMAND", "Unknown_Tool"]) {
    const result = mapNativeTool(name, "execute", raw)
    expect(result.toolName).toBe(`acp_${name}`)
    expect(result.input).toBe(raw)
  }
})

test("model switch rewrites ACP calls/results, but not native or relayed OpenCode parts", () => {
  const providerMetadata = { [ACP_PROVIDER_ID]: { acp: true } }
  const openCode = { type: "tool-call", providerExecuted: true, name: "bash", input: {} }
  const relayed = { ...openCode, providerExecuted: false, providerMetadata }
  const messages = [{ role: "assistant", content: [
    openCode,
      { ...openCode, name: "bash", providerMetadata },
      { type: "tool-result", toolName: "bash", providerExecuted: true, providerMetadata, result: "ok" },
      { ...openCode, name: undefined, toolName: "read", providerOptions: providerMetadata },
      { ...openCode, name: "acp_old" },
    relayed,
  ] }]
  sanitizeForNonAcpModel(messages)
  expect(messages[0].content).toEqual([
    openCode,
    { type: "text", text: "[Antigravity tool call `bash`: {}]\n[Antigravity tool `bash` result: ok]\n[Antigravity tool call `read`: {}]\n[Antigravity tool call `acp_old`: {}]" },
    relayed,
  ])
})
