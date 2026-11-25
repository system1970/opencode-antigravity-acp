interface FieldMapping {
  readonly aliases: readonly string[]
  readonly required?: boolean
  readonly fromChunk?: boolean
  readonly number?: true
  readonly differenceFrom?: string
}

interface ToolMapping {
  readonly names: readonly string[]
  readonly kind?: string
  readonly kindRequiresField?: string
  readonly toolName: string
  readonly firstChunk?: readonly string[]
  readonly fields: Readonly<Record<string, FieldMapping>>
}

// Named tools take precedence over kind fallbacks. The first matching kind wins.
const toolMappings: readonly ToolMapping[] = [
  {
    names: ["run_command", "execute"], kind: "execute", toolName: "bash",
    fields: {
      // WaitMsBeforeAsync is an agy execution option, not a card field.
      command: { aliases: ["CommandLine", "command", "cmd", "commandLine"], required: true },
      description: { aliases: ["title", "toolSummary", "summary"] },
      workdir: { aliases: ["Cwd", "cwd"] },
    },
  },
  {
    names: ["read_file", "view_file"], kind: "read", toolName: "read",
    fields: {
      filePath: { aliases: ["AbsolutePath", "TargetFile", "path", "file", "filePath"], required: true },
      offset: { aliases: ["StartLine"], number: true },
      limit: { aliases: ["EndLine"], number: true, differenceFrom: "offset" },
    },
  },
  {
    names: ["write_to_file"], kind: "edit", kindRequiresField: "content", toolName: "write",
    fields: {
      content: { aliases: ["CodeContent", "content"] },
      filePath: { aliases: ["TargetFile", "path"], required: true },
    },
  },
  {
    names: ["edit_file", "replace_file_content", "multi_replace_file_content"], kind: "edit", toolName: "edit",
    firstChunk: ["ReplacementChunks", "chunks"],
    fields: {
      filePath: { aliases: ["TargetFile", "path"], required: true },
      oldString: { aliases: ["TargetContent", "old_string", "Target"], fromChunk: true, required: true },
      newString: { aliases: ["ReplacementContent", "new_string", "Replacement"], fromChunk: true, required: true },
    },
  },
  {
    names: ["grep_search", "search"], kind: "search", toolName: "grep",
    fields: {
      pattern: { aliases: ["Query", "query", "pattern"], required: true },
      path: { aliases: ["SearchPath", "path"] },
    },
  },
  {
    names: ["find_by_name"], toolName: "glob",
    fields: {
      pattern: { aliases: ["Pattern", "pattern"], required: true },
      path: { aliases: ["SearchDirectory", "path"] },
    },
  },
  {
    names: ["list_dir"], toolName: "list",
    fields: { path: { aliases: ["DirectoryPath", "path"], required: true } },
  },
  {
    names: ["read_url_content", "search_web"], kind: "fetch", toolName: "webfetch",
    fields: { url: { aliases: ["Url", "url"], required: true } },
  },
]

const normalizeKey = (key: string) => key.replace(/_/g, "").toLowerCase()

function fields(value: unknown) {
  const entries = value && typeof value === "object" && !Array.isArray(value)
    ? Object.entries(value) : []
  function take<T>(isType: (value: unknown) => value is T, aliases: string[]): T | undefined {
    for (const alias of aliases) {
      const normalized = normalizeKey(alias)
      const match = entries.find(([key, value]) => key === alias && isType(value))
        ?? entries.find(([key, value]) => normalizeKey(key) === normalized && isType(value))
      if (match) return match[1] as T
    }
    return undefined
  }
  return {
    string: (...aliases: string[]) => take((v): v is string => typeof v === "string", aliases),
    number: (...aliases: string[]) => take((v): v is number => typeof v === "number" && Number.isFinite(v), aliases),
    array: (...aliases: string[]) => take(Array.isArray, aliases),
  }
}

/** Translate provider-executed tool args for display, not for local execution. */
export function mapNativeTool(name: string, kind: string | undefined, rawInput: unknown): {
  toolName: string
  input: unknown
} {
  const legacy = { toolName: `acp_${name}`, input: rawInput }
  if (process.env.ANTIGRAVITY_ACP_NATIVE_TOOLS === "off") return legacy

  const normalizedName = name.toLowerCase()
  const normalizedKind = kind?.toLowerCase()
  const args = fields(rawInput)
  const mapping = toolMappings.find((row) => row.names.includes(normalizedName))
    ?? toolMappings.find((row) => normalizedKind !== undefined && row.kind === normalizedKind
      && (row.kindRequiresField === undefined
        || args.string(...row.fields[row.kindRequiresField].aliases) !== undefined))
  if (!mapping) return legacy

  const chunks = mapping.firstChunk && args.array(...mapping.firstChunk)
  const replacement = chunks?.length ? fields(chunks[0]) : args
  const input: Record<string, unknown> = {}
  for (const [key, spec] of Object.entries(mapping.fields)) {
    const source = spec.fromChunk ? replacement : args
    let value = spec.number ? source.number(...spec.aliases) : source.string(...spec.aliases)
    if (spec.differenceFrom !== undefined) {
      const start = input[spec.differenceFrom]
      value = typeof value === "number" && typeof start === "number" && value > start
        ? value - start : undefined
    }
    if (value !== undefined) input[key] = value
    else if (spec.required) return legacy
  }

  return { toolName: mapping.toolName, input }
}
