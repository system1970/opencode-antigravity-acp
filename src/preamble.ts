import { closeSync, constants, fstatSync, openSync, readSync } from "node:fs"
import { homedir } from "node:os"
import { join } from "node:path"
import { stripUnsafeInternalLines, UNSAFE_INTERNAL_SIGNATURE } from "./history-sanitizer.js"

const MAX_FILE_BYTES = 64 * 1024

function readTemplate(file: string): string | undefined {
  try {
    const fd = openSync(file, constants.O_RDONLY | constants.O_NONBLOCK)
    try {
      const stat = fstatSync(fd)
      if (!stat.isFile() || stat.size > MAX_FILE_BYTES) return
      const buffer = Buffer.alloc(MAX_FILE_BYTES + 1)
      let size = 0
      while (size < buffer.length) {
        const count = readSync(fd, buffer, size, buffer.length - size, null)
        if (count === 0) return buffer.toString("utf8", 0, size)
        size += count
      }
    } finally {
      closeSync(fd)
    }
  } catch {
    // Missing or unreadable configuration leaves the prompt unchanged.
  }
}

/** Read the optional owner preamble without forwarding filesystem errors. */
export function readAcpPreamble(sessionID: string, directory: string, homeDir = homedir()): string | undefined {
  if (process.env.ANTIGRAVITY_ACP_PREAMBLE === "off") return

  const file = process.env.ANTIGRAVITY_ACP_PREAMBLE_FILE
    || join(homeDir, ".config", "opencode", "antigravity-acp-preamble.md")
  const template = readTemplate(file)
  if (template === undefined) return

  const rendered = template.replace(/\{\{(SESSION_ID|DIRECTORY)\}\}/g,
    (_placeholder, key) => key === "SESSION_ID" ? sessionID : directory)
  const text = (UNSAFE_INTERNAL_SIGNATURE.test(rendered) ? stripUnsafeInternalLines(rendered) : rendered).slice(0, 4000)
  return text.trim() ? text : undefined
}
