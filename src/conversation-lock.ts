import { spawn } from "node:child_process"
import crypto from "node:crypto"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import { buildAgySpawnEnv } from "./acp-client.js"

export type LockResult =
  | { readonly kind: "held"; readonly fd: number } // the caller owns fd and must close it (normally through LockHold)
  | { readonly kind: "busy" } // another open file description holds this key
  | { readonly kind: "unavailable" } // Linux, but no usable helper, profile or lock file: fail closed
  | { readonly kind: "unguarded" } // not Linux

const LOCK_DIR = path.join(".local", "state", "opencode-antigravity-acp", "locks")
const CONFLICT = 75
const HELPER_TIMEOUT_MS = 3_000
const FLOCK_PATHS = ["/usr/bin/flock", "/bin/flock"]

function findFlock(): string | undefined {
  const onPath = (process.env.PATH ?? "").split(path.delimiter).filter((d) => path.isAbsolute(d)).map((d) => path.join(d, "flock"))
  return [...FLOCK_PATHS, ...onPath].find((p) => { try { fs.accessSync(p, fs.constants.X_OK); return fs.statSync(p).isFile() } catch { return false } })
}

function lockFile(domain: "session" | "native", id: string): string {
  const home = buildAgySpawnEnv().HOME || os.homedir() // the profile is exactly what the child will be given
  if (!path.isAbsolute(home)) throw new Error("profile is not an absolute path")
  const dir = path.join(fs.realpathSync(home), LOCK_DIR) // symlinked profiles collapse onto one directory
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 })
  const st = fs.lstatSync(dir)
  if (!st.isDirectory() || (st.mode & 0o022) !== 0) throw new Error("lock directory is not private")
  return path.join(dir, crypto.createHash("sha256").update(`${domain}\0${id}`).digest("hex"))
}

export function closeLockFd(fd: number): void {
  try {
    fs.closeSync(fd)
  } catch {}
}

export class LockHold {
  private refs = 0
  private open = true
  private readonly held: number[]
  constructor(fds: readonly number[]) {
    this.held = [...fds]
  }
  get fds(): readonly number[] {
    return [...this.held]
  }
  add(fd: number): void {
    this.held.push(fd)
  }
  // One claim, or undefined when the last was already dropped (a closed hold is never revived). The returned function drops it, once.
  ref(): (() => void) | undefined {
    if (!this.open) return undefined
    this.refs++
    let dropped = false
    return () => {
      if (dropped) return
      dropped = true
      if (--this.refs > 0) return
      this.open = false
      for (const fd of this.held.splice(0)) closeLockFd(fd)
    }
  }
}

export function acquireConversationLock(domain: "session" | "native", id: string): Promise<LockResult> {
  if (process.platform !== "linux") return Promise.resolve({ kind: "unguarded" })
  let flock: string | undefined
  let fd: number
  try {
    flock = findFlock()
    if (!flock) return Promise.resolve({ kind: "unavailable" })
    fd = fs.openSync(lockFile(domain, id), fs.constants.O_RDWR | fs.constants.O_CREAT | fs.constants.O_NOFOLLOW, 0o600)
  } catch {
    return Promise.resolve({ kind: "unavailable" })
  }
  return new Promise<LockResult>((resolve) => {
    let child: ReturnType<typeof spawn> | undefined
    let timer: ReturnType<typeof setTimeout> | undefined
    let settled = false
    const done = (r: LockResult) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      if (r.kind !== "held") closeLockFd(fd)
      resolve(r)
    }
    timer = setTimeout(() => {
      try {
        child?.kill("SIGKILL")
      } catch {}
      done({ kind: "unavailable" })
    }, HELPER_TIMEOUT_MS)
    try {
      child = spawn(flock!, ["-n", "-E", String(CONFLICT), "3"], { stdio: ["ignore", "ignore", "ignore", fd], env: {} })
    } catch {
      return done({ kind: "unavailable" })
    }
    child.once("error", () => done({ kind: "unavailable" }))
    child.once("exit", (code) => done(code === 0 ? { kind: "held", fd } : code === CONFLICT ? { kind: "busy" } : { kind: "unavailable" }))
  })
}
