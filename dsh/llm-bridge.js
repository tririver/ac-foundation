import { randomBytes, randomUUID } from 'node:crypto'
import {
  chmodSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readFileSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs'
import { dirname, join } from 'node:path'
import { homedir } from 'node:os'
import { createServer } from 'node:net'

export const REQUEST_SCHEMA_VERSION = 'ac.dsh-llm.request.v1'
export const EVENT_SCHEMA_VERSION = 'ac.dsh-llm.event.v1'

const MAX_REQUEST_BYTES = 16 * 1024 * 1024

function defaultDshHome() {
  return process.env.DSH_HOME || join(homedir(), '.dsh')
}

export function resolveBridgePaths({ socketPath, tokenPath } = {}) {
  const resolvedSocketPath =
    socketPath ||
    process.env.DSH_AC_LLM_SOCKET ||
    join(
      defaultDshHome(),
      'runtime',
      `ac-llm-${process.pid}-${randomBytes(6).toString('hex')}.sock`,
    )
  return {
    socketPath: resolvedSocketPath,
    tokenPath:
      tokenPath ||
      process.env.DSH_AC_LLM_TOKEN_FILE ||
      `${resolvedSocketPath}.token`,
  }
}

function assertPathAvailable(path) {
  if (!existsSync(path)) return
  throw new Error(`Refusing to replace existing bridge path: ${path}`)
}

function ensureDirectory(path) {
  if (!existsSync(path)) {
    mkdirSync(path, { recursive: true, mode: 0o700 })
    return
  }
  const status = lstatSync(path)
  if (!status.isDirectory()) {
    throw new Error(`Bridge parent path is not a directory: ${path}`)
  }
}

function send(socket, event) {
  if (socket.destroyed || !socket.writable) return false
  socket.write(`${JSON.stringify(event)}\n`)
  return true
}

function normalizeUsage(usage) {
  if (!usage || typeof usage !== 'object') return null
  const value = {}
  if (Number.isInteger(usage.inputTokens) && usage.inputTokens >= 0) {
    value.input_tokens = usage.inputTokens
  }
  if (Number.isInteger(usage.outputTokens) && usage.outputTokens >= 0) {
    value.output_tokens = usage.outputTokens
  }
  if (Number.isInteger(usage.cacheReadTokens) && usage.cacheReadTokens >= 0) {
    value.cached_input_tokens = usage.cacheReadTokens
  }
  return Object.keys(value).length === 0 ? null : value
}

function normalizeChunk(chunk) {
  if (!chunk || typeof chunk !== 'object') return null
  if (chunk.type === 'text-delta' && typeof chunk.text === 'string') {
    return { type: 'text-delta', text: chunk.text }
  }
  if (chunk.type === 'reasoning-delta' && typeof chunk.text === 'string') {
    return { type: 'reasoning-delta', text: chunk.text }
  }
  if (chunk.type === 'usage') {
    const usage = normalizeUsage(chunk.usage)
    return usage === null ? null : { type: 'usage', usage }
  }
  if (chunk.type === 'finish') {
    const reason = chunk.reason?.kind
    if (typeof reason !== 'string') return null
    const event = { type: 'finish', reason }
    if (chunk.reason?.failure && typeof chunk.reason.failure === 'object') {
      event.failure = {
        code: String(chunk.reason.failure.code || 'provider_error'),
        message: String(chunk.reason.failure.message || 'Provider failed.'),
      }
    }
    return event
  }
  return null
}

function validateRequest(value, token) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Bridge request must be a JSON object.')
  }
  if (value.schema_version !== REQUEST_SCHEMA_VERSION) {
    throw new Error(`Unsupported bridge schema: ${String(value.schema_version)}`)
  }
  if (value.token !== token) throw new Error('Bridge authentication failed.')
  if (value.op !== 'health' && value.op !== 'generate') {
    throw new Error('Bridge request op must be health or generate.')
  }
  if (value.op === 'generate') {
    if (typeof value.prompt !== 'string' || value.prompt.length === 0) {
      throw new Error('Bridge generate.prompt must be a non-empty string.')
    }
    if (typeof value.model !== 'string' || value.model.length === 0) {
      throw new Error('Bridge generate.model must be a non-empty string.')
    }
  }
  return value
}

function messageForPrompt(prompt) {
  return {
    id: randomUUID(),
    role: 'user',
    content: [{ type: 'text', text: prompt }],
    source: { kind: 'user' },
  }
}

async function handleRequest(socket, value, token, llm, controllers) {
  let abort
  let controller
  try {
    const request = validateRequest(value, token)
    if (request.op === 'health') {
      send(socket, {
        schema_version: EVENT_SCHEMA_VERSION,
        type: 'health',
        ok: true,
      })
      return
    }

    controller = new AbortController()
    controllers.add(controller)
    abort = () => controller.abort()
    socket.once('close', abort)
    const provider =
      typeof request.provider === 'string' && request.provider.length > 0
        ? request.provider
        : 'deepseek-official'
    const callConfig = { provider, model: request.model }
    if (typeof request.temperature === 'number') {
      callConfig.temperature = request.temperature
    }
    if (Number.isInteger(request.max_tokens) && request.max_tokens > 0) {
      callConfig.maxTokens = request.max_tokens
    }
    if (typeof request.reasoning_effort === 'string') {
      callConfig.reasoningEffort = request.reasoning_effort
    }
    const prepared = await llm.prepareCall(callConfig, controller.signal)
    send(socket, {
      schema_version: EVENT_SCHEMA_VERSION,
      type: 'started',
      provider: prepared.config.provider,
      model: prepared.config.model,
    })
    const options = {
      ...prepared.config,
      messages: [messageForPrompt(request.prompt)],
      signal: controller.signal,
    }
    if (typeof request.system === 'string' && request.system.length > 0) {
      options.system = request.system
    }
    let finished = false
    for await (const chunk of prepared.stream(options)) {
      const event = normalizeChunk(chunk)
      if (event === null) continue
      if (!send(socket, { schema_version: EVENT_SCHEMA_VERSION, ...event })) {
        return
      }
      if (event.type === 'finish') finished = true
    }
    if (!finished && !socket.destroyed) {
      send(socket, {
        schema_version: EVENT_SCHEMA_VERSION,
        type: 'finish',
        reason: 'error',
        failure: {
          code: 'incomplete_stream',
          message: 'DSH ended the native stream without a finish event.',
        },
      })
    }
  } catch (error) {
    if (!socket.destroyed) {
      const message = error instanceof Error ? error.message : String(error)
      const code =
        error && typeof error === 'object' && typeof error.code === 'string'
          ? error.code
          : 'bridge_error'
      send(socket, {
        schema_version: EVENT_SCHEMA_VERSION,
        type: 'finish',
        reason: 'error',
        failure: { code, message },
      })
    }
  } finally {
    if (controller !== undefined) controllers.delete(controller)
    if (abort !== undefined) socket.removeListener('close', abort)
    if (!socket.destroyed) socket.end()
  }
}

export function startAcLlmBridge({ llm, socketPath, tokenPath } = {}) {
  if (!llm || typeof llm.prepareCall !== 'function') {
    throw new Error('AC DSH bridge requires the native DSH llm service.')
  }

  const paths = resolveBridgePaths({ socketPath, tokenPath })
  ensureDirectory(dirname(paths.socketPath))
  assertPathAvailable(paths.socketPath)
  assertPathAvailable(paths.tokenPath)
  const token = randomBytes(32).toString('hex')
  ensureDirectory(dirname(paths.tokenPath))
  writeFileSync(paths.tokenPath, `${token}\n`, { mode: 0o600 })
  chmodSync(paths.tokenPath, 0o600)

  const sockets = new Set()
  const controllers = new Set()
  const server = createServer((socket) => {
    sockets.add(socket)
    socket.once('close', () => sockets.delete(socket))
    socket.setEncoding('utf8')
    socket.on('error', () => {})
    let buffer = ''
    let handled = false
    socket.on('data', (chunk) => {
      if (handled) return
      buffer += chunk
      if (Buffer.byteLength(buffer, 'utf8') > MAX_REQUEST_BYTES) {
        send(socket, {
          schema_version: EVENT_SCHEMA_VERSION,
          type: 'finish',
          reason: 'error',
          failure: { code: 'request_too_large', message: 'Bridge request is too large.' },
        })
        socket.end()
        handled = true
        return
      }
      const newline = buffer.indexOf('\n')
      if (newline < 0) return
      handled = true
      const line = buffer.slice(0, newline)
      let value
      try {
        value = JSON.parse(line)
      } catch {
        send(socket, {
          schema_version: EVENT_SCHEMA_VERSION,
          type: 'finish',
          reason: 'error',
          failure: { code: 'invalid_json', message: 'Bridge request is not valid JSON.' },
        })
        socket.end()
        return
      }
      void handleRequest(socket, value, token, llm, controllers)
    })
  })
  let readyResolve
  let readyReject
  const ready = new Promise((resolve, reject) => {
    readyResolve = resolve
    readyReject = reject
  })
  server.on('error', (error) => {
    console.error(`[ac-dsh] LLM bridge error: ${error.message}`)
    readyReject(error)
  })
  let listening = false
  server.once('listening', () => {
    try {
      listening = true
      chmodSync(paths.socketPath, 0o600)
      readyResolve()
    } catch (error) {
      readyReject(error)
    }
  })
  server.listen(paths.socketPath)

  let closed = false
  return {
    ...paths,
    ready,
    close() {
      if (closed) return Promise.resolve()
      closed = true
      return new Promise((resolve) => {
        const cleanup = () => {
          if (existsSync(paths.socketPath) && lstatSync(paths.socketPath).isSocket()) {
            unlinkSync(paths.socketPath)
          }
          if (existsSync(paths.tokenPath)) unlinkSync(paths.tokenPath)
          resolve()
        }
        if (listening && server.listening) {
          server.close(cleanup)
          for (const controller of controllers) controller.abort()
          for (const socket of sockets) socket.destroy()
          return
        }
        for (const controller of controllers) controller.abort()
        for (const socket of sockets) socket.destroy()
        cleanup()
      })
    },
    readToken() {
      return readFileSync(paths.tokenPath, 'utf8').trim()
    },
  }
}
