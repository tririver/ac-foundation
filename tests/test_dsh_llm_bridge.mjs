import { chmodSync, existsSync, mkdtempSync, rmSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { once } from 'node:events'
import { createConnection } from 'node:net'
import assert from 'node:assert/strict'
import {
  resolveBridgePaths,
  startAcLlmBridge,
  REQUEST_SCHEMA_VERSION,
} from '../dsh/llm-bridge.js'

const root = mkdtempSync(join(tmpdir(), 'ac-dsh-bridge-'))
const socketPath = join(root, 'ac-llm.sock')
const tokenPath = join(root, 'ac-llm.token')
chmodSync(root, 0o755)
let prepareConfig
const llm = {
  async prepareCall(config, signal) {
    prepareConfig = config
    assert.ok(signal instanceof AbortSignal)
    const resolvedConfig = {
      ...config,
      maxTokens: 4096,
      reasoningEffort: 'high',
    }
    return {
      config: resolvedConfig,
      stream(options) {
        assert.equal(options.provider, 'fake-provider')
        assert.equal(options.model, 'fake-model')
        assert.equal(options.maxTokens, 4096)
        assert.equal(options.reasoningEffort, 'high')
        assert.equal(options.messages[0].content[0].text, 'hello')
        return (async function* () {
          yield { type: 'text-delta', index: 0, text: 'hello ' }
          yield { type: 'reasoning-delta', index: 1, text: 'internal' }
          yield {
            type: 'usage',
            usage: { inputTokens: 3, outputTokens: 2 },
          }
          yield { type: 'finish', reason: { kind: 'stop' } }
        })()
      },
    }
  },
}

const bridge = startAcLlmBridge({ llm, socketPath, tokenPath })
try {
  await bridge.ready
  assert.equal(statSync(root).mode & 0o777, 0o755)
  assert.equal(statSync(socketPath).mode & 0o777, 0o600)
  assert.equal(statSync(tokenPath).mode & 0o777, 0o600)
  assert.notEqual(resolveBridgePaths().socketPath, resolveBridgePaths().socketPath)
  assert.throws(
    () => startAcLlmBridge({ llm, socketPath, tokenPath }),
    /Refusing to replace existing bridge path/,
  )
  const health = await request(socketPath, {
    schema_version: REQUEST_SCHEMA_VERSION,
    token: bridge.readToken(),
    op: 'health',
  })
  assert.equal(health[0].type, 'health')
  assert.equal(health[0].ok, true)

  const events = await request(socketPath, {
    schema_version: REQUEST_SCHEMA_VERSION,
    token: bridge.readToken(),
    op: 'generate',
    provider: 'fake-provider',
    model: 'fake-model',
    prompt: 'hello',
  })
  assert.deepEqual(prepareConfig, { provider: 'fake-provider', model: 'fake-model' })
  assert.deepEqual(
    events.map((event) => event.type),
    ['started', 'text-delta', 'reasoning-delta', 'usage', 'finish'],
  )
  assert.equal(events[1].text, 'hello ')
  assert.deepEqual(events[3].usage, { input_tokens: 3, output_tokens: 2 })
  assert.equal(events[4].reason, 'stop')

  const rejected = await request(socketPath, {
    schema_version: REQUEST_SCHEMA_VERSION,
    token: 'wrong-token',
    op: 'health',
  })
  assert.equal(rejected[0].type, 'finish')
  assert.equal(rejected[0].reason, 'error')
} finally {
  await bridge.close()
  assert.equal(existsSync(socketPath), false)
  assert.equal(existsSync(tokenPath), false)
  rmSync(root, { recursive: true, force: true })
}

const abortRoot = mkdtempSync(join(tmpdir(), 'ac-dsh-bridge-abort-'))
const abortSocketPath = join(abortRoot, 'ac-llm.sock')
const abortTokenPath = join(abortRoot, 'ac-llm.token')
let aborted = false
const blockingLlm = {
  async prepareCall(config) {
    return {
      config,
      stream(options) {
        return (async function* () {
          await new Promise((resolve) => {
            if (options.signal.aborted) {
              aborted = true
              resolve()
              return
            }
            options.signal.addEventListener(
              'abort',
              () => {
                aborted = true
                resolve()
              },
              { once: true },
            )
          })
          yield { type: 'finish', reason: { kind: 'aborted' } }
        })()
      },
    }
  },
}
const blockingBridge = startAcLlmBridge({
  llm: blockingLlm,
  socketPath: abortSocketPath,
  tokenPath: abortTokenPath,
})
try {
  await blockingBridge.ready
  const socket = createConnection(abortSocketPath)
  socket.setEncoding('utf8')
  await once(socket, 'connect')
  socket.write(
    `${JSON.stringify({
      schema_version: REQUEST_SCHEMA_VERSION,
      token: blockingBridge.readToken(),
      op: 'generate',
      provider: 'fake-provider',
      model: 'fake-model',
      prompt: 'wait',
    })}\n`,
  )
  await once(socket, 'data')
  await blockingBridge.close()
  assert.equal(aborted, true)
  assert.equal(existsSync(abortSocketPath), false)
  assert.equal(existsSync(abortTokenPath), false)
} finally {
  await blockingBridge.close()
  rmSync(abortRoot, { recursive: true, force: true })
}

async function request(path, payload) {
  const socket = createConnection(path)
  socket.setEncoding('utf8')
  await once(socket, 'connect')
  socket.write(`${JSON.stringify(payload)}\n`)
  const events = []
  let buffer = ''
  for await (const chunk of socket) {
    buffer += chunk
    while (buffer.includes('\n')) {
      const index = buffer.indexOf('\n')
      const line = buffer.slice(0, index)
      buffer = buffer.slice(index + 1)
      if (line.trim()) events.push(JSON.parse(line))
    }
  }
  return events
}
