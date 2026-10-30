import { describe, expect, it } from "vitest"

import type { ServerFrame, StoredMessage } from "@/lib/protocol"
import { initialState, reduce, type SessionState } from "@/lib/reducer"

function event(type: string, payload: Record<string, unknown> = {}): ServerFrame {
  return { kind: "event", event: { type, run_id: "r1", ts: 0, payload } }
}

function play(...frames: ServerFrame[]): SessionState {
  return frames.reduce(
    (state, frame) => reduce(state, { type: "frame", frame }),
    initialState
  )
}

const ready: ServerFrame = {
  kind: "ready",
  session_id: "s1",
  model: "qwen",
  persona: "dev",
  tools: [],
}

describe("reduce", () => {
  it("joins the streamed chunks into one answer", () => {
    const state = play(
      ready,
      event("run.started"),
      event("model.chunk", { text: "Hel" }),
      event("model.chunk", { text: "lo" }),
      event("run.completed"),
      { kind: "idle" }
    )
    expect(state.status).toBe("ready")
    expect(state.items).toEqual([
      { kind: "assistant", id: null, text: "Hello", status: "complete" },
    ])
  })

  it("puts a tool call between two answers and records its outcome", () => {
    const state = play(
      ready,
      event("model.chunk", { text: "Reading" }),
      event("tool.started", {
        call_id: "c1",
        tool: "read_file",
        risk: "read",
        args: { path: "a.py" },
      }),
      event("tool.finished", {
        call_id: "c1",
        tool: "read_file",
        is_error: false,
        content: "x = 1",
        duration_ms: 12,
      }),
      event("model.chunk", { text: "Done" })
    )
    expect(state.items.map((item) => item.kind)).toEqual([
      "assistant",
      "tool",
      "assistant",
    ])
    expect(state.items[1]).toMatchObject({
      status: "done",
      result: "x = 1",
      durationMs: 12,
    })
  })

  it("shows a refused tool as failed", () => {
    const state = play(
      ready,
      event("approval.resolved", { call_id: "c2", tool: "run_command", approved: false })
    )
    expect(state.items).toEqual([
      {
        kind: "tool",
        callId: "c2",
        tool: "run_command",
        risk: null,
        args: {},
        status: "failed",
        result: "Refused by the user",
        durationMs: null,
      },
    ])
  })

  it("marks a cancelled answer as interrupted", () => {
    const state = play(
      ready,
      event("model.chunk", { text: "partial" }),
      event("run.cancelled", { partial: true })
    )
    expect(state.items[0]).toMatchObject({ status: "interrupted" })
  })

  it("shows the error of a failed turn in the thread", () => {
    const state = play(
      ready,
      event("model.chunk", { text: "Hi" }),
      event("run.failed", { error: "Ollama est injoignable" })
    )
    expect(state.items).toEqual([
      { kind: "assistant", id: null, text: "Hi", status: "complete" },
      { kind: "error", text: "Ollama est injoignable" },
    ])
  })

  it("adds up the tokens and resets the turn budget at each run", () => {
    const state = play(
      ready,
      event("run.started"),
      event("model.usage", { prompt_tokens: 100, completion_tokens: 10 }),
      event("run.started"),
      event("model.usage", { prompt_tokens: 200, completion_tokens: 20 })
    )
    expect(state.usage).toEqual({
      prompt: 300,
      completion: 30,
      turn: 220,
      lastPrompt: 200,
    })
  })

  it("keeps an approval until it is answered or the turn ends", () => {
    const request = {
      run_id: "r1",
      tool: "write_file",
      risk: "write" as const,
      preview: "+x",
      args: {},
    }
    const asked = play(ready, { kind: "confirm", id: "a1", request })
    expect(asked.approval).toEqual({ id: "a1", request })
    expect(reduce(asked, { type: "replied" }).approval).toBeNull()
    expect(reduce(asked, { type: "frame", frame: { kind: "idle" } }).approval).toBeNull()
  })

  it("shows the user message as soon as it is sent", () => {
    const state = reduce(play(ready), { type: "sent", text: "hi" })
    expect(state.status).toBe("busy")
    expect(state.items).toEqual([{ kind: "user", id: null, text: "hi" }])
  })

  it("rebuilds the thread from the saved messages", () => {
    const messages: StoredMessage[] = [
      { id: "m1", role: "user", content: "fix", status: "complete", tool_calls: [], tool_name: null },
      {
        id: "m2",
        role: "assistant",
        content: "",
        status: "complete",
        tool_calls: [{ name: "read_file", args: { path: "a.py" } }],
        tool_name: null,
      },
      { id: "m3", role: "tool", content: "x = 1", status: "complete", tool_calls: [], tool_name: "read_file" },
      { id: "m4", role: "assistant", content: "half", status: "interrupted", tool_calls: [], tool_name: null },
    ]
    const state = reduce(initialState, { type: "history", messages })
    expect(state.items).toEqual([
      { kind: "user", id: "m1", text: "fix" },
      {
        kind: "tool",
        callId: null,
        tool: "read_file",
        risk: null,
        args: { path: "a.py" },
        status: "done",
        result: "x = 1",
        durationMs: null,
      },
      { kind: "assistant", id: "m4", text: "half", status: "interrupted" },
    ])
  })
})
