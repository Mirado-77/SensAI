import type {
  ApprovalRequest,
  Question,
  Risk,
  SensaiEvent,
  ServerFrame,
  StoredMessage,
  ToolInfo,
} from "@/lib/protocol"

export type ToolStatus = "running" | "done" | "error" | "failed"

export type ThreadItem =
  | { kind: "user"; id: string | null; text: string }
  | {
      kind: "assistant"
      id: string | null
      text: string
      status: "streaming" | "complete" | "interrupted"
    }
  | { kind: "plan"; goal: string; steps: string[] }
  | { kind: "error"; text: string }
  | {
      kind: "tool"
      callId: string | null
      tool: string
      risk: Risk | null
      args: Record<string, unknown>
      status: ToolStatus
      result: string | null
      durationMs: number | null
    }

export interface Usage {
  prompt: number
  completion: number
  turn: number
  lastPrompt: number
}

export interface SessionState {
  status: "connecting" | "ready" | "busy" | "closed"
  sessionId: string | null
  model: string | null
  persona: string | null
  tools: ToolInfo[]
  items: ThreadItem[]
  approval: { id: string; request: ApprovalRequest } | null
  question: { id: string; question: Question } | null
  usage: Usage
}

export type Action =
  | { type: "frame"; frame: ServerFrame }
  | { type: "history"; messages: StoredMessage[] }
  | { type: "sent"; text: string }
  | { type: "replied" }
  | { type: "closed" }
  | { type: "reset" }

export const initialState: SessionState = {
  status: "connecting",
  sessionId: null,
  model: null,
  persona: null,
  tools: [],
  items: [],
  approval: null,
  question: null,
  usage: { prompt: 0, completion: 0, turn: 0, lastPrompt: 0 },
}

export function reduce(state: SessionState, action: Action): SessionState {
  switch (action.type) {
    case "frame":
      return onFrame(state, action.frame)
    case "history":
      return { ...state, items: fromHistory(action.messages) }
    case "sent":
      return {
        ...state,
        status: "busy",
        items: [...state.items, { kind: "user", id: null, text: action.text }],
      }
    case "replied":
      return { ...state, approval: null, question: null }
    case "closed":
      return { ...state, status: "closed", approval: null, question: null }
    case "reset":
      return initialState
  }
}

function onFrame(state: SessionState, frame: ServerFrame): SessionState {
  switch (frame.kind) {
    case "ready":
      return {
        ...state,
        status: "ready",
        sessionId: frame.session_id,
        model: frame.model,
        persona: frame.persona,
        tools: frame.tools,
      }
    case "event":
      return onEvent(state, frame.event)
    case "confirm":
      return { ...state, approval: { id: frame.id, request: frame.request } }
    case "ask":
      return { ...state, question: { id: frame.id, question: frame.question } }
    case "persona":
      return { ...state, persona: frame.name }
    case "idle":
      return {
        ...state,
        status: "ready",
        approval: null,
        question: null,
        items: settle(state.items, "complete"),
      }
    case "warning":
    case "error":
      return state
  }
}

function onEvent(state: SessionState, event: SensaiEvent): SessionState {
  const payload = event.payload
  switch (event.type) {
    case "run.started":
      return {
        ...state,
        status: "busy",
        usage: { ...state.usage, turn: 0 },
      }
    case "model.chunk":
      return { ...state, items: appendChunk(state.items, String(payload.text)) }
    case "model.usage": {
      const prompt = Number(payload.prompt_tokens) || 0
      const completion = Number(payload.completion_tokens) || 0
      return {
        ...state,
        usage: {
          prompt: state.usage.prompt + prompt,
          completion: state.usage.completion + completion,
          turn: state.usage.turn + prompt + completion,
          lastPrompt: prompt,
        },
      }
    }
    case "react.plan": {
      const steps = (payload.steps as string[] | undefined) ?? []
      if (steps.length === 0) {
        return state
      }
      const plan: ThreadItem = { kind: "plan", goal: String(payload.goal), steps }
      return { ...state, items: [...settle(state.items, "complete"), plan] }
    }
    case "tool.started": {
      const tool: ThreadItem = {
        kind: "tool",
        callId: String(payload.call_id),
        tool: String(payload.tool),
        risk: (payload.risk as Risk | undefined) ?? null,
        args: (payload.args as Record<string, unknown> | undefined) ?? {},
        status: "running",
        result: null,
        durationMs: null,
      }
      return { ...state, items: [...settle(state.items, "complete"), tool] }
    }
    case "tool.finished":
      return {
        ...state,
        items: updateTool(state.items, payload, {
          status: payload.is_error ? "error" : "done",
          result: payload.content == null ? null : String(payload.content),
          durationMs: Number(payload.duration_ms) || null,
        }),
      }
    case "tool.failed":
      return {
        ...state,
        items: updateTool(state.items, payload, {
          status: "failed",
          result: String(payload.error),
        }),
      }
    case "approval.resolved":
      if (payload.approved) {
        return state
      }
      return {
        ...state,
        items: updateTool(settle(state.items, "complete"), payload, {
          status: "failed",
          result: "Refused by the user",
        }),
      }
    case "run.cancelled":
      return { ...state, items: settle(state.items, "interrupted") }
    case "run.completed":
      return { ...state, items: settle(state.items, "complete") }
    case "run.failed": {
      const error: ThreadItem = { kind: "error", text: String(payload.error) }
      return { ...state, items: [...settle(state.items, "complete"), error] }
    }
    default:
      return state
  }
}

function appendChunk(items: ThreadItem[], text: string): ThreadItem[] {
  const last = items.at(-1)
  if (last?.kind === "assistant" && last.status === "streaming") {
    return [...items.slice(0, -1), { ...last, text: last.text + text }]
  }
  return [...items, { kind: "assistant", id: null, text, status: "streaming" }]
}

function settle(
  items: ThreadItem[],
  status: "complete" | "interrupted"
): ThreadItem[] {
  const last = items.at(-1)
  if (last?.kind === "assistant" && last.status === "streaming") {
    return [...items.slice(0, -1), { ...last, status }]
  }
  return items
}

function updateTool(
  items: ThreadItem[],
  payload: Record<string, unknown>,
  changes: Partial<Extract<ThreadItem, { kind: "tool" }>>
): ThreadItem[] {
  const callId = String(payload.call_id)
  const index = items.findLastIndex(
    (item) => item.kind === "tool" && item.callId === callId
  )
  if (index === -1) {
    const tool: ThreadItem = {
      kind: "tool",
      callId,
      tool: String(payload.tool),
      risk: null,
      args: {},
      status: "failed",
      result: null,
      durationMs: null,
      ...changes,
    }
    return [...items, tool]
  }
  return items.map((item, i) =>
    i === index ? ({ ...item, ...changes } as ThreadItem) : item
  )
}

export function fromHistory(messages: StoredMessage[]): ThreadItem[] {
  const items: ThreadItem[] = []
  for (const message of messages) {
    if (message.role === "user") {
      items.push({ kind: "user", id: message.id, text: message.content })
    } else if (message.role === "assistant") {
      if (message.content) {
        items.push({
          kind: "assistant",
          id: message.id,
          text: message.content,
          status: message.status === "interrupted" ? "interrupted" : "complete",
        })
      }
      for (const call of message.tool_calls) {
        items.push({
          kind: "tool",
          callId: null,
          tool: call.name,
          risk: null,
          args: call.args,
          status: "done",
          result: null,
          durationMs: null,
        })
      }
    } else if (message.role === "tool") {
      const index = items.findLastIndex(
        (item) =>
          item.kind === "tool" &&
          item.tool === message.tool_name &&
          item.result === null
      )
      if (index !== -1) {
        items[index] = {
          ...(items[index] as Extract<ThreadItem, { kind: "tool" }>),
          result: message.content,
        }
      }
    }
  }
  return items
}
