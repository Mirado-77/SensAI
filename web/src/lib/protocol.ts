export type Risk = "read" | "write" | "exec"

export interface SensaiEvent {
  type: string
  run_id: string
  ts: number
  payload: Record<string, unknown>
}

export interface ApprovalRequest {
  run_id: string
  tool: string
  risk: Risk
  preview: string
  args: Record<string, unknown>
}

export interface Question {
  question: string
  options?: string[]
}

export interface ToolInfo {
  name: string
  risk: Risk
  description: string
}

export type ServerFrame =
  | {
      kind: "ready"
      session_id: string
      model: string
      persona: string | null
      tools: ToolInfo[]
    }
  | { kind: "event"; event: SensaiEvent }
  | { kind: "confirm"; id: string; request: ApprovalRequest }
  | { kind: "ask"; id: string; question: Question }
  | { kind: "persona"; name: string | null }
  | { kind: "idle" }
  | { kind: "warning"; message: string }
  | { kind: "error"; message: string }

export type ClientFrame =
  | { type: "message"; text: string }
  | { type: "cancel" }
  | { type: "approval"; id: string; allowed: boolean }
  | { type: "answer"; id: string; answer: string }
  | { type: "persona"; name: string | null }

export interface StoredToolCall {
  name: string
  args: Record<string, unknown>
}

export interface StoredMessage {
  id: string | null
  role: "system" | "user" | "assistant" | "tool"
  content: string
  status: "complete" | "interrupted"
  tool_calls: StoredToolCall[]
  tool_name: string | null
}

export interface SessionSummary {
  id: string
  created_at: number
  first_message: string | null
}

export interface Persona {
  name: string
  description: string
}

export interface ApiConfig {
  model: string
  temperature: number
  prompt: string | null
  prompts: string[]
  persona: string | null
  personas: Persona[]
  max_steps: number
  max_tokens: number
}
