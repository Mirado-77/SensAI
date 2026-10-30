import type { ApiConfig, SessionSummary, StoredMessage } from "@/lib/protocol"

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  })
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(body?.detail ?? `Erreur HTTP ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  config: () => request<ApiConfig>("/api/config"),
  sessions: () => request<SessionSummary[]>("/api/sessions?limit=50"),
  createSession: () =>
    request<{ id: string }>("/api/sessions", { method: "POST" }),
  messages: (sessionId: string) =>
    request<StoredMessage[]>(`/api/sessions/${sessionId}/messages`),
  fork: (sessionId: string, messageId: string) =>
    request<{ id: string }>(`/api/sessions/${sessionId}/fork`, {
      method: "POST",
      body: JSON.stringify({ message_id: messageId }),
    }),
}

export function socketUrl(sessionId: string): string {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws"
  return `${scheme}://${window.location.host}/ws/sessions/${sessionId}`
}
