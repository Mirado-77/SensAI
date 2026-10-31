import { useCallback, useEffect, useState } from "react"
import { toast } from "sonner"

import { ApprovalDialog } from "@/components/approval-dialog"
import { Composer } from "@/components/composer"
import { QuestionCard } from "@/components/question-card"
import { SessionHeader } from "@/components/session-header"
import { SessionList } from "@/components/session-list"
import { Thread } from "@/components/thread"
import { useSession } from "@/hooks/use-session"
import { api } from "@/lib/api"
import type { ApiConfig, SessionSummary } from "@/lib/protocol"

function hashSession(): string | null {
  return window.location.hash.slice(1) || null
}

export function App() {
  const [config, setConfig] = useState<ApiConfig | null>(null)
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [current, setCurrent] = useState<string | null>(hashSession)
  const [connection, setConnection] = useState(0)

  const refreshSessions = useCallback(() => {
    api.sessions().then(setSessions).catch((error: Error) => toast.error(error.message))
  }, [])

  const open = useCallback((id: string) => {
    window.location.hash = id
    setCurrent(id)
  }, [])

  const newSession = useCallback(() => {
    api
      .createSession()
      .then(({ id }) => open(id))
      .catch((error: Error) => toast.error(error.message))
  }, [open])

  useEffect(() => {
    api.config().then(setConfig).catch((error: Error) => toast.error(error.message))
    refreshSessions()
    if (hashSession() === null) {
      newSession()
    }
    const onHash = () => setCurrent(hashSession())
    window.addEventListener("hashchange", onHash)
    return () => window.removeEventListener("hashchange", onHash)
  }, [refreshSessions, newSession])

  const session = useSession(current, connection, refreshSessions)
  const { state } = session
  const busy = state.status === "busy"

  const fork = (messageId: string) => {
    if (current === null) {
      return
    }
    api
      .fork(current, messageId)
      .then(({ id }) => {
        open(id)
        refreshSessions()
        toast.success("Forked into a new session")
      })
      .catch((error: Error) => toast.error(error.message))
  }

  return (
    <div className="flex h-svh bg-background text-foreground">
      <SessionList
        sessions={sessions}
        current={current}
        onSelect={open}
        onNew={newSession}
      />
      <main className="flex min-w-0 flex-1 flex-col">
        <SessionHeader
          state={state}
          config={config}
          onPersona={session.setPersona}
          onReconnect={() => setConnection((value) => value + 1)}
          onNew={newSession}
        />
        <Thread items={state.items} busy={busy} onFork={fork} />
        {state.question && (
          <div className="mx-auto w-full max-w-3xl px-4 pb-2">
            <QuestionCard question={state.question} onAnswer={session.answer} />
          </div>
        )}
        <Composer
          busy={busy}
          disabled={state.status !== "ready" && !busy}
          onSend={session.sendMessage}
          onCancel={session.cancel}
        />
      </main>
      <ApprovalDialog approval={state.approval} onAnswer={session.approve} />
    </div>
  )
}

export default App
