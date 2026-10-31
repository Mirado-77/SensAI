import { useCallback, useEffect, useReducer, useRef } from "react"
import { toast } from "sonner"

import { api, socketUrl } from "@/lib/api"
import type { ClientFrame, ServerFrame } from "@/lib/protocol"
import { initialState, reduce } from "@/lib/reducer"

function notify(frame: ServerFrame) {
  if (frame.kind === "error") {
    toast.error(frame.message)
  } else if (frame.kind === "warning") {
    toast.warning(frame.message)
  } else if (frame.kind === "event") {
    const { type } = frame.event
    if (type === "guard.blocked") {
      toast.warning("Message blocked by the guards (EV2)")
    } else if (type === "cache.hit") {
      toast.info("Answered from the semantic cache (A6)")
    }
  }
}

export function useSession(
  sessionId: string | null,
  connection: number,
  onTurnEnd: () => void
) {
  const [state, dispatch] = useReducer(reduce, initialState)
  const socketRef = useRef<WebSocket | null>(null)
  const onTurnEndRef = useRef(onTurnEnd)

  useEffect(() => {
    onTurnEndRef.current = onTurnEnd
  }, [onTurnEnd])

  useEffect(() => {
    dispatch({ type: "reset" })
    if (sessionId === null) {
      return
    }
    let alive = true
    let failed = false
    const loadHistory = () =>
      api
        .messages(sessionId)
        .then((messages) => alive && dispatch({ type: "history", messages }))
        .catch((error: Error) => toast.error(error.message))
    loadHistory()
    const socket = new WebSocket(socketUrl(sessionId))
    socketRef.current = socket
    socket.onmessage = (message) => {
      if (!alive) {
        return
      }
      const frame = JSON.parse(message.data) as ServerFrame
      dispatch({ type: "frame", frame })
      notify(frame)
      if (frame.kind === "event" && frame.event.type === "run.failed") {
        failed = true
      } else if (frame.kind === "idle") {
        // A failed turn is not saved: reloading would erase the message and the error.
        if (!failed) {
          loadHistory()
        }
        failed = false
        onTurnEndRef.current()
      }
    }
    socket.onclose = () => alive && dispatch({ type: "closed" })
    return () => {
      alive = false
      socket.close()
      socketRef.current = null
    }
  }, [sessionId, connection])

  const send = useCallback((frame: ClientFrame) => {
    const socket = socketRef.current
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(frame))
    }
  }, [])

  const sendMessage = useCallback(
    (text: string) => {
      dispatch({ type: "sent", text })
      send({ type: "message", text })
    },
    [send]
  )

  const reply = useCallback(
    (frame: ClientFrame) => {
      dispatch({ type: "replied" })
      send(frame)
    },
    [send]
  )

  return {
    state,
    sendMessage,
    cancel: useCallback(() => send({ type: "cancel" }), [send]),
    approve: (id: string, allowed: boolean) =>
      reply({ type: "approval", id, allowed }),
    answer: (id: string, answer: string) =>
      reply({ type: "answer", id, answer }),
    setPersona: useCallback(
      (name: string | null) => send({ type: "persona", name }),
      [send]
    ),
  }
}
