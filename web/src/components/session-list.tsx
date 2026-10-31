import { PlusIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { SessionSummary } from "@/lib/protocol"
import { cn } from "@/lib/utils"

interface SessionListProps {
  sessions: SessionSummary[]
  current: string | null
  onSelect: (id: string) => void
  onNew: () => void
}

function date(seconds: number): string {
  return new Date(seconds * 1000).toLocaleString(undefined, {
    dateStyle: "short",
    timeStyle: "short",
  })
}

export function SessionList({ sessions, current, onSelect, onNew }: SessionListProps) {
  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r md:flex">
      <div className="flex items-center justify-between px-4 py-3">
        <span className="font-heading font-medium">SENSAI</span>
        <Button variant="outline" size="sm" onClick={onNew}>
          <PlusIcon data-icon="inline-start" />
          New
        </Button>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <nav className="flex flex-col gap-0.5 px-2 pb-4">
          {sessions.map((session) => (
            <button
              key={session.id}
              onClick={() => onSelect(session.id)}
              className={cn(
                "flex flex-col rounded-md px-2 py-1.5 text-left text-sm hover:bg-muted",
                session.id === current && "bg-muted"
              )}
            >
              <span className="truncate">
                {session.first_message?.split("\n")[0] || "(empty)"}
              </span>
              <span className="text-xs text-muted-foreground">
                {date(session.created_at)} · {session.id.slice(0, 8)}
              </span>
            </button>
          ))}
        </nav>
      </ScrollArea>
    </aside>
  )
}
