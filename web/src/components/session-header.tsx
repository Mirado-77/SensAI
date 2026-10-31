import { PlusIcon, RefreshCwIcon } from "lucide-react"

import { ConfigSheet } from "@/components/config-sheet"
import { UsageMeter } from "@/components/usage-meter"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import type { ApiConfig } from "@/lib/protocol"
import type { SessionState } from "@/lib/reducer"

const NO_PERSONA = "none"

interface SessionHeaderProps {
  state: SessionState
  config: ApiConfig | null
  onPersona: (name: string | null) => void
  onReconnect: () => void
  onNew: () => void
}

export function SessionHeader({
  state,
  config,
  onPersona,
  onReconnect,
  onNew,
}: SessionHeaderProps) {
  return (
    <header className="flex flex-wrap items-center gap-3 border-b px-4 py-2">
      <Button variant="outline" size="icon-sm" className="md:hidden" onClick={onNew} aria-label="New session">
        <PlusIcon />
      </Button>
      <Badge variant="secondary">{state.model ?? config?.model ?? "…"}</Badge>
      <Select
        value={state.persona ?? NO_PERSONA}
        onValueChange={(value) => onPersona(value === NO_PERSONA ? null : value)}
        disabled={state.status !== "ready"}
      >
        <SelectTrigger size="sm" aria-label="Persona">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectItem value={NO_PERSONA}>No persona</SelectItem>
            {config?.personas.map((persona) => (
              <SelectItem key={persona.name} value={persona.name}>
                {persona.name}
              </SelectItem>
            ))}
          </SelectGroup>
        </SelectContent>
      </Select>
      {state.status === "connecting" && <Badge variant="outline">connecting</Badge>}
      {state.status === "closed" && (
        <Button variant="destructive" size="sm" onClick={onReconnect}>
          <RefreshCwIcon data-icon="inline-start" />
          Disconnected, reconnect
        </Button>
      )}
      <div className="ml-auto flex items-center gap-3">
        <UsageMeter usage={state.usage} maxTokens={config?.max_tokens ?? null} />
        <ConfigSheet config={config} tools={state.tools} />
      </div>
    </header>
  )
}
