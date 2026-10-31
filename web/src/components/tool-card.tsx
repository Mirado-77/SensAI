import {
  ChevronRightIcon,
  CircleAlertIcon,
  CircleCheckIcon,
  CircleXIcon,
  WrenchIcon,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { Spinner } from "@/components/ui/spinner"
import type { ThreadItem, ToolStatus } from "@/lib/reducer"

const MAX_RESULT_CHARS = 4000

const STATUS_ICONS: Record<ToolStatus, React.ReactNode> = {
  running: <Spinner />,
  done: <CircleCheckIcon className="text-muted-foreground" />,
  error: <CircleAlertIcon className="text-destructive" />,
  failed: <CircleXIcon className="text-destructive" />,
}

function summary(args: Record<string, unknown>): string {
  const first = Object.values(args).find((value) => typeof value === "string")
  return typeof first === "string" ? first.split("\n")[0] : ""
}

export function ToolCard({ item }: { item: Extract<ThreadItem, { kind: "tool" }> }) {
  const result =
    item.result && item.result.length > MAX_RESULT_CHARS
      ? `${item.result.slice(0, MAX_RESULT_CHARS)}\n…`
      : item.result
  return (
    <Collapsible className="rounded-lg border text-sm">
      <CollapsibleTrigger className="group flex w-full items-center gap-2 px-3 py-2 text-left">
        <ChevronRightIcon className="size-4 transition-transform group-data-[state=open]:rotate-90" />
        <WrenchIcon className="size-4 text-muted-foreground" />
        <span className="font-mono">{item.tool}</span>
        <span className="min-w-0 flex-1 truncate text-muted-foreground">
          {summary(item.args)}
        </span>
        {item.risk && <Badge variant="outline">{item.risk}</Badge>}
        {item.durationMs !== null && (
          <span className="text-xs text-muted-foreground">
            {Math.round(item.durationMs)} ms
          </span>
        )}
        {STATUS_ICONS[item.status]}
      </CollapsibleTrigger>
      <CollapsibleContent className="flex flex-col gap-2 border-t px-3 py-2">
        <pre className="overflow-auto font-mono text-xs text-muted-foreground">
          {JSON.stringify(item.args, null, 2)}
        </pre>
        {result && (
          <pre className="max-h-64 overflow-auto rounded-md bg-muted p-2 font-mono text-xs">
            {result}
          </pre>
        )}
      </CollapsibleContent>
    </Collapsible>
  )
}
