import { useEffect, useRef } from "react"
import {
  CircleAlertIcon,
  GitBranchIcon,
  ListChecksIcon,
  MessageSquareIcon,
} from "lucide-react"

import { ToolCard } from "@/components/tool-card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty"
import { Spinner } from "@/components/ui/spinner"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import type { ThreadItem } from "@/lib/reducer"
import { cn } from "@/lib/utils"

interface ThreadProps {
  items: ThreadItem[]
  busy: boolean
  onFork: (messageId: string) => void
}

function ForkButton({ id, onFork }: { id: string | null; onFork: (id: string) => void }) {
  if (id === null) {
    return null
  }
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon-xs"
          className="opacity-0 group-hover:opacity-100"
          onClick={() => onFork(id)}
          aria-label="Fork from this message"
        >
          <GitBranchIcon />
        </Button>
      </TooltipTrigger>
      <TooltipContent>Continue in a new session from here</TooltipContent>
    </Tooltip>
  )
}

function Item({ item, onFork }: { item: ThreadItem; onFork: (id: string) => void }) {
  switch (item.kind) {
    case "user":
      return (
        <div className="group flex items-start justify-end gap-1">
          <ForkButton id={item.id} onFork={onFork} />
          <div className="max-w-[80%] rounded-2xl bg-muted px-4 py-2 whitespace-pre-wrap">
            {item.text}
          </div>
        </div>
      )
    case "assistant":
      return (
        <div className="group flex items-start gap-1">
          <div
            className={cn(
              "min-w-0 flex-1 leading-relaxed whitespace-pre-wrap",
              item.status === "streaming" && "after:animate-pulse after:content-['▍']"
            )}
          >
            {item.text}
            {item.status === "interrupted" && (
              <Badge variant="outline" className="ml-2 align-middle">
                interrupted
              </Badge>
            )}
          </div>
          <ForkButton id={item.id} onFork={onFork} />
        </div>
      )
    case "plan":
      return (
        <div className="flex flex-col gap-1 rounded-lg border border-dashed px-3 py-2 text-sm">
          <div className="flex items-center gap-2 font-medium">
            <ListChecksIcon className="size-4" />
            {item.goal}
          </div>
          <ol className="list-decimal pl-9 text-muted-foreground">
            {item.steps.map((step, index) => (
              <li key={index}>{step}</li>
            ))}
          </ol>
        </div>
      )
    case "tool":
      return <ToolCard item={item} />
    case "error":
      return (
        <Alert variant="destructive">
          <CircleAlertIcon />
          <AlertTitle>The turn failed and was not saved</AlertTitle>
          <AlertDescription>{item.text}</AlertDescription>
        </Alert>
      )
  }
}

export function Thread({ items, busy, onFork }: ThreadProps) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" })
  }, [items])

  if (items.length === 0 && !busy) {
    return (
      <Empty className="flex-1">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <MessageSquareIcon />
          </EmptyMedia>
          <EmptyTitle>New conversation</EmptyTitle>
          <EmptyDescription>
            Ask SENSAI about the project, or to fix a failing test.
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }
  const waiting = busy && items.at(-1)?.kind !== "assistant"
  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto flex max-w-3xl flex-col gap-4 px-4 py-6">
        {items.map((item, index) => (
          <Item key={index} item={item} onFork={onFork} />
        ))}
        {waiting && <Spinner className="text-muted-foreground" />}
        <div ref={endRef} />
      </div>
    </div>
  )
}
