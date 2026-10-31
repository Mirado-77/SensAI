import { Progress } from "@/components/ui/progress"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import type { Usage } from "@/lib/reducer"

function compact(tokens: number): string {
  return tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k` : String(tokens)
}

export function UsageMeter({ usage, maxTokens }: { usage: Usage; maxTokens: number | null }) {
  const percent = maxTokens ? Math.min(100, (usage.turn / maxTokens) * 100) : 0
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="flex w-40 flex-col gap-1 text-xs text-muted-foreground">
          <div className="flex justify-between">
            <span>Turn</span>
            <span>
              {compact(usage.turn)}
              {maxTokens ? ` / ${compact(maxTokens)}` : ""}
            </span>
          </div>
          <Progress value={percent} />
        </div>
      </TooltipTrigger>
      <TooltipContent>
        <div className="flex flex-col gap-0.5">
          <span>Context of the last call: {compact(usage.lastPrompt)} tokens</span>
          <span>
            Session: {compact(usage.prompt + usage.completion)} tokens (prompt{" "}
            {compact(usage.prompt)}, completion {compact(usage.completion)})
          </span>
          <span>Turn budget: [runtime] max_tokens</span>
        </div>
      </TooltipContent>
    </Tooltip>
  )
}
