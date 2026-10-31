import { SettingsIcon } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import type { ApiConfig, ToolInfo } from "@/lib/protocol"

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right font-mono">{value}</span>
    </div>
  )
}

export function ConfigSheet({ config, tools }: { config: ApiConfig | null; tools: ToolInfo[] }) {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Configuration">
          <SettingsIcon />
        </Button>
      </SheetTrigger>
      <SheetContent className="overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Configuration</SheetTitle>
          <SheetDescription>
            Read from the TOML files. Change them and restart `sensai start:api`.
          </SheetDescription>
        </SheetHeader>
        {config && (
          <div className="flex flex-col gap-2 px-4">
            <Row label="Model" value={config.model} />
            <Row label="Temperature" value={config.temperature} />
            <Row label="System prompt" value={config.prompt || "none"} />
            <Row label="Prompt versions" value={config.prompts.join(", ")} />
            <Row label="Default persona" value={config.persona || "none"} />
            <Row label="Max steps per turn" value={config.max_steps} />
            <Row label="Max tokens per turn" value={config.max_tokens} />
          </div>
        )}
        <Separator />
        <div className="flex flex-col gap-3 px-4 pb-4">
          <span className="text-sm font-medium">Tools ({tools.length})</span>
          {tools.map((tool) => (
            <div key={tool.name} className="flex flex-col gap-1">
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm">{tool.name}</span>
                <Badge variant={tool.risk === "read" ? "secondary" : "outline"}>
                  {tool.risk}
                </Badge>
              </div>
              <span className="text-xs text-muted-foreground">
                {tool.description.split("\n")[0]}
              </span>
            </div>
          ))}
        </div>
      </SheetContent>
    </Sheet>
  )
}
