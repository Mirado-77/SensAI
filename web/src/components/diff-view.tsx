import { cn } from "@/lib/utils"

function lineClass(line: string): string {
  if (line.startsWith("+++") || line.startsWith("---")) {
    return "font-semibold"
  }
  if (line.startsWith("@@")) {
    return "text-muted-foreground"
  }
  if (line.startsWith("+")) {
    return "text-diff-added"
  }
  if (line.startsWith("-")) {
    return "text-diff-removed"
  }
  return ""
}

export function DiffView({ preview }: { preview: string }) {
  const isDiff = preview.startsWith("--- ")
  return (
    <pre className="max-h-[50vh] overflow-auto rounded-lg bg-muted p-3 font-mono text-xs leading-relaxed">
      {preview.split("\n").map((line, index) => (
        <div key={index} className={cn(isDiff && lineClass(line))}>
          {line || " "}
        </div>
      ))}
    </pre>
  )
}
