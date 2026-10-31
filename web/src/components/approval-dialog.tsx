import { CheckIcon, XIcon } from "lucide-react"

import { DiffView } from "@/components/diff-view"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import type { ApprovalRequest } from "@/lib/protocol"

interface ApprovalDialogProps {
  approval: { id: string; request: ApprovalRequest } | null
  onAnswer: (id: string, allowed: boolean) => void
}

export function ApprovalDialog({ approval, onAnswer }: ApprovalDialogProps) {
  if (approval === null) {
    return null
  }
  const { id, request } = approval
  return (
    <Dialog open onOpenChange={(open) => !open && onAnswer(id, false)}>
      <DialogContent className="sm:max-w-2xl" showCloseButton={false}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            Approval required
            <Badge variant={request.risk === "exec" ? "destructive" : "secondary"}>
              {request.risk}
            </Badge>
          </DialogTitle>
          <DialogDescription>
            SENSAI wants to run <code className="font-mono">{request.tool}</code>.
            Nothing happens until you answer.
          </DialogDescription>
        </DialogHeader>
        <DiffView preview={request.preview} />
        <DialogFooter>
          <Button variant="outline" onClick={() => onAnswer(id, false)}>
            <XIcon data-icon="inline-start" />
            Refuse
          </Button>
          <Button onClick={() => onAnswer(id, true)} autoFocus>
            <CheckIcon data-icon="inline-start" />
            Allow
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
