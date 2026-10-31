import { useState } from "react"
import { SendIcon, SquareIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

interface ComposerProps {
  busy: boolean
  disabled: boolean
  onSend: (text: string) => void
  onCancel: () => void
}

export function Composer({ busy, disabled, onSend, onCancel }: ComposerProps) {
  const [text, setText] = useState("")

  const submit = () => {
    const value = text.trim()
    if (value && !busy && !disabled) {
      onSend(value)
      setText("")
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-3xl items-end gap-2 px-4 pb-4">
      <Textarea
        value={text}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault()
            submit()
          }
        }}
        placeholder="Message SENSAI (Enter to send, Shift+Enter for a new line)"
        className="max-h-48 min-h-11"
        disabled={disabled}
        autoFocus
      />
      {busy ? (
        <Button variant="outline" size="icon-lg" onClick={onCancel} aria-label="Stop">
          <SquareIcon />
        </Button>
      ) : (
        <Button
          size="icon-lg"
          onClick={submit}
          disabled={disabled || !text.trim()}
          aria-label="Send"
        >
          <SendIcon />
        </Button>
      )}
    </div>
  )
}
