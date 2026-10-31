import { useState } from "react"
import { MessageCircleQuestionIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import type { Question } from "@/lib/protocol"

interface QuestionCardProps {
  question: { id: string; question: Question }
  onAnswer: (id: string, answer: string) => void
}

export function QuestionCard({ question, onAnswer }: QuestionCardProps) {
  const [text, setText] = useState("")
  const { id } = question
  const options = question.question.options ?? []
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MessageCircleQuestionIcon />
          {question.question.question}
        </CardTitle>
        <CardDescription>
          SENSAI needs this answer to continue.
        </CardDescription>
      </CardHeader>
      {options.length > 0 && (
        <CardContent className="flex flex-wrap gap-2">
          {options.map((option) => (
            <Button
              key={option}
              variant="outline"
              size="sm"
              onClick={() => onAnswer(id, option)}
            >
              {option}
            </Button>
          ))}
        </CardContent>
      )}
      <CardFooter className="flex gap-2">
        <Textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Or type your own answer"
          className="min-h-9"
          rows={1}
        />
        <Button
          disabled={!text.trim()}
          onClick={() => onAnswer(id, text.trim())}
        >
          Answer
        </Button>
      </CardFooter>
    </Card>
  )
}
