import { Database } from 'lucide-react'
import { getConversationMemoryProgress } from '../api/conversationMemory'

interface ConversationMemoryMeterProps {
  messages: readonly unknown[]
}

function ConversationMemoryMeter({ messages }: ConversationMemoryMeterProps) {
  const progress = getConversationMemoryProgress(messages)

  return (
    <div
      className="conversation-memory-meter"
      data-level={progress.level}
      role="progressbar"
      aria-label="对话记忆"
      aria-valuemin={0}
      aria-valuemax={progress.capacity}
      aria-valuenow={progress.used}
      aria-valuetext={progress.label}
      title={`当前对话记忆进度 ${progress.label}`}
    >
      <Database className="w-3.5 h-3.5 shrink-0" strokeWidth={1.5} />
      <span className="conversation-memory-meter-label">对话记忆</span>
      <span className="conversation-memory-meter-track" aria-hidden="true">
        <span
          className="conversation-memory-meter-fill"
          style={{ transform: `scaleX(${progress.percent / 100})` }}
        />
      </span>
      <span className="conversation-memory-meter-value">{progress.label}</span>
    </div>
  )
}

export default ConversationMemoryMeter
