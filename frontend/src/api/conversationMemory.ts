export const CONVERSATION_MEMORY_WINDOW = 60

export type ConversationMemoryLevel = 'empty' | 'steady' | 'high' | 'full'

export interface ConversationMemoryProgress {
  used: number
  capacity: number
  percent: number
  level: ConversationMemoryLevel
  label: string
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max)
}

export function getConversationMemoryProgress(
  messages: readonly unknown[],
  capacity = CONVERSATION_MEMORY_WINDOW,
): ConversationMemoryProgress {
  const safeCapacity = Math.max(1, Math.floor(capacity))
  const used = clamp(messages.length, 0, safeCapacity)
  const percent = Math.round((used / safeCapacity) * 100)
  const level: ConversationMemoryLevel =
    used === 0 ? 'empty' :
    percent >= 100 ? 'full' :
    percent >= 80 ? 'high' :
    'steady'

  return {
    used,
    capacity: safeCapacity,
    percent,
    level,
    label: `${used} / ${safeCapacity}`,
  }
}
