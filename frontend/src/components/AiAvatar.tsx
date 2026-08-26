import { Heart, Bot } from 'lucide-react'

interface AiAvatarProps {
  type: 'love' | 'agent'
  size?: number
}

function AiAvatar({ type, size = 40 }: AiAvatarProps) {
  const className = `ai-avatar ${type}`
  const Icon = type === 'love' ? Heart : Bot

  return (
    <div
      className={className}
      style={{ width: size, height: size }}
    >
      <Icon size={size * 0.5} />
    </div>
  )
}

export default AiAvatar
