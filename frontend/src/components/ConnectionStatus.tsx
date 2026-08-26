interface ConnectionStatusProps {
  status: 'disconnected' | 'connected' | 'error'
}

const STATUS_LABELS: Record<string, string> = {
  disconnected: '未连接',
  connected: '连接成功',
  error: '连接错误',
}

function ConnectionStatus({ status }: ConnectionStatusProps) {
  return (
    <div className="connection-status">
      <span className={`status-dot ${status}`} />
      <span>{STATUS_LABELS[status] || status}</span>
    </div>
  )
}

export default ConnectionStatus
