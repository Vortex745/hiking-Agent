type ConfigObject = Record<string, unknown>

export type McpImportChoice = { name: string; config: ConfigObject }
export type McpImportedForm = {
  name: string
  transport: 'http' | 'stdio'
  url: string
  command: string
  args: string
  credentials: string
  confirmed: false
}

const isObject = (value: unknown): value is ConfigObject => value !== null && typeof value === 'object' && !Array.isArray(value)

export function readMcpImport(text: string): McpImportChoice[] {
  let value: unknown
  try { value = JSON.parse(text) }
  catch { throw new Error('JSON 格式无效，请检查括号、逗号和引号') }
  if (!isObject(value)) throw new Error('配置必须是 JSON 对象')

  const containers = ['mcpServers', 'servers'].filter(key => key in value)
  if (containers.length > 1) throw new Error('请只提供 mcpServers 或 servers 中的一种配置')
  if (!containers.length) return [{ name: '', config: value }]

  const container = containers[0]
  const extra = Object.keys(value).filter(key => key !== container && key !== '$schema')
  if (extra.length) throw new Error(`暂不支持配置项：${extra.join('、')}。请移除后重试`)
  const entries = value[container]
  if (!isObject(entries) || !Object.keys(entries).length) throw new Error(`${container} 中没有 MCP 服务`)
  return Object.entries(entries).map(([name, config]) => {
    if (!name.trim() || !isObject(config)) throw new Error(`${container} 中的服务名称或配置无效`)
    return { name, config }
  })
}

export function importMcpChoice({ name, config }: McpImportChoice): McpImportedForm {
  const allowed = ['name', 'type', 'transport', 'command', 'args', 'env', 'url', 'headers']
  const extra = Object.keys(config).filter(key => !allowed.includes(key))
  if (extra.length) throw new Error(`暂不支持配置项：${extra.join('、')}。请移除后重试`)
  if (config.name !== undefined && (typeof config.name !== 'string' || (name && config.name !== name))) {
    throw new Error('服务名称不匹配，请检查 name 配置')
  }
  if (config.type !== undefined && config.transport !== undefined && config.type !== config.transport) {
    throw new Error('type 和 transport 必须一致')
  }
  const type = config.type !== undefined ? config.type : config.transport !== undefined ? config.transport : (config.command !== undefined ? 'stdio' : 'http')
  if (type !== 'stdio' && type !== 'http') throw new Error('仅支持 HTTP 和 stdio，请检查服务类型')
  const conflicting = type === 'stdio' ? ['url', 'headers'] : ['command', 'args', 'env']
  if (conflicting.some(key => key in config)) throw new Error('连接方式与配置字段不匹配，请检查命令或 URL')

  if (type === 'stdio' && (typeof config.command !== 'string' || !config.command.trim())) {
    throw new Error('stdio 配置需要非空的 command')
  }
  if (type === 'http' && (typeof config.url !== 'string' || !config.url.trim())) {
    throw new Error('HTTP 配置需要非空的 url')
  }
  const args = config.args === undefined ? [] : config.args
  if (!Array.isArray(args) || args.some(value => typeof value !== 'string')) throw new Error('args 必须是字符串数组')
  const credentialKey = type === 'http' ? 'headers' : 'env'
  const credentials = config[credentialKey] === undefined ? {} : config[credentialKey]
  if (!isObject(credentials) || Object.values(credentials).some(value => typeof value !== 'string')) {
    throw new Error(`${credentialKey} 必须是字符串键值对象`)
  }
  const values = [config.command, config.url, ...args, ...Object.values(credentials)]
  if (values.some(value => typeof value === 'string' && /\$\{[^}]+\}/.test(value))) {
    throw new Error('暂不支持变量引用，请将 ${...} 替换为实际值后重试')
  }

  return {
    name: name || (config.name as string | undefined) || '',
    transport: type,
    url: type === 'http' ? config.url as string : '',
    command: type === 'stdio' ? config.command as string : '',
    args: JSON.stringify(args),
    credentials: Object.keys(credentials).length ? JSON.stringify(credentials) : '',
    confirmed: false,
  }
}
