/**
 * Pure view-model adapter for the inline Agent Activity Trace.
 *
 * This module intentionally has no React, DOM, page, API, or assistant-ui
 * imports: it consumes the synthetic `traceEvents` payload and returns plain
 * data that the presenter renders.
 */

export type ActivityStatus = 'running' | 'done' | 'error' | 'pending';

export type ActivityKind =
  | 'thought'
  | 'tool'
  | 'artifact'
  | 'approval'
  | 'review'
  | 'answer';

export interface ActivityEvent {
  type: string;
  content?: string;
  metadata?: Record<string, unknown>;
}

export interface ActivityLine {
  /** Stable for a given event array so the presenter can key lines. */
  id: string;
  kind: ActivityKind;
  status: ActivityStatus;
  title: string;
  description?: string;
  /** Present on tool lines so results can close the right call. */
  toolName?: string;
}

export interface AgentActivityModel {
  visible: boolean;
  summary: string;
  defaultExpanded: boolean;
  lines: ActivityLine[];
}

const TRACE_TOOL_NAME = 'traceEvents';

const TOOL_LABELS: Record<string, string> = {
  hiking_knowledge_search: '检索徒步知识库',
  gear_checklist: '检查装备清单',
  risk_assessment: '评估路线风险',
  sandbox_bash: '执行沙箱命令',
  generate_pdf: '生成计划书 PDF',
  memory_search: '检索长期记忆',
  memory_save: '保存长期记忆',
  sandbox_read: '读取工作区文件',
  sandbox_write: '写入工作区文件',
  web_search: '搜索网页',
  web_scraping: '抓取网页内容',
};

const RESOLVED_APPROVAL_STATUSES = new Set([
  'confirmed',
  'rejected',
  'already_resolved',
]);

const GENERIC_ARTIFACT_TITLE = '正在生成产物';
const GENERIC_REVIEW_TITLE = '需要人工核查';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function readText(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const text = value.trim();
  return text.length > 0 ? text : undefined;
}

function readMetadataText(
  metadata: Record<string, unknown> | undefined,
  key: string,
): string | undefined {
  return readText(metadata?.[key]);
}

function normalizeEvent(value: unknown): ActivityEvent | undefined {
  if (!isRecord(value)) return undefined;
  const type = readText(value.type);
  if (!type) return undefined;

  const event: ActivityEvent = { type };
  if (typeof value.content === 'string') event.content = value.content;
  if (isRecord(value.metadata)) event.metadata = value.metadata;
  return event;
}

/**
 * Finds the synthetic `traceEvents` tool-call part inside assistant message
 * content. Malformed or missing input yields an empty array; malformed events
 * inside the payload are dropped.
 */
export function readActivityEvents(
  content: readonly unknown[],
): ActivityEvent[] {
  if (!Array.isArray(content)) return [];

  for (const part of content) {
    if (!isRecord(part)) continue;

    const nested = isRecord(part.part) ? part.part : undefined;
    const toolName =
      part.toolName ?? part.tool_name ?? nested?.toolName ?? nested?.tool_name;
    if (toolName !== TRACE_TOOL_NAME) continue;

    const args = part.args ?? part.result ?? nested?.args ?? nested?.result;
    if (!Array.isArray(args)) return [];

    const events: ActivityEvent[] = [];
    for (const raw of args) {
      const event = normalizeEvent(raw);
      if (event) events.push(event);
    }
    return events;
  }

  return [];
}

function toolAction(tool: string): string {
  return TOOL_LABELS[tool] ?? `执行 ${tool}`;
}

function formatDuration(value: unknown): string | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined;
  const ms = Math.max(0, value);
  if (ms >= 1000) {
    return `${Number((ms / 1000).toFixed(1))}s`;
  }
  return `${Math.round(ms)}ms`;
}

function isUnresolvedApproval(metadata: Record<string, unknown> | undefined) {
  const status = readMetadataText(metadata, 'confirmation_status');
  return !status || !RESOLVED_APPROVAL_STATUSES.has(status);
}

function pushToolLine(
  lines: ActivityLine[],
  index: number,
  status: ActivityStatus,
  title: string,
  tool: string,
  description?: string,
) {
  const line: ActivityLine = {
    id: `event-${index}`,
    kind: 'tool',
    status,
    title,
    toolName: tool,
  };
  if (description) line.description = description;
  lines.push(line);
}

export function buildAgentActivity(input: {
  events: readonly ActivityEvent[];
  isRunning: boolean;
  hasText: boolean;
}): AgentActivityModel {
  const { events, isRunning, hasText } = input;
  const lines: ActivityLine[] = [];
  // Tracks whether the last meaningful event is an unresolved approval.
  let lastApprovalUnresolved = false;
  let hasRejectedApproval = false;

  events.forEach((event, index) => {
    const metadata = event.metadata;

    switch (event.type) {
      case 'tool_call': {
        const tool = readMetadataText(metadata, 'tool');
        if (!tool) return;
        lastApprovalUnresolved = false;

        const alreadyRunning = lines.some(
          line =>
            line.kind === 'tool' &&
            line.toolName === tool &&
            line.status === 'running',
        );
        if (alreadyRunning) return;

        lines.push({
          id: `event-${index}`,
          kind: 'tool',
          status: 'running',
          title: `正在${toolAction(tool)}`,
          toolName: tool,
        });
        return;
      }

      case 'tool_result': {
        const tool = readMetadataText(metadata, 'tool');
        const status = readMetadataText(metadata, 'status');
        if (!tool || !status) return;
        lastApprovalUnresolved = false;

        const succeeded = status === 'ok' || status === 'recovered';
        const denied = status === 'denied';
        const action = toolAction(tool);
        const title = succeeded
          ? `已${action}`
          : denied
            ? `${action}被拒绝`
            : `${action}失败`;
        const lineStatus: ActivityStatus = succeeded ? 'done' : 'error';
        const description = succeeded ? formatDuration(metadata?.duration_ms) : undefined;

        let runningIndex = -1;
        for (let i = lines.length - 1; i >= 0; i -= 1) {
          const candidate = lines[i];
          if (
            candidate &&
            candidate.kind === 'tool' &&
            candidate.toolName === tool &&
            candidate.status === 'running'
          ) {
            runningIndex = i;
            break;
          }
        }

        if (runningIndex >= 0) {
          const running = lines[runningIndex];
          if (running) {
            const merged: ActivityLine = {
              ...running,
              status: lineStatus,
              title,
            };
            if (description) merged.description = description;
            lines[runningIndex] = merged;
          }
          return;
        }

        // Kernel tools emit only a result; append a settled line directly.
        pushToolLine(lines, index, lineStatus, title, tool, description);
        return;
      }

      case 'artifact_progress': {
        lastApprovalUnresolved = false;
        lines.push({
          id: `event-${index}`,
          kind: 'artifact',
          status: 'running',
          title: readText(event.content) ?? GENERIC_ARTIFACT_TITLE,
        });
        return;
      }

      case 'review_notice': {
        lastApprovalUnresolved = false;
        lines.push({
          id: `event-${index}`,
          kind: 'review',
          status: 'error',
          title: readText(event.content) ?? GENERIC_REVIEW_TITLE,
        });
        return;
      }

      case 'approval_required': {
        const tool = readMetadataText(metadata, 'tool');
        const message =
          readMetadataText(metadata, 'message') ?? readText(event.content);
        lastApprovalUnresolved = isUnresolvedApproval(metadata);

        const line: ActivityLine = {
          id: `event-${index}`,
          kind: 'approval',
          status: 'pending',
          title: tool ? `等待确认：${toolAction(tool)}` : '等待你的确认',
        };
        if (tool) line.toolName = tool;
        if (message) line.description = message;
        lines.push(line);
        return;
      }

      case 'approval_resolved': {
        // The page forwards the approval.resolved SSE payload into the trace
        // so the wait line can settle without the presenter knowing about
        // approvals.
        const tool = readMetadataText(metadata, 'tool');
        const decision = readMetadataText(metadata, 'decision');
        const status = readMetadataText(metadata, 'status');
        const approved = decision === 'approve' || status === 'approved';
        const rejected = decision === 'reject' || status === 'rejected';
        if (!approved && !rejected) return;

        lastApprovalUnresolved = false;
        if (rejected) hasRejectedApproval = true;

        let pendingIndex = -1;
        for (let i = lines.length - 1; i >= 0; i -= 1) {
          const candidate = lines[i];
          if (
            !candidate ||
            candidate.kind !== 'approval' ||
            candidate.status !== 'pending'
          ) {
            continue;
          }
          if (tool && candidate.toolName && candidate.toolName !== tool) continue;
          pendingIndex = i;
          break;
        }
        if (pendingIndex < 0) return;

        const pending = lines[pendingIndex];
        if (!pending) return;

        const action = tool
          ? toolAction(tool)
          : pending.toolName
            ? toolAction(pending.toolName)
            : pending.title.replace(/^等待确认[:：]\s*/, '');
        const settled: ActivityLine = {
          ...pending,
          status: approved ? 'done' : 'error',
          title: `${approved ? '已确认' : '已拒绝'}：${action}`,
        };
        delete settled.description;
        const resolvedMessage = readMetadataText(metadata, 'message');
        if (resolvedMessage) settled.description = resolvedMessage;
        lines[pendingIndex] = settled;
        return;
      }

      default:
        // Unknown future event types must not break the UI.
        return;
    }
  });

  const hadEventLines = lines.length > 0;

  if (!isRunning) {
    // A completed run settles every still-running line. Errors and pending
    // approvals are deliberately kept as-is.
    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];
      if (line && line.status === 'running') {
        lines[i] = { ...line, status: 'done' };
      }
    }
  } else {
    const hasRunningLine = lines.some(line => line.status === 'running');
    if (hadEventLines) {
      if (!hasRunningLine && hasText) {
        lines.push({
          id: 'state-answer',
          kind: 'answer',
          status: 'running',
          title: '正在整理答案',
        });
      }
    } else if (hasText) {
      lines.push({
        id: 'state-answer',
        kind: 'answer',
        status: 'running',
        title: '正在整理答案',
      });
    } else {
      lines.push({
        id: 'state-thinking',
        kind: 'thought',
        status: 'running',
        title: '正在理解你的问题与上下文',
      });
    }
  }

  if (!isRunning && lines.length === 0) {
    return {
      visible: false,
      summary: '',
      defaultExpanded: false,
      lines: [],
    };
  }

  const approvalWait = !isRunning && lastApprovalUnresolved;
  const hasError = lines.some(line => line.status === 'error');
  const stepCount = lines.filter(
    line => line.kind === 'tool' || line.kind === 'artifact',
  ).length;

  let summary: string;
  if (approvalWait) {
    summary = '等待你的确认';
  } else if (hasRejectedApproval) {
    summary = '操作已拒绝';
  } else if (hasError) {
    summary = '有步骤执行失败';
  } else if (isRunning) {
    if (hasText) {
      summary = '正在生成回答';
    } else if (!hadEventLines) {
      summary = '正在思考';
    } else {
      summary = '正在处理';
    }
  } else {
    summary = `已完成 ${stepCount} 个步骤`;
  }

  return {
    visible: true,
    summary,
    defaultExpanded: isRunning || approvalWait || hasError || hasRejectedApproval,
    lines,
  };
}
