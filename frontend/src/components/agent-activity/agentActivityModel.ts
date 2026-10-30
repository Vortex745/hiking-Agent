/**
 * Pure view-model adapter for the inline Agent Activity Trace.
 *
 * This module intentionally has no React, DOM, page, API, or assistant-ui
 * imports: it consumes the synthetic `traceEvents` payload and returns plain
 * data that the presenter renders.
 */

export type ActivityStatus = 'running' | 'done' | 'error' | 'pending' | 'unknown';

export type ActivityKind =
  | 'thought'
  | 'model'
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
  /** Full reasoning text for thought lines; rendered as a paragraph. */
  detail?: string;
  /** Present on tool lines so results can close the right call. */
  toolName?: string;
  /** Present on approval lines so resolutions match the right approval. */
  approvalId?: string;
}

export interface AgentActivityModel {
  visible: boolean;
  /** Summary status; the presenter must not re-derive it from the lines. */
  status: ActivityStatus;
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
  focus_map: '地图定位',
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

/** Discovery/describe rows are capabilities, not execution steps. */
function isDiscoveryEvent(tool: string): boolean {
  return tool.endsWith('__discovery');
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

const MAX_ARG_LENGTH = 48;

function truncateText(value: string, limit = MAX_ARG_LENGTH): string {
  return value.length > limit ? `${value.slice(0, limit - 1)}…` : value;
}

function readScalar(value: unknown): string | undefined {
  if (typeof value === 'string') return readText(value);
  if (typeof value === 'number' && Number.isFinite(value)) return String(value);
  if (typeof value === 'boolean') return value ? '是' : '否';
  return undefined;
}

/**
 * One compact, human-readable line for the tool's own arguments. Only known
 * tools get bespoke wording; unknown tools fall back to their first scalars so
 * a new tool still shows something truthful instead of nothing.
 */
function describeToolArgs(tool: string, args: unknown): string | undefined {
  if (!isRecord(args)) return undefined;

  switch (tool) {
    case 'hiking_knowledge_search':
    case 'web_search':
    case 'memory_search': {
      const query = readText(args.query);
      return query ? `查询「${truncateText(query)}」` : undefined;
    }
    case 'web_scraping': {
      const url = readText(args.url);
      return url ? `抓取 ${truncateText(url, 56)}` : undefined;
    }
    case 'risk_assessment': {
      const target = readText(args.destination) ?? readText(args.route);
      const days = readScalar(args.days);
      if (target && days) return `${truncateText(target)} · ${days} 天`;
      return target ? `路线 ${truncateText(target)}` : undefined;
    }
    case 'gear_checklist': {
      const days = readScalar(args.days);
      const season = readText(args.season);
      const parts = [days ? `${days} 天` : undefined, season].filter(
        (part): part is string => Boolean(part),
      );
      return parts.length > 0 ? parts.join(' · ') : undefined;
    }
    case 'generate_pdf': {
      const filename = readText(args.filename);
      return filename ? truncateText(filename, 40) : undefined;
    }
    case 'sandbox_bash': {
      const command = readText(args.command);
      return command ? truncateText(command, 56) : undefined;
    }
    default: {
      const parts: string[] = [];
      for (const value of Object.values(args)) {
        const scalar = readScalar(value);
        if (scalar) parts.push(truncateText(scalar, 24));
        if (parts.length === 2) break;
      }
      return parts.length > 0 ? parts.join(' · ') : undefined;
    }
  }
}

/** Duration, evidence count and failure reason for a settled tool call. */
function describeToolResult(
  metadata: Record<string, unknown> | undefined,
): string | undefined {
  if (!metadata) return undefined;

  const parts: string[] = [];
  const duration = formatDuration(metadata.duration_ms);
  if (duration) parts.push(duration);

  const evidence = isRecord(metadata.evidence) ? metadata.evidence : undefined;
  const chunks = Array.isArray(evidence?.chunks) ? evidence.chunks.length : 0;
  if (chunks > 0) parts.push(`${chunks} 条`);

  const status = readMetadataText(metadata, 'status');
  if (status && status !== 'ok' && status !== 'recovered') {
    const reason =
      readMetadataText(metadata, 'reason') ?? readMetadataText(metadata, 'message');
    if (reason) parts.push(truncateText(reason, 36));
  }

  return parts.length > 0 ? parts.join(' · ') : undefined;
}

function joinDescriptions(
  left: string | undefined,
  right: string | undefined,
): string | undefined {
  if (!left) return right;
  if (!right || right === left) return left;
  if (left.includes(right)) return left;
  return `${left} · ${right}`;
}

/**
 * Returns the index of the first line matching `match`, searching forward
 * from the start (FIFO call pairing) or backward from the end (most recent
 * pending approval). Returns -1 when nothing matches.
 */
function findLineIndex(
  lines: readonly ActivityLine[],
  match: (line: ActivityLine) => boolean,
  direction: 'forward' | 'backward' = 'forward',
): number {
  if (direction === 'forward') {
    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];
      if (line && match(line)) return i;
    }
    return -1;
  }
  for (let i = lines.length - 1; i >= 0; i -= 1) {
    const line = lines[i];
    if (line && match(line)) return i;
  }
  return -1;
}

/** Reasoning ends as soon as the next real step starts. */
function settleThoughtLines(lines: ActivityLine[]) {
  for (const line of lines) {
    if (line.kind === 'thought' && line.status === 'running') {
      line.status = 'done';
    }
  }
}

function findRunningToolLine(
  lines: readonly ActivityLine[],
  tool: string,
): number {
  return findLineIndex(
    lines,
    line =>
      line.kind === 'tool' && line.toolName === tool && line.status === 'running',
  );
}

function findPendingApprovalLine(
  lines: readonly ActivityLine[],
  approvalId: string | undefined,
  tool: string | undefined,
): number {
  const isPendingApproval = (line: ActivityLine) =>
    line.kind === 'approval' && line.status === 'pending';

  if (approvalId) {
    const byId = findLineIndex(
      lines,
      line => isPendingApproval(line) && line.approvalId === approvalId,
      'backward',
    );
    if (byId >= 0) return byId;
  }

  if (tool) {
    const byTool = findLineIndex(
      lines,
      line => isPendingApproval(line) && line.toolName === tool,
      'backward',
    );
    if (byTool >= 0) return byTool;
  }

  return findLineIndex(lines, isPendingApproval, 'backward');
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
  let hasExpiredApproval = false;
  let hasUnknownResult = false;

  events.forEach((event, index) => {
    const metadata = event.metadata;
    // Any non-reasoning step means the previous reasoning segment is over.
    if (event.type !== 'thought') settleThoughtLines(lines);

    switch (event.type) {
      case 'thought': {
        const content =
          readText(event.content) ?? readMetadataText(metadata, 'content');
        if (!content) return;
        const previous = lines[lines.length - 1];
        if (previous && previous.kind === 'thought' && previous.status === 'running') {
          previous.detail = `${previous.detail ?? ''}${content}`;
          return;
        }
        lines.push({
          id: `event-${index}`,
          kind: 'thought',
          status: 'running',
          title: '思考',
          detail: content,
        });
        return;
      }

      case 'model_round': {
        // Agent process tracking ticket 03: one main-model round. The start
        // event opens the line; the finish event settles it with its
        // duration. ``step`` identifies the round so parallel or repeated
        // rounds cannot settle someone else's line.
        lastApprovalUnresolved = false;
        const state = readMetadataText(metadata, 'state');
        if (state === 'finished') {
          const runningIndex = findLineIndex(
            lines,
            line => line.kind === 'model' && line.status === 'running',
            'backward',
          );
          if (runningIndex < 0) return;
          const running = lines[runningIndex];
          if (!running) return;
          const settled: ActivityLine = { ...running, status: 'done' };
          const duration = formatDuration(metadata?.duration_ms);
          if (duration) settled.description = duration;
          lines[runningIndex] = settled;
          return;
        }
        const label =
          readText(event.content) ??
          readMetadataText(metadata, 'message') ??
          '模型处理中';
        lines.push({
          id: `event-${index}`,
          kind: 'model',
          status: 'running',
          title: label,
        });
        return;
      }

      case 'tool_call': {
        const tool = readMetadataText(metadata, 'tool');
        if (!tool || isDiscoveryEvent(tool)) return;
        lastApprovalUnresolved = false;

        // Every call creates a line, including parallel calls to the same
        // tool. Results pair with the earliest still-running call (FIFO).
        const callLine: ActivityLine = {
          id: `event-${index}`,
          kind: 'tool',
          status: 'running',
          title: `正在${toolAction(tool)}`,
          toolName: tool,
        };
        const callDetail = describeToolArgs(tool, metadata?.args);
        if (callDetail) callLine.description = callDetail;
        lines.push(callLine);
        return;
      }

      case 'tool_result': {
        const tool = readMetadataText(metadata, 'tool');
        const status = readMetadataText(metadata, 'status');
        if (!tool || !status) return;
        // Discovery describes capabilities; ``authorized`` records a
        // permission decision. Neither is an execution result or a failure.
        if (isDiscoveryEvent(tool) || status === 'authorized') return;
        lastApprovalUnresolved = false;

        const unknown = status === 'outcome_unknown';
        const succeeded = status === 'ok' || status === 'recovered';
        const denied = status === 'denied';
        const action = toolAction(tool);
        const title = unknown
          ? '本次调用结果未确认'
          : succeeded
            ? `已${action}`
            : denied
              ? `${action}被拒绝`
              : `${action}失败`;
        const lineStatus: ActivityStatus = unknown
          ? 'unknown'
          : succeeded
            ? 'done'
            : 'error';
        if (unknown) hasUnknownResult = true;
        const description = describeToolResult(metadata);

        const runningIndex = findRunningToolLine(lines, tool);

        if (runningIndex >= 0) {
          const running = lines[runningIndex];
          if (running) {
            const merged: ActivityLine = {
              ...running,
              status: lineStatus,
              title,
            };
            const detail = joinDescriptions(running.description, description);
            if (detail) merged.description = detail;
            else delete merged.description;
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
        const approvalId = readMetadataText(metadata, 'approval_id');
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
        if (approvalId) line.approvalId = approvalId;
        if (message) line.description = message;
        lines.push(line);
        return;
      }

      case 'approval_resolved': {
        // The page forwards the approval.resolved SSE payload into the trace
        // so the wait line can settle without the presenter knowing about
        // approvals.
        const tool = readMetadataText(metadata, 'tool');
        const approvalId = readMetadataText(metadata, 'approval_id');
        const decision = readMetadataText(metadata, 'decision');
        const status = readMetadataText(metadata, 'status');
        const expired = decision === 'expired' || status === 'expired';
        const approved =
          !expired && (decision === 'approve' || status === 'approved');
        const rejected =
          !expired && (decision === 'reject' || status === 'rejected');
        if (!approved && !rejected && !expired) return;

        lastApprovalUnresolved = false;
        if (rejected) hasRejectedApproval = true;
        if (expired) hasExpiredApproval = true;

        const pendingIndex = findPendingApprovalLine(lines, approvalId, tool);
        if (pendingIndex < 0) return;

        const pending = lines[pendingIndex];
        if (!pending) return;

        const action = tool
          ? toolAction(tool)
          : pending.toolName
            ? toolAction(pending.toolName)
            : pending.title.replace(/^等待确认[:：]\s*/, '');
        const prefix = approved ? '已确认' : expired ? '已过期' : '已拒绝';
        const settled: ActivityLine = {
          ...pending,
          status: approved ? 'done' : 'error',
          title: `${prefix}：${action}`,
        };
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
  // An unresolved approval wins even while `isStreaming` keeps the message
  // running: the run is paused waiting for the user, not generating.
  const approvalWait = lastApprovalUnresolved;

  // The answer has started (or the run ended), so no reasoning is still live.
  if (!isRunning || hasText) settleThoughtLines(lines);

  if (!isRunning) {
    // A completed run settles every still-running line. Errors and pending
    // approvals are deliberately kept as-is.
    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];
      if (line && line.status === 'running') {
        lines[i] = { ...line, status: 'done' };
      }
    }
  } else if (!approvalWait) {
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

  if (!isRunning && hasText && lines.length === 0) {
    // Keep a thinking record after the answer arrives: a plain chat run has
    // no tool events, so without this line the trace disappears entirely
    // once streaming ends.
    lines.push({
      id: 'state-thought-done',
      kind: 'thought',
      status: 'done',
      title: '已理解你的问题并生成回答',
    });
  }

  if (!isRunning && lines.length === 0) {
    return {
      visible: false,
      status: 'done',
      summary: '',
      defaultExpanded: false,
      lines: [],
    };
  }

  const hasError = lines.some(line => line.status === 'error');
  const hasUnknown = hasUnknownResult || lines.some(line => line.status === 'unknown');
  // A model round is a step too: the count answers "how much actually ran".
  const stepCount = lines.filter(
    line => line.kind === 'tool' || line.kind === 'artifact' || line.kind === 'model',
  ).length;

  const status: ActivityStatus = approvalWait
    ? 'pending'
    : hasRejectedApproval || hasExpiredApproval || hasError
      ? 'error'
      : !isRunning && hasUnknown
        ? 'unknown'
        : isRunning
          ? 'running'
          : 'done';

  let summary: string;
  if (approvalWait) {
    summary = '等待你的确认';
  } else if (hasRejectedApproval) {
    summary = '操作已拒绝';
  } else if (hasExpiredApproval) {
    summary = '确认已过期';
  } else if (hasError) {
    summary = hasUnknown ? '有调用失败或未确认' : '有步骤执行失败';
  } else if (hasUnknown) {
    summary = '有调用结果未确认';
  } else if (isRunning) {
    // The collapsed summary IS the current step: whatever real line is
    // executing right now (a tool, a model round) names itself, so the
    // reader never sees a generic "正在处理". Synthetic placeholder lines
    // carry no step information and are skipped.
    const runningLine = lines.find(
      line => line.status === 'running' && !line.id.startsWith('state-'),
    )
    if (runningLine) {
      summary = runningLine.title
    } else if (hasText) {
      summary = '正在生成回答'
    } else if (!hadEventLines) {
      summary = '正在思考'
    } else {
      summary = '正在处理'
    }
  } else {
    summary = stepCount > 0 ? `已完成 ${stepCount} 个步骤` : '思考过程';
  }

  return {
    visible: true,
    status,
    summary,
    defaultExpanded:
      isRunning || approvalWait || hasRejectedApproval || hasExpiredApproval || hasError || hasUnknown,
    lines,
  };
}
