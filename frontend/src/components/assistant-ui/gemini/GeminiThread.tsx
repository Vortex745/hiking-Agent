import {
  AuiIf,
  ComposerPrimitive,
  ThreadPrimitive,
  MessagePrimitive,
  useMessage,
} from "@assistant-ui/react";
import { ArrowUp, RotateCw, Copy, Check, X, MoreHorizontal, Pin, PinOff, GripHorizontal, Paperclip, Download } from 'lucide-react';
import { getConversationMemoryProgress } from '../../../api/conversationMemory';
import { useRef, useState, createContext, useContext, useCallback, useEffect, type RefObject, type PointerEvent as ReactPointerEvent } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(useGSAP);

interface GeminiThreadProps {
  emptyTitle: string;
  /** Called when user clicks "regenerate" — parent should resend the last user prompt */
  onRegenerate?: () => void;
  onUploadClick?: () => void;
  onConfirmTool?: (confirmationId: string, action: 'confirm' | 'reject') => void;
  /** Real message count from the parent page's state (for accurate memory meter) */
  realMessageCount?: number;
}

/** Context to pass callbacks down to ChatMessage without prop drilling */
const ThreadActionsContext = createContext<{
  onRegenerate?: () => void;
  onConfirmTool?: (confirmationId: string, action: 'confirm' | 'reject') => void;
  realMessageCount: number;
}>({ realMessageCount: 0 });

export const GeminiThread = ({ emptyTitle, onRegenerate, onUploadClick, onConfirmTool, realMessageCount = 0 }: GeminiThreadProps) => (
  <ThreadActionsContext.Provider value={{ onRegenerate, onConfirmTool, realMessageCount }}>
    <ThreadPrimitive.Root className="h-full bg-[#fdfcfc] dark:bg-[#131314] flex flex-col overflow-hidden w-full relative">
      <AuiIf condition={(s) => s.thread.isEmpty}>
        <EmptyState emptyTitle={emptyTitle} onUploadClick={onUploadClick} />
      </AuiIf>
      <AuiIf condition={(s) => !s.thread.isEmpty}>
        <div className="flex-1 overflow-hidden flex flex-col relative w-full">
          <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto px-4 md:px-8 pt-24 pb-32 scrollbar-thin">
            <ThreadPrimitive.Messages components={{ Message: ChatMessage }} />
          </ThreadPrimitive.Viewport>
          <div className="absolute bottom-0 left-0 right-0 pt-10 pb-4 bg-gradient-to-t from-[#fdfcfc] via-[#fdfcfc] dark:from-[#131314] dark:via-[#131314] to-transparent pointer-events-none">
            <div className="pointer-events-auto">
              <Composer onUploadClick={onUploadClick} />
            </div>
          </div>
        </div>
      </AuiIf>
    </ThreadPrimitive.Root>
  </ThreadActionsContext.Provider>
);

const EmptyState = ({ emptyTitle, onUploadClick }: { emptyTitle: string; onUploadClick?: () => void }) => (
  <div className="relative flex grow h-full items-center justify-center">
    <div
      aria-hidden="true"
      className="pointer-events-none absolute top-1/2 left-1/2 h-[330px] w-[720px] -translate-x-1/2 -translate-y-1/2 bg-[radial-gradient(closest-side,#a9d1fb,transparent)] opacity-70 blur-[55px] dark:bg-[radial-gradient(closest-side,#1d4068,transparent)] dark:opacity-65"
    />
    <div className="relative z-10 w-full max-w-[800px] px-6 flex flex-col items-center">
      <h1 className="text-center text-3xl font-medium mb-12 text-[#1f1f1f] dark:text-[#e3e3e3]">{emptyTitle}</h1>
      <Composer onUploadClick={onUploadClick} />
    </div>
  </div>
);

const Composer = ({ onUploadClick }: { onUploadClick?: () => void }) => (
  <ComposerPrimitive.Root className="flex flex-col rounded-3xl bg-white p-2.5 shadow-[0_2px_10px_-2px_rgba(0,0,0,0.18)] dark:bg-[#1e1f20] w-[calc(100%-2rem)] md:w-full max-w-[800px] mx-auto border border-black/5 dark:border-white/5">
    <div className="flex items-end gap-2 px-2">
      {onUploadClick && (
        <button
          type="button"
          onClick={onUploadClick}
          className="text-[#444746] hover:bg-black/5 hover:text-[#1f1f1f] dark:text-[#c4c7c5] dark:hover:bg-white/5 dark:hover:text-[#e3e3e3] p-2 rounded-full transition-colors flex shrink-0 h-10 w-10 items-center justify-center press-scale"
          title="上传文档"
          aria-label="上传文档"
        >
          <Paperclip className="w-[18px] h-[18px]" strokeWidth={2} />
        </button>
      )}
      <ComposerPrimitive.Input
        placeholder="输入您的问题..."
        aria-label="输入徒步问题"
        name="message"
        className="flex-1 resize-none bg-transparent text-[16px] outline-none border-none py-1.5 min-h-[32px] text-[#1f1f1f] dark:text-[#e3e3e3] placeholder:text-[#444746] dark:placeholder:text-[#c4c7c5]"
      />
      
      {/* Send: disabled grey when empty, blue when there is text */}
      <AuiIf condition={(s) => !s.thread.isRunning}>
        <ComposerPrimitive.Send
          aria-label="发送消息"
          title="发送消息"
          className="bg-[#d3e3fd] text-[#062e6f] disabled:bg-[#e8eaed] disabled:text-[#1f1f1f]/40 dark:disabled:bg-[#333537] dark:disabled:text-[#c4c7c5]/40 p-2 rounded-full transition-colors flex shrink-0 h-10 w-10 items-center justify-center press-scale"
        >
          <ArrowUp className="w-[18px] h-[18px]" strokeWidth={2.5} aria-hidden="true" />
        </ComposerPrimitive.Send>
      </AuiIf>
      
      {/* Cancel: stop square while a response streams */}
      <AuiIf condition={(s) => s.thread.isRunning}>
        <ComposerPrimitive.Cancel
          aria-label="停止生成"
          title="停止生成"
          className="bg-[#d3e3fd] text-[#062e6f] p-2 rounded-full transition-colors flex shrink-0 h-10 w-10 items-center justify-center press-scale"
        >
          <span className="size-3 rounded-[3px] bg-current" aria-hidden="true" />
        </ComposerPrimitive.Cancel>
      </AuiIf>
    </div>
  </ComposerPrimitive.Root>
);

const sourceLabel = (source: string) => {
  if (source === 'cloud_docs') return '云知识库'
  if (source === 'upload') return '上传文档'
  if (source === 'unknown') return '未知来源'
  return source
}

const formatArtifactSize = (value: unknown) => {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) return ''
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / 1024 / 1024).toFixed(1)} MB`
}

const MARKDOWN_PATTERN = /(!\[[^\]]*\]\([^)]+\)|\[[^\]]*\]\([^)]+\)|\*\*(?=\S)[\s\S]*?\S\*\*)/g;
const IMAGE_URL_PATTERN = /\.(jpg|jpeg|png|gif|webp|avif|bmp)(\?.*)?$/i;
type PreviewImage = { src: string; alt: string };

const MarkdownText = (part: any) => {
  const [previewImage, setPreviewImage] = useState<PreviewImage | null>(null);
  const raw = part.text || part.part?.text || '';
  // Split on bold, images, and links patterns
  const segments = raw.split(MARKDOWN_PATTERN);

  useEffect(() => {
    if (!previewImage) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setPreviewImage(null);
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [previewImage]);

  return (
    <div className="whitespace-pre-wrap leading-relaxed relative">
      {segments.map((seg: string, i: number) => {
        if (!seg) return null;
        
        const imgMatch = seg.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
        if (imgMatch) {
          const alt = imgMatch[1] || '图片';
          const src = imgMatch[2];
          return (
            <button
              key={i}
              type="button"
              onClick={() => setPreviewImage({ src, alt })}
              className="my-2 block max-w-full cursor-zoom-in overflow-hidden rounded-xl border border-black/5 shadow-sm transition-opacity hover:opacity-95 focus:outline-none focus:ring-2 focus:ring-[#0b57d0]/60 dark:border-white/5"
              aria-label={`放大图片：${alt}`}
              title="放大图片"
            >
              <img 
                src={src} 
                alt={alt} 
                className="block max-w-full max-h-[350px] object-cover" 
                loading="lazy"
              />
            </button>
          );
        }
        
        const linkMatch = seg.match(/^\[([^\]]*)\]\(([^)]+)\)$/);
        if (linkMatch) {
          const linkText = linkMatch[1];
          const linkUrl = linkMatch[2];
          // Auto-detect image URLs and render as thumbnails
          if (IMAGE_URL_PATTERN.test(linkUrl)) {
            return (
              <button
                key={i}
                type="button"
                onClick={() => setPreviewImage({ src: linkUrl, alt: linkText || '图片' })}
                className="my-2 block max-w-full cursor-zoom-in overflow-hidden rounded-xl border border-black/5 shadow-sm transition-opacity hover:opacity-95 focus:outline-none focus:ring-2 focus:ring-[#0b57d0]/60 dark:border-white/5"
                aria-label={`放大图片：${linkText || '图片'}`}
                title="放大图片"
              >
                <img
                  src={linkUrl}
                  alt={linkText || '图片'}
                  className="block max-w-full max-h-[350px] object-cover"
                  loading="lazy"
                />
              </button>
            );
          }
          return (
            <a
              key={i}
              href={linkUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#0b57d0] hover:underline dark:text-[#a8c7fa]"
            >
              {linkText}
            </a>
          );
        }

        if (seg.startsWith('**') && seg.endsWith('**')) {
          return <strong key={i} className="font-bold">{seg.slice(2, -2)}</strong>;
        }
        
        return <span key={i}>{seg}</span>;
      })}
      {previewImage && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-label={`图片预览：${previewImage.alt}`}
          onClick={() => setPreviewImage(null)}
        >
          <button
            type="button"
            className="absolute right-4 top-4 flex h-10 w-10 items-center justify-center rounded-full bg-black/60 text-white transition-colors hover:bg-black/80 focus:outline-none focus:ring-2 focus:ring-white"
            onClick={(event) => {
              event.stopPropagation();
              setPreviewImage(null);
            }}
            aria-label="关闭图片预览"
            title="关闭图片预览"
          >
            <X className="size-5" aria-hidden="true" />
          </button>
          <img
            src={previewImage.src}
            alt={previewImage.alt}
            className="max-h-[90vh] max-w-[92vw] rounded-xl object-contain shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
};

/** Resolve a relative download_url against the configured API base URL */
function resolveDownloadUrl(downloadUrl: string): string {
  if (!downloadUrl) return '';
  // Already a full URL — return as-is
  if (downloadUrl.startsWith('http://') || downloadUrl.startsWith('https://')) return downloadUrl;

  // Check for a configured API base URL (e.g. https://api.example.com/api/v1)
  const configuredApiBase = import.meta.env?.VITE_API_BASE_URL?.trim().replace(/\/+$/, '');

  if (configuredApiBase) {
    // download_url is like /api/v1/artifacts/xxx.pdf
    // Replace the /api/v1 prefix with the configured full URL
    const apiBase = '/api/v1';
    if (downloadUrl.startsWith(apiBase)) {
      return configuredApiBase + downloadUrl.slice(apiBase.length);
    }
    return configuredApiBase + (downloadUrl.startsWith('/') ? '' : '/') + downloadUrl;
  }

  // No VITE_API_BASE_URL configured — relative path works via same-origin Gateway
  return downloadUrl;
}

const ArtifactList = ({ artifacts }: { artifacts: any[] }) => {
  const [downloading, setDownloading] = useState<string | null>(null);

  const handleDownload = async (rawUrl: string, filename: string) => {
    const url = resolveDownloadUrl(rawUrl);
    setDownloading(filename);
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`下载失败：${response.status} ${response.statusText}`);
      }
      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(blobUrl);
    } catch (error) {
      alert(error instanceof Error ? error.message : '下载失败，请稍后重试');
    } finally {
      setDownloading(null);
    }
  };

  return (
    <div className="mt-3 border-t border-border/40 pt-3 space-y-2">
      {artifacts.map((artifact, i) => {
        const metadata = artifact?.metadata || {};
        const downloadUrl = typeof metadata.download_url === 'string' ? metadata.download_url : '';
        const filename = typeof metadata.filename === 'string' ? metadata.filename : '下载文件';
        const size = formatArtifactSize(metadata.size_bytes);
        const isDownloading = downloading === filename;
        return (
          <div key={i} className="flex flex-wrap items-center gap-2 text-[13px] text-text-secondary">
            {downloadUrl ? (
              <button
                type="button"
                onClick={() => handleDownload(downloadUrl, filename)}
                disabled={isDownloading}
                className="inline-flex min-h-8 items-center gap-1.5 rounded-md border border-border/70 bg-background px-2.5 py-1.5 font-medium text-text-primary transition-colors hover:border-primary/60 hover:text-primary disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Download className={`size-4${isDownloading ? ' animate-pulse' : ''}`} aria-hidden="true" />
                <span>{isDownloading ? '下载中…' : `下载 ${filename}`}</span>
              </button>
            ) : (
              <span className="leading-relaxed">{artifact.content}</span>
            )}
            {downloadUrl && size && <span className="text-[12px] opacity-70">{size}</span>}
          </div>
        )
      })}
    </div>
  );
}

const CustomToolCall = (part: any) => {
  const { onConfirmTool } = useContext(ThreadActionsContext);
  const toolName = part.toolName || part.part?.toolName;
  const args = part.args || part.part?.args;
  
  if (toolName === "processSteps") {
    return null;
  }
  
  if (toolName === "traceEvents") {
    const traceEvents = args as any[];
    if (!traceEvents || traceEvents.length === 0) return null;
    return (
      <details className="mt-4 border-t border-border/40 pt-3 text-[13px] text-text-secondary">
        <summary className="cursor-pointer select-none font-medium text-text-primary outline-none">
          思考流程信息 · {traceEvents.length} 个步骤
        </summary>
        <div className="mt-3 max-h-[300px] overflow-y-auto space-y-3 opacity-90 pr-2 scrollbar-thin">
          {traceEvents.map((event: any, idx: number) => {
            const confirmationId = typeof event?.metadata?.confirmation_id === 'string'
              ? event.metadata.confirmation_id
              : '';
            const confirmationStatus = typeof event?.metadata?.confirmation_status === 'string'
              ? event.metadata.confirmation_status
              : '';
            const confirmationResolved = ['confirmed', 'rejected', 'already_resolved'].includes(confirmationStatus);
            const confirmationBusy = ['confirming', 'rejecting'].includes(confirmationStatus);
            return (
              <div key={idx} className="border-l-2 border-primary/25 pl-3">
                <div className="font-medium text-text-primary">
                  {idx + 1}. {traceLabels[event?.type] || event?.type || '事件'}
                </div>
                {event?.content && (
                  <div className="mt-1.5 whitespace-pre-wrap text-text-secondary">
                    {event.content}
                  </div>
                )}
                {event?.type === 'approval_required' && event?.metadata?.confirmation_id && confirmationId && (
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      title="确认执行工具"
                      aria-label="确认执行工具"
                      disabled={!onConfirmTool || confirmationResolved || confirmationBusy}
                      onClick={() => onConfirmTool?.(confirmationId, 'confirm')}
                      className="inline-flex min-h-8 items-center gap-1.5 rounded-md bg-[#0b57d0] px-2.5 py-1.5 text-[12px] font-medium text-white transition-colors hover:bg-[#0842a0] disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <Check className="size-3.5" aria-hidden="true" />
                      <span>确认</span>
                    </button>
                    <button
                      type="button"
                      title="拒绝执行工具"
                      aria-label="拒绝执行工具"
                      disabled={!onConfirmTool || confirmationResolved || confirmationBusy}
                      onClick={() => onConfirmTool?.(confirmationId, 'reject')}
                      className="inline-flex min-h-8 items-center gap-1.5 rounded-md border border-border/70 bg-background px-2.5 py-1.5 text-[12px] font-medium text-text-primary transition-colors hover:border-red-500/60 hover:text-red-600 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <X className="size-3.5" aria-hidden="true" />
                      <span>拒绝</span>
                    </button>
                    {confirmationStatus && (
                      <span className="text-[12px] text-text-secondary">{confirmationStatus}</span>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </details>
    );
  }

  if (toolName === "artifacts") {
    return null;
  }

  if (part.toolName === "searchSummary") {
    const summary = part.args;
    if (!summary || summary.documents.length === 0) return null;
    return (
      <details className="mt-4 border-t border-border/40 pt-3 text-[13px] text-text-secondary">
        <summary className="cursor-pointer select-none font-medium text-text-primary outline-none">
          文档搜索 · {summary.searchedCount} 篇文档 / {summary.matchedChunks} 个片段
        </summary>
        <div className="mt-3 space-y-3 opacity-90">
          {summary.documents.map((doc: any, idx: number) => (
            <div key={`${doc.title}-${idx}`} className="border-l-2 border-primary/25 pl-3">
              <div className="font-medium text-text-primary">
                {idx + 1}. {doc.title}
              </div>
              <div className="text-[12px] mt-0.5 opacity-70">
                来源：{sourceLabel(doc.source)}
                {doc.chunks ? ` · ${doc.chunks} 个片段` : ''}
              </div>
              {doc.content && (
                <div className="mt-1.5 whitespace-pre-wrap text-text-secondary">
                  {doc.content}
                </div>
              )}
            </div>
          ))}
        </div>
      </details>
    );
  }

  return null;
};

type LifecycleStep = {
  title: string;
  detail: string;
  state: 'complete' | 'active' | 'pending';
};

type LifecyclePosition = {
  x: number;
  y: number;
};

const LIFECYCLE_PANEL_WIDTH = 330;
const LIFECYCLE_PANEL_HEIGHT = 420;
const LIFECYCLE_PANEL_MARGIN = 12;

function prefersReducedMotion() {
  return typeof window !== 'undefined'
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

const traceLabels: Record<string, string> = {
  thought: '思考',
  process: '流程',
  documents: '文档',
  tool_call: '工具调用',
  tool_result: '工具结果',
  approval_required: '需要确认',
  artifact: '产物',
  text: '回答',
  done: '完成',
  error: '错误',
};

function getPartToolName(part: any) {
  return part?.toolName || part?.part?.toolName || part?.tool_name || '';
}

function getPartArgs(part: any) {
  return part?.args ?? part?.part?.args ?? part?.result ?? null;
}

function getToolArgs(content: readonly any[], toolName: string) {
  const part = content.find(item => getPartToolName(item) === toolName);
  return part ? getPartArgs(part) : null;
}

function getMessageText(content: readonly any[]) {
  return content
    .filter(part => part?.type === 'text' || typeof part?.text === 'string')
    .map(part => part.text || part.part?.text || '')
    .join('');
}

function stepState(done: boolean, active: boolean): LifecycleStep['state'] {
  if (done) return 'complete';
  if (active) return 'active';
  return 'pending';
}

function getAgentLifecycle(events: any[], isRunning: boolean, hasText: boolean): LifecycleStep[] {
  const phases = new Set(events.map(event => event?.metadata?.phase).filter(Boolean));
  const hasToolEvent = events.some(event => ['tool_call', 'tool_result', 'approval_required'].includes(event?.type));

  return [
    {
      title: '请求理解',
      detail: '识别徒步意图、场景和缺失信息',
      state: stepState(events.length > 0, isRunning && events.length === 0),
    },
    {
      title: '记忆读取',
      detail: '读取会话历史、摘要和长期记忆',
      state: stepState(phases.has('memory'), isRunning && events.length > 0 && !phases.has('memory')),
    },
    {
      title: '查询改写',
      detail: '把用户问题整理成更适合工具执行的任务',
      state: stepState(phases.has('query_rewrite'), isRunning && phases.has('memory') && !phases.has('query_rewrite')),
    },
    {
      title: '工具编排',
      detail: '按风险等级选择并执行徒步工具',
      state: stepState(hasToolEvent, isRunning && phases.has('start') && !hasToolEvent && !hasText),
    },
    {
      title: '回答生成',
      detail: '整合工具结果并流式生成回复',
      state: stepState(hasText, isRunning && !hasText),
    },
    {
      title: '完成收尾',
      detail: '结束 SSE、提交可沉淀的会话记忆',
      state: stepState(!isRunning && hasText, false),
    },
  ];
}

function getRagLifecycle(processSteps: string[], hasSearchSummary: boolean, isRunning: boolean, hasText: boolean): LifecycleStep[] {
  const hasProcess = processSteps.length > 0;

  return [
    {
      title: '接收问题',
      detail: '构建 RAG 查询载荷并打开 SSE 连接',
      state: 'complete',
    },
    {
      title: '查询理解',
      detail: '进行简单问题识别或查询改写',
      state: stepState(hasProcess || hasText || hasSearchSummary, isRunning && !hasProcess && !hasText),
    },
    {
      title: '知识检索',
      detail: '向量检索、BM25 召回和候选片段融合',
      state: stepState(hasSearchSummary, isRunning && hasProcess && !hasSearchSummary && !hasText),
    },
    {
      title: '上下文组装',
      detail: '筛选命中文档并准备回答上下文',
      state: stepState(hasSearchSummary || hasText, isRunning && hasSearchSummary && !hasText),
    },
    {
      title: '回答生成',
      detail: '根据上下文流式输出最终回复',
      state: stepState(hasText, isRunning && !hasText),
    },
    {
      title: '完成收尾',
      detail: '发送 done 事件并关闭本次流',
      state: stepState(!isRunning && hasText, false),
    },
  ];
}

function getGenericLifecycle(isRunning: boolean, hasText: boolean): LifecycleStep[] {
  return [
    {
      title: '建立连接',
      detail: '向后端发送请求并等待首个 SSE 事件',
      state: stepState(hasText, isRunning && !hasText),
    },
    {
      title: '回答生成',
      detail: '接收 text 事件并增量更新消息',
      state: stepState(hasText, isRunning && !hasText),
    },
    {
      title: '完成收尾',
      detail: '接收 done 事件后解除运行状态',
      state: stepState(!isRunning && hasText, false),
    },
  ];
}

function buildLifecycle(content: readonly any[], isRunning: boolean) {
  const text = getMessageText(content).trim();
  const hasText = text.length > 0;
  const traceEvents = (getToolArgs(content, 'traceEvents') || []) as any[];
  const processSteps = (getToolArgs(content, 'processSteps') || []) as string[];
  const searchSummary = getToolArgs(content, 'searchSummary') as any;

  if (traceEvents.length > 0) {
    return {
      title: 'Agent 技术生命周期',
      steps: getAgentLifecycle(traceEvents, isRunning, hasText),
      recentEvents: traceEvents.map(event => ({
        label: traceLabels[event?.type] || event?.type || '事件',
        content: String(event?.content || '').trim(),
      })).filter(event => event.content),
    };
  }

  if (processSteps.length > 0 || searchSummary) {
    const documentCount = typeof searchSummary?.searchedCount === 'number'
      ? `命中 ${searchSummary.searchedCount} 篇文档 / ${searchSummary.matchedChunks || 0} 个片段`
      : '';
    return {
      title: 'RAG 技术生命周期',
      steps: getRagLifecycle(processSteps, Boolean(searchSummary), isRunning, hasText),
      recentEvents: [
        ...processSteps.slice(-3).map(step => ({ label: '流程', content: step })),
        ...(documentCount ? [{ label: '检索', content: documentCount }] : []),
      ],
    };
  }

  return {
    title: '流式回复生命周期',
    steps: getGenericLifecycle(isRunning, hasText),
    recentEvents: [],
  };
}

function clampLifecyclePosition(position: LifecyclePosition): LifecyclePosition {
  if (typeof window === 'undefined') return position;

  const maxX = Math.max(
    LIFECYCLE_PANEL_MARGIN,
    window.innerWidth - LIFECYCLE_PANEL_WIDTH - LIFECYCLE_PANEL_MARGIN,
  );
  const maxY = Math.max(
    LIFECYCLE_PANEL_MARGIN,
    window.innerHeight - LIFECYCLE_PANEL_HEIGHT - LIFECYCLE_PANEL_MARGIN,
  );

  return {
    x: Math.min(Math.max(position.x, LIFECYCLE_PANEL_MARGIN), maxX),
    y: Math.min(Math.max(position.y, LIFECYCLE_PANEL_MARGIN), maxY),
  };
}

function AiThinking({ text = '模型思考中' }: { text?: string }) {
  return (
    <div className="ai-thinking mt-1" role="status" aria-label="AI 正在思考">
      <span className="ai-thinking-label">
        <span key={text} className="font-medium bg-gradient-to-r from-[#a3a3a3] via-[#111] to-[#a3a3a3] dark:from-[#5f6368] dark:via-[#ffffff] dark:to-[#5f6368] bg-[length:200%_auto] bg-clip-text text-transparent animate-shimmer">
          {text}
        </span>
      </span>
    </div>
  );
}

const LifecycleHoverPanel = ({
  content,
  isRunning,
  pinned,
  position,
  panelRef,
  onPinToggle,
  onDragStart,
}: {
  content: readonly any[];
  isRunning: boolean;
  pinned: boolean;
  position: LifecyclePosition;
  panelRef: RefObject<HTMLDivElement>;
  onPinToggle: () => void;
  onDragStart: (event: ReactPointerEvent<HTMLDivElement>) => void;
}) => {
  const lifecycle = buildLifecycle(content, isRunning);
  const [activeItem, setActiveItem] = useState<{type: 'step' | 'event', index: number} | null>(null);
  const [sliderIndex, setSliderIndex] = useState(lifecycle.recentEvents.length - 1);
  const [userDragging, setUserDragging] = useState(false);
  const prevEventCountRef = useRef(lifecycle.recentEvents.length);

  // Auto-advance slider to latest event when new events arrive (unless user is dragging)
  useEffect(() => {
    const count = lifecycle.recentEvents.length;
    if (count > prevEventCountRef.current && !userDragging) {
      setSliderIndex(count - 1);
    }
    prevEventCountRef.current = count;
  }, [lifecycle.recentEvents.length, userDragging]);

  return (
    <div
      ref={panelRef}
      className={`z-50 flex w-[240px] flex-col overflow-visible rounded-2xl border border-black/10 bg-white/95 p-4 text-left shadow-[0_16px_50px_rgba(0,0,0,0.18)] backdrop-blur-xl dark:border-white/10 dark:bg-[#1e1f20]/95 ${
        pinned
          ? 'fixed left-0 top-0 max-h-[72vh] will-change-transform'
          : 'absolute bottom-full left-1/2 mb-2 -translate-x-1/2'
      }`}
      style={pinned ? { transform: `translate3d(${position.x}px, ${position.y}px, 0)` } : undefined}
      data-lifecycle-panel={pinned ? 'pinned' : 'hover'}
    >
      <div
        className={`mb-3 flex shrink-0 items-center justify-between gap-3 ${pinned ? 'cursor-grab active:cursor-grabbing' : ''}`}
        onPointerDown={onDragStart}
      >
        <div className="min-w-0 text-[13px] font-medium text-[#1f1f1f] dark:text-[#e3e3e3]">
          {lifecycle.title}
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {pinned && <GripHorizontal className="h-3.5 w-3.5 text-[#5f6368] dark:text-[#bdc1c6]" strokeWidth={1.7} />}
          <button
            type="button"
            className="rounded-full p-1 text-[#5f6368] transition-colors hover:bg-black/5 hover:text-[#1f1f1f] dark:text-[#bdc1c6] dark:hover:bg-white/10 dark:hover:text-[#e3e3e3]"
            aria-pressed={pinned}
            aria-label={pinned ? '取消钉住流程' : '钉住流程'}
            title={pinned ? '取消钉住' : '钉住流程'}
            onPointerDown={(event) => {
              event.preventDefault();
              event.stopPropagation();
            }}
            onClick={onPinToggle}
          >
            {pinned
              ? <PinOff className="h-3.5 w-3.5" strokeWidth={1.8} />
              : <Pin className="h-3.5 w-3.5" strokeWidth={1.8} />
            }
          </button>
        </div>
      </div>
      <div className="shrink-0 space-y-1" onMouseLeave={() => setActiveItem(null)}>
        {lifecycle.steps.map((step, index) => (
          <div
            key={`${step.title}-${index}`}
            className="flex items-center gap-2 p-1.5 hover:bg-black/5 dark:hover:bg-white/10 rounded cursor-pointer transition-colors"
            onMouseEnter={() => setActiveItem({type: 'step', index})}
          >
            <span
              className={`h-2.5 w-2.5 rounded-full shrink-0 ${
                step.state === 'complete'
                  ? 'bg-[#0b57d0]'
                  : step.state === 'active'
                    ? 'bg-[#0b57d0] shadow-[0_0_0_5px_rgba(11,87,208,0.12)] animate-pulse'
                    : 'bg-black/15 dark:bg-white/20'
              }`}
            />
            <span className="text-[12px] font-medium text-[#1f1f1f] dark:text-[#e3e3e3]">
              {step.title}
            </span>
          </div>
        ))}
      </div>
      {lifecycle.recentEvents.length > 0 && (() => {
        const maxIdx = lifecycle.recentEvents.length - 1;
        const safeIdx = Math.min(Math.max(sliderIndex, 0), maxIdx);
        const currentEvent = lifecycle.recentEvents[safeIdx];
        return (
          <div className="mt-2 flex min-h-0 flex-1 flex-col border-t border-black/10 pt-2 dark:border-white/10">
            <div className="mb-1.5 shrink-0 flex items-center justify-between px-1.5">
              <span className="text-[11px] font-medium text-[#5f6368] dark:text-[#bdc1c6]">事件流程</span>
              <span className="text-[10px] tabular-nums text-[#5f6368] dark:text-[#bdc1c6] opacity-70">{safeIdx + 1} / {lifecycle.recentEvents.length}</span>
            </div>
            {/* Slider */}
            <div className="px-1.5 mb-2">
              <input
                type="range"
                min={0}
                max={maxIdx}
                value={safeIdx}
                onChange={(e) => { setSliderIndex(Number(e.target.value)); setUserDragging(true); }}
                onMouseUp={() => setUserDragging(false)}
                onTouchEnd={() => setUserDragging(false)}
                className="lifecycle-slider w-full"
                aria-label="浏览事件"
              />
            </div>
            {/* Current event card */}
            {currentEvent && (
              <div
                className="mx-1.5 rounded-lg bg-black/[0.03] dark:bg-white/[0.05] p-2.5 transition-all duration-150"
                onMouseEnter={() => setActiveItem({type: 'event', index: safeIdx})}
                onMouseLeave={() => setActiveItem(null)}
              >
                <div className="flex items-center gap-1.5 mb-1">
                  <span className={`h-1.5 w-1.5 rounded-full shrink-0 ${
                    safeIdx === maxIdx ? 'bg-[#0b57d0] animate-pulse' : 'bg-[#0b57d0]/60'
                  }`} />
                  <span className="text-[12px] font-medium text-[#1f1f1f] dark:text-[#e3e3e3]">{currentEvent.label}</span>
                </div>
                <div className="text-[11px] leading-relaxed text-[#444746] dark:text-[#c4c7c5] whitespace-pre-wrap break-all line-clamp-4">
                  {currentEvent.content}
                </div>
              </div>
            )}
          </div>
        );
      })()}

      {/* Right Side Detail Panel */}
      {activeItem && (
        <div
          className="absolute left-full top-1/2 -translate-y-1/2 ml-2 w-[260px] z-50 flex flex-col rounded-xl border border-black/10 bg-white/95 p-4 text-left shadow-[0_8px_30px_rgba(0,0,0,0.12)] backdrop-blur-xl dark:border-white/10 dark:bg-[#1e1f20]/95 pointer-events-none"
        >
          {activeItem.type === 'step' && (
            <>
               <div className="font-medium text-[13px] text-[#1f1f1f] dark:text-[#e3e3e3] mb-1.5">{lifecycle.steps[activeItem.index].title}</div>
               <div className="text-[12px] leading-snug text-[#5f6368] dark:text-[#bdc1c6]">{lifecycle.steps[activeItem.index].detail}</div>
            </>
          )}
          {activeItem.type === 'event' && (
             <>
               <div className="font-medium text-[13px] text-[#1f1f1f] dark:text-[#e3e3e3] mb-1.5">{lifecycle.recentEvents[activeItem.index].label}</div>
               <div className="text-[12px] leading-snug text-[#444746] dark:text-[#c4c7c5] whitespace-pre-wrap break-all">{lifecycle.recentEvents[activeItem.index].content}</div>
             </>
          )}
        </div>
      )}
    </div>
  );
};

const ChatMessage = () => {
  const msgRef = useRef<HTMLDivElement>(null);
  const lifecycleAnchorRef = useRef<HTMLDivElement>(null);
  const { onRegenerate, realMessageCount } = useContext(ThreadActionsContext);
  const [copied, setCopied] = useState(false);
  const [showLifecycle, setShowLifecycle] = useState(false);
  const [lifecyclePinned, setLifecyclePinned] = useState(false);
  const [lifecyclePosition, setLifecyclePosition] = useState<LifecyclePosition>(() => ({
    x: LIFECYCLE_PANEL_MARGIN,
    y: 96,
  }));
  const lifecyclePanelRef = useRef<HTMLDivElement>(null);
  const lifecyclePositionRef = useRef(lifecyclePosition);
  const messageContent = useMessage((state) => state.content) as any[] || [];
  const messageRole = useMessage((state) => state.role);
  const messageStatusType = useMessage((state) => state.status?.type);
  const artifactArgs = getToolArgs(messageContent, "artifacts");
  const messageArtifacts = Array.isArray(artifactArgs) ? artifactArgs : [];
  const isMessageRunning = messageStatusType === 'running';
  const isAssistantThinking = messageRole === 'assistant'
    && isMessageRunning
    && getMessageText(messageContent).trim().length === 0;

  let thinkingText = '模型思考中';
  if (isAssistantThinking) {
    const lifecycle = buildLifecycle(messageContent, isMessageRunning);
    // keep lifecycle evaluation if needed for other side effects, but text is fixed
  }

  useEffect(() => {
    lifecyclePositionRef.current = lifecyclePosition;
  }, [lifecyclePosition]);
  
  useGSAP(() => {
    const message = msgRef.current;
    if (!message || prefersReducedMotion()) return;

    gsap.fromTo(message, {
      autoAlpha: 0,
      scale: 0.98,
      y: 8,
      willChange: 'transform, opacity',
    }, {
      autoAlpha: 1,
      scale: 1,
      y: 0,
      duration: 0.22,
      ease: 'power3.out',
      force3D: true,
      overwrite: 'auto',
      clearProps: 'transform,opacity,visibility,willChange',
    });
  }, []);

  const handleCopy = useCallback(() => {
    // Grab the text content from the message DOM node
    const textEl = msgRef.current?.querySelector('.whitespace-pre-wrap');
    const text = textEl?.textContent || '';
    if (!text.trim()) return;
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, []);

  const handleRegenerate = useCallback(() => {
    onRegenerate?.();
  }, [onRegenerate]);

  const placePinnedLifecycle = useCallback(() => {
    const rect = lifecycleAnchorRef.current?.getBoundingClientRect();
    const fallback = {
      x: typeof window === 'undefined'
        ? LIFECYCLE_PANEL_MARGIN
        : window.innerWidth - LIFECYCLE_PANEL_WIDTH - 24,
      y: 96,
    };
    if (!rect) {
      setLifecyclePosition(clampLifecyclePosition(fallback));
      return;
    }

    setLifecyclePosition(clampLifecyclePosition({
      x: rect.left + rect.width / 2 - LIFECYCLE_PANEL_WIDTH / 2,
      y: rect.top - LIFECYCLE_PANEL_HEIGHT - 10,
    }));
  }, []);

  const handleLifecyclePinToggle = useCallback(() => {
    if (lifecyclePinned) {
      setLifecyclePinned(false);
      setShowLifecycle(false);
      return;
    }

    placePinnedLifecycle();
    setLifecyclePinned(true);
    setShowLifecycle(true);
  }, [lifecyclePinned, placePinnedLifecycle]);

  const handleLifecycleDragStart = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (!lifecyclePinned || event.button !== 0) return;

    event.preventDefault();
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Some synthetic/browser combinations do not allow capture here.
    }
    const startX = event.clientX;
    const startY = event.clientY;
    const startPosition = lifecyclePositionRef.current;
    let nextPosition = startPosition;
    let frame = 0;

    const applyDragPosition = () => {
      frame = 0;
      const panel = lifecyclePanelRef.current;
      if (panel) {
        gsap.set(panel, {
          x: nextPosition.x,
          y: nextPosition.y,
          force3D: true,
        });
      }
    };

    const handlePointerMove = (moveEvent: PointerEvent) => {
      const dx = moveEvent.clientX - startX;
      const dy = moveEvent.clientY - startY;
      nextPosition = clampLifecyclePosition({
        x: startPosition.x + dx,
        y: startPosition.y + dy,
      });
      lifecyclePositionRef.current = nextPosition;
      if (!frame) frame = window.requestAnimationFrame(applyDragPosition);
    };

    const handlePointerUp = () => {
      if (frame) window.cancelAnimationFrame(frame);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointercancel', handlePointerUp);
      setLifecyclePosition(lifecyclePositionRef.current);
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp, { once: true });
    window.addEventListener('pointercancel', handlePointerUp, { once: true });
  }, [lifecyclePinned]);

  return (
    <MessagePrimitive.Root className="w-full flex flex-col mb-8 max-w-[800px] mx-auto px-4">
      <div ref={msgRef} className="w-full flex flex-col">
        {/* User Message */}
        <AuiIf condition={(s) => s.message.role === "user"}>
          <div className="flex justify-end w-full">
            <div className="max-w-[85%] md:max-w-[75%] rounded-3xl bg-[#f2f0f0] dark:bg-[#333537] px-5 py-2.5 text-[15px] text-[#1f1f1f] dark:text-[#e3e3e3] w-fit break-words">
              <MessagePrimitive.Parts components={{ Text: MarkdownText, ToolCall: CustomToolCall } as any} />
            </div>
          </div>
        </AuiIf>
        
        {/* Assistant Message */}
        <AuiIf condition={(s) => s.message.role === "assistant"}>
          <div className="flex flex-col items-start w-full group">
            <div className="max-w-full text-[15px] text-[#1f1f1f] dark:text-[#e3e3e3] leading-relaxed break-words">
              <MessagePrimitive.Parts components={{ Text: MarkdownText, ToolCall: CustomToolCall } as any} />
              {messageArtifacts.length > 0 && <ArtifactList artifacts={messageArtifacts} />}
              {isAssistantThinking && <AiThinking text={thinkingText} />}
            </div>
            
            {/* Action Bar — only show when not streaming */}
            <AuiIf condition={(s) => !s.message.isLast || !s.thread.isRunning}>
              <div className="flex items-center gap-2 mt-2 text-[#444746] dark:text-[#c4c7c5]">
                <button
                  className="p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-full transition-colors press-scale"
                  onClick={handleRegenerate}
                  title="重新生成"
                >
                  <RotateCw className="w-[15px] h-[15px]" strokeWidth={1.5} />
                </button>
                <button
                  className="p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-full transition-colors press-scale"
                  onClick={handleCopy}
                  title={copied ? "已复制" : "复制"}
                >
                  {copied
                    ? <Check className="w-[15px] h-[15px] text-green-500" strokeWidth={1.5} />
                    : <Copy className="w-[15px] h-[15px]" strokeWidth={1.5} />
                  }
                </button>
                <div
                  ref={lifecycleAnchorRef}
                  className="relative"
                  onMouseEnter={() => setShowLifecycle(true)}
                  onMouseLeave={() => { if (!lifecyclePinned) setShowLifecycle(false); }}
                  onFocus={() => setShowLifecycle(true)}
                  onBlur={() => { if (!lifecyclePinned) setShowLifecycle(false); }}
                >
                  <button
                    className="p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-full transition-colors press-scale"
                    title="思考流程"
                    aria-label="查看思考流程"
                    onClick={() => setShowLifecycle(true)}
                  >
                    <MoreHorizontal className="w-[15px] h-[15px]" strokeWidth={1.5} />
                  </button>
                  {(showLifecycle || lifecyclePinned) && (
                    <LifecycleHoverPanel
                      content={messageContent}
                      isRunning={isMessageRunning}
                      pinned={lifecyclePinned}
                      position={lifecyclePosition}
                      panelRef={lifecyclePanelRef}
                      onPinToggle={handleLifecyclePinToggle}
                      onDragStart={handleLifecycleDragStart}
                    />
                  )}
                </div>
                
                {/* Only show memory meter on the last message */}
                <AuiIf condition={(s) => s.message.isLast}>
                  <MemoryLimitUI messageCount={realMessageCount} />
                </AuiIf>
              </div>
            </AuiIf>
          </div>
        </AuiIf>
      </div>
    </MessagePrimitive.Root>
  );
};

const MemoryLimitUI = ({ messageCount }: { messageCount: number }) => {
  const progress = getConversationMemoryProgress(
    Array(messageCount).fill(null)
  );
  const percent = progress.percent;
  const strokeDasharray = 56.5;
  const strokeDashoffset = strokeDasharray * (1 - percent / 100);

  return (
    <div className="flex items-center gap-1.5 ml-1 pl-3 border-l border-black/10 dark:border-white/10 h-4">
      <div className="relative flex items-center justify-center w-3.5 h-3.5">
        <svg className="w-3.5 h-3.5 -rotate-90" viewBox="0 0 24 24">
          {/* Background circle */}
          <circle
            cx="12"
            cy="12"
            r="9"
            fill="none"
            stroke="currentColor"
            strokeWidth="3.5"
            className="opacity-20"
          />
          {/* Progress circle */}
          <circle
            cx="12"
            cy="12"
            r="9"
            fill="none"
            stroke="currentColor"
            strokeWidth="3.5"
            strokeDasharray={strokeDasharray}
            strokeDashoffset={strokeDashoffset}
            className="opacity-80 transition-[stroke-dashoffset,opacity] duration-200 ease-[var(--ease-out)]"
          />
        </svg>
      </div>
      <span className="text-[13px] font-medium text-[#444746] dark:text-[#c4c7c5]">
        {percent}%
      </span>
    </div>
  );
};
