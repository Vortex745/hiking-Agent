import {
  AuiIf,
  ComposerPrimitive,
  ThreadPrimitive,
  MessagePrimitive,
  useMessage,
} from "@assistant-ui/react";
import { ArrowUp, Check, Copy, Download, Paperclip, RotateCw } from 'lucide-react';
import { getConversationMemoryProgress } from '../../../api/conversationMemory';
import { API } from '../../../api/config';
import { splitLegacyMcpOutput } from '../../../lib/chatOutput';
import { StreamingText, useStreamingText } from '../../StreamingText';
import { AgentActivityTrace } from '../../agent-activity/AgentActivityTrace';
import { TextReveal } from '../../ui/text-reveal';
import { ViewOnMap } from '../../ui/view-on-map';
import { ChatImages, type ChatImage } from '../../ui/chat-images';
import {
  buildAgentActivity,
  readActivityEvents,
} from '../../agent-activity/agentActivityModel';
import { useRef, useState, createContext, useContext, useCallback, type RefObject } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(useGSAP);

interface GeminiThreadProps {
  emptyTitle: string;
  /** Called when user clicks "regenerate" — parent should resend the last user prompt */
  onRegenerate?: () => void;
  onUploadClick?: () => void;
  /** Real message count from the parent page's state (for accurate memory meter) */
  realMessageCount?: number;
  /** 当前会话最近一次由 AI 聚焦的唯一景点 */
  mapFocus?: { name: string } | null;
  /** 会话标识，切换后地图重新解析目标 */
  mapSessionKey?: string;
}

/** Context to pass callbacks down to ChatMessage without prop drilling */
const ThreadActionsContext = createContext<{
  onRegenerate?: () => void;
  realMessageCount: number;
}>({ realMessageCount: 0 });

export const GeminiThread = ({ emptyTitle, onRegenerate, onUploadClick, realMessageCount = 0, mapFocus, mapSessionKey }: GeminiThreadProps) => (
  <ThreadActionsContext.Provider value={{ onRegenerate, realMessageCount }}>
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
      <ViewOnMap focus={mapFocus} sessionKey={mapSessionKey} />
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
      <TextReveal
        as="h1"
        className="text-center text-3xl font-medium mb-12 text-[#1f1f1f] dark:text-[#e3e3e3]"
        text={emptyTitle}
      />
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
          <Paperclip className="w-[18px] h-[18px]" />
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
          <ArrowUp className="w-[18px] h-[18px]" aria-hidden="true" />
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
// Only image syntax becomes a ChatImages preview. A normal link to an image
// URL is still a link; it opens in a new tab and never fetches a protected
// original just because its path ends in .jpg.
const imageFromMarkdown = (segment: string): ChatImage | null => {
  const imageMatch = segment.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
  if (imageMatch) return { alt: imageMatch[1] || '图片', src: imageMatch[2] };
  return null;
};

const MarkdownText = (part: any) => {
  const raw = part.text || part.part?.text || '';
  // Older messages persisted raw MCP JSON after this marker. Strip it at
  // display time (and surface any image URLs) without rewriting history.
  const displayText = splitLegacyMcpOutput(raw);
  const role = useMessage((state) => state.role);
  const status = useMessage((state) => state.status?.type);
  const reveal = useStreamingText(displayText, role === 'assistant' && status === 'running');
  // Split on bold, images, and links patterns
  const segments = reveal.text.split(MARKDOWN_PATTERN);
  const imageRuns: { start: number; end: number; images: ChatImage[] }[] = [];
  let imageRun: (typeof imageRuns)[number] | null = null;
  segments.forEach((seg: string, i: number) => {
    const image = imageFromMarkdown(seg);
    if (image) {
      if (!imageRun) {
        imageRun = { start: i, end: i, images: [] };
        imageRuns.push(imageRun);
      }
      imageRun.end = i;
      imageRun.images.push(image);
      return;
    }
    if (imageRun && !seg.trim()) { imageRun.end = i; return; }
    imageRun = null;
  });

  return (
    <div className="whitespace-pre-wrap leading-relaxed relative streaming-text" data-full-text={displayText}>
      {segments.map((seg: string, i: number) => {
        if (!seg) return null;
        if (imageFromMarkdown(seg)) {
          const run = imageRuns.find((entry) => entry.start === i);
          return run ? <ChatImages key={`images-${i}`} images={run.images} /> : null;
        }
        if (!seg.trim() && imageRuns.some((entry) => i > entry.start && i <= entry.end)) return null;

        const linkMatch = seg.match(/^\[([^\]]*)\]\(([^)]+)\)$/);
        if (linkMatch) {
          const linkText = linkMatch[1];
          const linkUrl = linkMatch[2];
          return (
            <a
              key={i}
              href={linkUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#0b57d0] hover:underline dark:text-[#a8c7fa]"
            >
              <StreamingText text={linkText} animate={reveal.animate} />
            </a>
          );
        }

        if (seg.startsWith('**') && seg.endsWith('**')) {
          return <strong key={i} className="font-bold"><StreamingText text={seg.slice(2, -2)} animate={reveal.animate} /></strong>;
        }
        
        return <span key={i}><StreamingText text={seg} animate={reveal.animate} /></span>;
      })}
    </div>
  );
};

/** A03: v2 artifact download card. The payload carries only the opaque
 * artifact_id + public metadata; download goes through the shared
 * /api/v2/artifacts/{artifact_id} endpoint (no v1 download_url exists). */
const ArtifactList = ({ artifacts }: { artifacts: any[] }) => {
  const [downloading, setDownloading] = useState<string | null>(null);

  const handleDownload = async (artifactId: string, filename: string) => {
    const url = API.chatV2Artifact(artifactId);
    setDownloading(artifactId);
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
        const artifactId = typeof artifact?.artifact_id === 'string' ? artifact.artifact_id : '';
        const filename = typeof artifact?.public_filename === 'string' ? artifact.public_filename : '下载文件';
        const size = formatArtifactSize(artifact?.size_bytes);
        const isDownloading = downloading === artifactId;
        if (!artifactId) return null;
        return (
          <div key={i} className="flex flex-wrap items-center gap-2 text-[13px] text-text-secondary">
            <button
              type="button"
              onClick={() => handleDownload(artifactId, filename)}
              disabled={isDownloading}
              className="inline-flex min-h-8 items-center gap-1.5 rounded-md border border-border/70 bg-background px-2.5 py-1.5 font-medium text-text-primary transition-colors hover:border-primary/60 hover:text-primary disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Download className={`size-4${isDownloading ? ' animate-pulse' : ''}`} aria-hidden="true" />
              <span>{isDownloading ? '下载中…' : `下载 ${filename}`}</span>
            </button>
            {size && <span className="text-[12px] opacity-70">{size}</span>}
          </div>
        )
      })}
    </div>
  );
}

const CustomToolCall = (part: any) => {
  const toolName = part.toolName || part.part?.toolName;
  const args = part.args || part.part?.args;
  
  if (toolName === "traceEvents") {
    // The synthetic trace part is rendered by AgentActivityTrace above the
    // answer; it must not also appear as an inline tool call.
    return null;
  }

  if (toolName === "artifacts") {
    // A03: render v2 download cards — the artifacts tool-call part carries
    // the message's artifact list as args (wired by convertMessage).
    const artifactList = Array.isArray(args) ? args : [];
    if (artifactList.length === 0) return null;
    return <ArtifactList artifacts={artifactList} />;
  }

  if (part.toolName === "evidence") {
    // A09: retrieval sources aggregated from tool_result evidence — the
    // chat knowledge tool rides citations on the public payload.
    const sources = Array.isArray(part.args) ? part.args : [];
    if (sources.length === 0) return null;
    return (
      <details className="mt-4 border-t border-border/40 pt-3 text-[13px] text-text-secondary">
        <summary className="cursor-pointer select-none font-medium text-text-primary outline-none">
          检索来源 · {sources.length} 条片段
        </summary>
        <div className="mt-3 space-y-3 opacity-90">
          {sources.map((doc: any, idx: number) => (
            <div key={`${doc.source}-${idx}`} className="border-l-2 border-primary/25 pl-3">
              <div className="font-medium text-text-primary">
                {idx + 1}. {doc.title || doc.source}
              </div>
              <div className="text-[12px] mt-0.5 opacity-70">
                来源：{sourceLabel(doc.source)}
              </div>
              {doc.preview && (
                <div className="mt-1.5 whitespace-pre-wrap text-text-secondary">
                  {doc.preview}
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

function prefersReducedMotion() {
  return typeof window !== 'undefined'
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

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

const ChatMessage = () => {
  const msgRef = useRef<HTMLDivElement>(null);
  const { onRegenerate, realMessageCount } = useContext(ThreadActionsContext);
  const [copied, setCopied] = useState(false);
  const messageContent = useMessage((state) => state.content) as any[] || [];
  const messageStatusType = useMessage((state) => state.status?.type);
  const artifactArgs = getToolArgs(messageContent, "artifacts");
  const messageArtifacts = Array.isArray(artifactArgs) ? artifactArgs : [];
  const isMessageRunning = messageStatusType === 'running';
  const activityModel = buildAgentActivity({
    events: readActivityEvents(messageContent),
    isRunning: isMessageRunning,
    hasText: getMessageText(messageContent).trim().length > 0,
  });

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
    const text = textEl?.getAttribute('data-full-text') || textEl?.textContent || '';
    if (!text.trim()) return;
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, []);

  const handleRegenerate = useCallback(() => {
    onRegenerate?.();
  }, [onRegenerate]);

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
              <AgentActivityTrace model={activityModel} className="mb-3" />
              <MessagePrimitive.Parts components={{ Text: MarkdownText, ToolCall: CustomToolCall } as any} />
              {messageArtifacts.length > 0 && <ArtifactList artifacts={messageArtifacts} />}
            </div>
            
            {/* Action Bar — only show when not streaming */}
            <AuiIf condition={(s) => !s.message.isLast || !s.thread.isRunning}>
              <div className="flex items-center gap-2 mt-2 text-[#444746] dark:text-[#c4c7c5]">
                <button
                  className="p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-full transition-colors press-scale"
                  onClick={handleRegenerate}
                  title="重新生成"
                >
                  <RotateCw className="w-[15px] h-[15px]" />
                </button>
                <button
                  className="p-1.5 hover:bg-black/5 dark:hover:bg-white/5 rounded-full transition-colors press-scale"
                  onClick={handleCopy}
                  title={copied ? "已复制" : "复制"}
                >
                  {copied
                    ? <Check className="w-[15px] h-[15px] text-green-500" />
                    : <Copy className="w-[15px] h-[15px]" />
                  }
                </button>
                
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
