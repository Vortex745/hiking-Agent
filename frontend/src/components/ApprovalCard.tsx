import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { Button } from './Button';
import GlideMenu from './GlideMenu';

/* ─────────────────────────────────────────────────────────
 * APPROVAL CARD（human-in-the-loop）
 * 一次只问一个问题。切题时整叠竖向滑行（卡片高度动画自适应），
 * 步数计数器像里程表一样滚动，底部是胶囊操作——安静的「跳过」
 * 和品牌绿「继续」（带 ⏎）。单选自动前进，多选等待用户确认。
 * 视觉 token 已对齐本项目：森林绿强调、米色 body、var() 双主题。
 * ───────────────────────────────────────────────────────── */

export type ApprovalQuestion = {
  q: string;
  type: 'radio' | 'check';
  options: string[];
};

export const DEFAULT_APPROVAL_QUESTIONS: ApprovalQuestion[] = [
  {
    q: '这条线路打算走几天？',
    type: 'radio',
    options: ['一日往返', '两天一夜', '三天及以上'],
  },
  {
    q: '更看重哪些体验？',
    type: 'check',
    options: ['观景拍照', '露营看星空', '沿途美食'],
  },
  {
    q: '希望的强度难度？',
    type: 'radio',
    options: ['轻松休闲', '中等挑战', '硬核穿越'],
  },
];

export type ApprovalLabels = {
  skip: string;
  continue: string;
  send: string;
  customPlaceholder: string;
  sentMessage: string;
};

const DEFAULT_LABELS: ApprovalLabels = {
  skip: '跳过',
  continue: '继续',
  send: '发送',
  customPlaceholder: '其他想法…',
  sentMessage: '已提交',
};

const ROLL_MS = 400;
const SLIDE = '360ms cubic-bezier(0.22, 1, 0.36, 1)';

/* odometer digits — each character that changes rolls up (or down) */
function RollingDigits({ value }: { value: string }) {
  const prevRef = useRef(value);
  const [oldVal, setOldVal] = useState(value);
  const [newVal, setNewVal] = useState(value);
  const [rolling, setRolling] = useState(false);
  const [shifted, setShifted] = useState(false);
  const [dir, setDir] = useState<'up' | 'down'>('up');

  useEffect(() => {
    if (prevRef.current === value) return;
    const from = prevRef.current;
    prevRef.current = value;
    const fromN = parseInt(from, 10);
    const toN = parseInt(value, 10);
    setDir(Number.isFinite(fromN) && Number.isFinite(toN) && toN < fromN ? 'down' : 'up');
    setOldVal(from);
    setNewVal(value);
    setRolling(true);
    setShifted(false);

    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => setShifted(true));
    });
    const done = setTimeout(() => {
      setRolling(false);
      setOldVal(value);
      setShifted(false);
    }, ROLL_MS);

    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
      clearTimeout(done);
    };
  }, [value]);

  const chars = rolling ? newVal : oldVal;

  return (
    <>
      {Array.from({ length: chars.length }, (_, i) => {
        const o = oldVal[i] ?? '';
        const n = chars[i] ?? '';
        if (!rolling || o === n) {
          return <span key={`${i}-${n}`}>{n}</span>;
        }
        const top = dir === 'down' ? n : o;
        const bottom = dir === 'down' ? o : n;
        const restY = dir === 'down' ? '0' : '-1em';
        const startY = dir === 'down' ? '-1em' : '0';
        return (
          <span
            key={`${i}-${o}-${n}-${dir}`}
            style={{ display: 'inline-block', position: 'relative', overflow: 'hidden', height: '1em', lineHeight: '1em', verticalAlign: '-0.05em' }}
          >
            <span
              style={{
                display: 'flex',
                flexDirection: 'column',
                transition: 'transform 350ms cubic-bezier(0.4, 0, 0.2, 1)',
                transform: `translateY(${shifted ? restY : startY})`,
              }}
            >
              <span style={{ height: '1em', lineHeight: '1em' }}>{top}</span>
              <span style={{ height: '1em', lineHeight: '1em' }}>{bottom}</span>
            </span>
          </span>
        );
      })}
    </>
  );
}

function Ico({ path, size = 14, sw = 2 }: { path: ReactNode; size?: number; sw?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      {path}
    </svg>
  );
}

export default function ApprovalCard({
  questions = DEFAULT_APPROVAL_QUESTIONS,
  labels,
  onSubmitted,
  onAnswerChange,
  resettable = true,
}: {
  questions?: ApprovalQuestion[];
  labels?: Partial<ApprovalLabels>;
  onSubmitted?: (answers: Record<number, number[]>) => void;
  onAnswerChange?: (questionIndex: number, answer: number[]) => void;
  resettable?: boolean;
} = {}) {
  const t = { ...DEFAULT_LABELS, ...labels };
  const [qi, setQi] = useState(0);
  const [answers, setAnswers] = useState<Record<number, number[]>>({});
  const [custom, setCustom] = useState<Record<number, string>>({});
  const [sent, setSent] = useState(false);
  const [open, setOpen] = useState(true);

  const advanceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const questionRefs = useRef<(HTMLDivElement | null)[]>([]);
  const measured = useRef(false);
  const [viewportH, setViewportH] = useState<number | undefined>(undefined);
  const [trackY, setTrackY] = useState(0);
  const [animate, setAnimate] = useState(false);
  // 首次量到第一题高度前只渲染当前题，避免初始（及 SSR）高度是全部
  // 问题叠加的高度——那样挂载时会先撑满再回缩，产生一次闪烁。
  const [ready, setReady] = useState(false);

  const last = qi === questions.length - 1;
  const selected = answers[qi] ?? [];
  const hasAnswer = selected.length > 0 || Boolean(custom[qi]?.trim());

  const sync = (withAnim: boolean) => {
    const item = questionRefs.current[qi];
    if (!item) return;
    const reduce = typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    setViewportH(item.offsetHeight);
    setTrackY(item.offsetTop);
    setAnimate(withAnim && !reduce);
  };

  useLayoutEffect(() => {
    const withAnim = measured.current;
    measured.current = true;
    sync(withAnim);
    setReady(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [qi, answers, custom, open, sent]);

  useEffect(() => {
    const id = requestAnimationFrame(() => sync(measured.current));
    return () => cancelAnimationFrame(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [qi]);

  useEffect(() => () => { if (advanceTimer.current) clearTimeout(advanceTimer.current); }, []);

  const goTo = (next: number) => {
    if (advanceTimer.current) clearTimeout(advanceTimer.current);
    setQi(Math.min(Math.max(next, 0), questions.length - 1));
  };

  const send = () => {
    if (advanceTimer.current) clearTimeout(advanceTimer.current);
    setSent(true);
    onSubmitted?.(answers);
  };

  const advance = () => {
    if (last) send();
    else goTo(qi + 1);
  };

  const toggle = (index: number) => {
    const type = questions[qi].type;
    setAnswers((current) => {
      const picked = current[qi] ?? [];
      const next = type === 'radio'
        ? [index]
        : picked.includes(index)
          ? picked.filter((item) => item !== index)
          : [...picked, index];
      onAnswerChange?.(qi, next);
      return { ...current, [qi]: next };
    });
    if (type === 'radio') {
      setCustom((current) => ({ ...current, [qi]: '' }));
      if (advanceTimer.current) clearTimeout(advanceTimer.current);
      advanceTimer.current = setTimeout(() => {
        if (last) send();
        else setQi((current) => Math.min(questions.length - 1, current + 1));
      }, 480);
    }
  };

  const reset = () => {
    setQi(0);
    setAnswers({});
    setCustom({});
    setSent(false);
    setOpen(true);
    measured.current = false;
  };

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="rounded-lg border border-[color:var(--border-color)] bg-[color:var(--bg-card)] px-3 py-2 text-[12.5px] font-medium text-text-primary shadow-sm transition-colors duration-150 hover:bg-[color:var(--bg-body)]"
      >
        展开提问
      </button>
    );
  }

  if (sent) {
    return (
      <div className="flex w-full max-w-sm items-center gap-3" style={{ animation: 'msgEnter 260ms cubic-bezier(0.23,1,0.32,1) both' }}>
        <span className="inline-flex items-center gap-1.5 rounded-full bg-[rgba(45,106,79,0.12)] py-1 pr-2.5 pl-1 text-[12.5px] font-medium text-[color:var(--primary-light)]">
          <span className="flex size-[18px] items-center justify-center rounded-full bg-[color:var(--primary-light)] text-white">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5" /></svg>
          </span>
          {t.sentMessage}
        </span>
        {resettable && (
          <button type="button" onClick={reset} className="text-[12px] font-medium text-text-muted transition-colors duration-150 hover:text-text-primary">
            重新开始
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="w-full max-w-sm">
      <div className="relative overflow-hidden rounded-xl border border-[color:var(--border-color)] bg-[color:var(--bg-card)] shadow-md" style={{ animation: 'fadeUp 380ms cubic-bezier(0.23,1,0.32,1) both' }}>
        <button
          type="button"
          aria-label="关闭"
          onClick={() => setOpen(false)}
          className="absolute right-2.5 top-2.5 z-10 flex size-7 items-center justify-center rounded-md text-text-muted transition-colors duration-100 hover:bg-[color:var(--bg-body)] hover:text-text-primary"
        >
          <Ico size={14} sw={2.2} path={<path d="M18 6L6 18M6 6l12 12" />} />
        </button>
        <div className="p-4">
          {/* the question itself is the heading */}
          <div
            className="overflow-hidden"
            style={{ height: viewportH, transition: animate ? `height ${SLIDE}` : undefined }}
            aria-live="polite"
          >
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: 26,
                transform: `translate3d(0, ${-trackY}px, 0)`,
                transition: animate ? `transform ${SLIDE}` : undefined,
                willChange: 'transform',
              }}
            >
              {questions.map((question, qIdx) => {
                const active = qIdx === qi;
                // 首次测量前只挂当前题，让卡片以真实高度打开而非闪到全高。
                if (!ready && !active) return null;
                const picked = answers[qIdx] ?? [];
                const questionStyle: CSSProperties = {
                  opacity: active ? 1 : 0,
                  transition: animate ? `opacity ${SLIDE}` : undefined,
                  pointerEvents: active ? undefined : 'none',
                };
                return (
                  <div
                    key={qIdx}
                    ref={(el) => { questionRefs.current[qIdx] = el; }}
                    aria-hidden={active ? undefined : true}
                    style={questionStyle}
                  >
                    <div className="pr-7 text-[14px] font-medium text-text-primary">{question.q}</div>
                    <GlideMenu className="mt-2.5 flex flex-col gap-1" highlightClassName="inset-x-0 rounded-md bg-[color:var(--bg-body)]">
                      {question.options.map((option, i) => {
                        const on = picked.includes(i);
                        return (
                          <button
                            key={option}
                            type="button"
                            data-menu-row
                            aria-pressed={on}
                            tabIndex={active ? 0 : -1}
                            onClick={() => { if (active) toggle(i); }}
                            className="relative z-10 flex items-center gap-1.5 rounded-md py-1 pl-1 pr-2 text-left transition-colors duration-100"
                          >
                            <span
                              className={`flex size-4 shrink-0 items-center justify-center transition-colors duration-200
                                ${question.type === 'radio' ? 'rounded-full' : 'rounded-[5px]'}
                                ${on ? 'bg-[color:var(--primary-light)] text-white' : 'text-transparent shadow-[inset_0_0_0_1.5px_var(--border-color)]'}`}
                            >
                              {question.type === 'radio' ? (
                                <span className="size-1.5 rounded-full bg-white transition-transform duration-200" style={{ transform: on ? 'scale(1)' : 'scale(0)' }} />
                              ) : (
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5" /></svg>
                              )}
                            </span>
                            <span className={`text-[13px] leading-none transition-colors duration-200 ${on ? 'text-text-primary' : 'text-text-secondary'}`}>
                              {option}
                            </span>
                          </button>
                        );
                      })}
                      <label data-menu-row className="relative z-10 flex items-center gap-1.5 rounded-md py-1 pl-1 pr-2 transition-colors duration-100">
                        <input
                          value={custom[qIdx] ?? ''}
                          tabIndex={active ? 0 : -1}
                          onChange={(event) => {
                            if (!active) return;
                            setCustom((current) => ({ ...current, [qIdx]: event.target.value }));
                            if (question.type === 'radio') setAnswers((current) => ({ ...current, [qIdx]: [] }));
                          }}
                          onKeyDown={(event) => {
                            if (event.key === 'Enter' && hasAnswer) {
                              event.preventDefault();
                              advance();
                            }
                          }}
                          placeholder={t.customPlaceholder}
                          aria-label="自定义回答"
                          className="min-w-0 flex-1 bg-transparent pl-1.5 text-[13px] text-text-primary outline-none placeholder:text-text-muted"
                        />
                      </label>
                    </GlideMenu>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* footer — step nav (rolling counter) + pill actions */}
        <div className="flex items-center justify-between gap-3 border-t border-[color:var(--border-color)] px-3 py-2.5">
          <div className="flex items-center gap-1 text-text-muted">
            <button
              type="button"
              aria-label="上一题"
              disabled={qi <= 0}
              onClick={() => goTo(qi - 1)}
              className="flex size-[18px] items-center justify-center rounded-[5px] transition-colors duration-100 enabled:hover:text-text-primary disabled:opacity-30"
            >
              <Ico size={14} path={<path d="M18 15l-6-6-6 6" />} />
            </button>
            <span className="inline-flex items-center text-[12px] font-medium tabular-nums text-text-muted" style={{ letterSpacing: '-0.1px', lineHeight: 1 }}>
              <RollingDigits value={`${qi + 1} / ${questions.length}`} />
            </span>
            <button
              type="button"
              aria-label="下一题"
              disabled={last}
              onClick={() => goTo(qi + 1)}
              className="flex size-[18px] items-center justify-center rounded-[5px] transition-colors duration-100 enabled:hover:text-text-primary disabled:opacity-30"
            >
              <Ico size={14} path={<path d="M6 9l6 6 6-6" />} />
            </button>
          </div>

          <div className="-mr-0.5 flex items-center gap-1.5">
            <Button variant="ghost" size="sm" onClick={() => (last ? setOpen(false) : goTo(qi + 1))}>
              {t.skip}
            </Button>
            <Button variant="accent" size="sm" disabled={!hasAnswer} onClick={advance}>
              {last ? t.send : t.continue}
              <span className="ml-0.5 opacity-60">⏎</span>
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
