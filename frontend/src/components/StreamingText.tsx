import { Fragment, useEffect, useRef, useState } from 'react';

// Keep whitespace verbatim; Chinese/Japanese do not use spaces between words.
const STREAM_UNITS = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]|[^\s\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]+|\s+/gu;

export function useStreamingText(text: string, running: boolean) {
  const [reducedMotion, setReducedMotion] = useState(
    () => window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  );
  const [visible, setVisible] = useState(() => running && !reducedMotion ? '' : text);
  const targetRef = useRef(text);
  const liveRef = useRef(running);
  targetRef.current = text;
  if (running) liveRef.current = true;

  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReducedMotion(query.matches);
    update();
    query.addEventListener('change', update);
    return () => query.removeEventListener('change', update);
  }, []);

  useEffect(() => {
    if (reducedMotion || !liveRef.current) {
      setVisible(text);
      return;
    }
    if (!text.startsWith(visible)) {
      // Recovery/replacement is a new source, not an append to the old queue.
      setVisible(text);
      return;
    }
    if (visible === text && !running) return;
    const rawGap = getComputedStyle(document.documentElement).getPropertyValue('--p30-word-gap').trim();
    const gap = Number.parseFloat(rawGap) * (rawGap.endsWith('ms') ? 1 : rawGap.endsWith('s') ? 1000 : 1);
    const timer = window.setInterval(() => {
      setVisible(previous => {
        const target = targetRef.current;
        if (!target.startsWith(previous)) return target;
        const pending = target.slice(previous.length).match(STREAM_UNITS) || [];
        // Large buffered paragraphs catch up in batches; small deltas
        // use the configured word cadence.
        const count = Math.max(1, Math.ceil(pending.length / 12));
        return previous + pending.slice(0, count).join('');
      });
    }, Number.isFinite(gap) && gap > 0 ? gap : 20);
    return () => window.clearInterval(timer);
  // New transport chunks update targetRef without resetting the reveal clock.
  }, [running, reducedMotion, visible === text]);

  return { text: reducedMotion || !liveRef.current ? text : visible,
    animate: liveRef.current && !reducedMotion };
}

/** New spans cross-blur in; stable keys keep already visible words at rest. */
export function StreamingText({ text, animate }: { text: string; animate: boolean }) {
  if (!animate) return <>{text}</>;
  return <>{(text.match(STREAM_UNITS) || []).map((unit, index) =>
    /^\s+$/.test(unit)
      ? <Fragment key={index}>{unit}</Fragment>
      : <span key={index} className="streaming-word">{unit}</span>,
  )}</>;
}
