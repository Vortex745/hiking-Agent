import { Fragment, type CSSProperties } from 'react'

/**
 * Arc Text Reveal：片段从各自的裁切框中升起，同时从柔和模糊中收紧。
 * 英文按词切分；中文逐字显现并保留标点跟随，长句也能正常换行。
 */
export interface TextRevealProps {
  /** 要显现的文字；用 `\n` 明确换行。 */
  text: string
  as?: 'h1' | 'h2' | 'h3' | 'p'
  className?: string
  id?: string
  /** 第一个片段升起前的等待秒数，默认 0。 */
  delay?: number
}

/** 无论文字多长，总错峰不超过该秒数。 */
const MAX_STAGGER = 0.48
/** 英文单词之间的基础错峰。 */
const UNIT_STAGGER = 0.04
const CJK = /[\u3400-\u9fff\uf900-\ufaff]/
/** 这些标点跟随前一个片段，避免出现在行首。 */
const TRAILING_PUNCTUATION = new Set([
  '，', '。', '、', '；', '：', '！', '？',
  '）', '】', '》', '」', '』', '”', '’', '…',
  ',', '.', '!', '?', ';', ':', ')', ']', '%',
])

type Unit = { text: string; joiner: 'space' | 'break' }

/** 纯中文逐字切分；混合 token 中的英文串保持完整。 */
function splitToken(token: string): string[] {
  if (!CJK.test(token)) return [token]

  const units: string[] = []
  let latin = ''
  const flushLatin = () => {
    if (!latin) return
    units.push(latin)
    latin = ''
  }

  for (const char of token) {
    if (CJK.test(char)) {
      flushLatin()
      units.push(char)
    } else if (TRAILING_PUNCTUATION.has(char)) {
      flushLatin()
      if (units.length) units[units.length - 1] += char
      else units.push(char)
    } else {
      latin += char
    }
  }
  flushLatin()
  return units
}

function splitLine(line: string): Unit[] {
  const units: Unit[] = []
  line
    .split(/\s+/)
    .filter(Boolean)
    .forEach((token, tokenIndex) => {
      splitToken(token).forEach((part, partIndex) => {
        units.push({
          text: part,
          joiner: partIndex > 0 ? 'break' : tokenIndex > 0 ? 'space' : 'break',
        })
      })
    })
  return units
}

export function TextReveal({ text, as = 'h2', className, id, delay = 0 }: TextRevealProps) {
  const Tag = as
  const lines = text.split('\n').map(splitLine)
  const count = lines.reduce((total, units) => total + units.length, 0)
  const step = Math.min(UNIT_STAGGER, MAX_STAGGER / Math.max(count, 1))
  const blur = as === 'p' ? 4 : 8
  const spoken = text.replace(/\n/g, ' ')
  const named = as !== 'p'
  let index = 0

  return (
    <Tag
      id={id}
      aria-label={named ? spoken : undefined}
      className={['arc-reveal', className].filter(Boolean).join(' ')}
      style={{ '--reveal-blur': `${blur}px` } as CSSProperties}
    >
      {named ? null : <span className="arc-reveal-sr">{spoken}</span>}
      {lines.map((units, lineIndex) => (
        <Fragment key={lineIndex}>
          <span className="arc-reveal-line" aria-hidden="true">
            {units.map((unit, unitIndex) => {
              const position = index++
              return (
                <Fragment key={`${unit.text}-${position}`}>
                  {unitIndex > 0 ? (unit.joiner === 'space' ? ' ' : <wbr />) : null}
                  <span className="arc-reveal-clip">
                    <span
                      className="arc-reveal-word"
                      style={{ '--reveal-delay': `${delay + position * step + lineIndex * step}s` } as CSSProperties}
                    >
                      {unit.text}
                    </span>
                  </span>
                </Fragment>
              )
            })}
          </span>
          {lineIndex < lines.length - 1 ? ' ' : null}
        </Fragment>
      ))}
    </Tag>
  )
}

export default TextReveal
