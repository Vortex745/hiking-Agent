import { forwardRef, type ButtonHTMLAttributes } from 'react';

export type ButtonVariant = 'ghost' | 'accent' | 'outline';
export type ButtonSize = 'sm' | 'md';

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: ButtonSize;
};

const BASE =
  'inline-flex select-none items-center justify-center gap-1.5 rounded-lg font-medium ' +
  'transition-colors duration-150 press-scale disabled:pointer-events-none disabled:opacity-45';

const VARIANTS: Record<ButtonVariant, string> = {
  // 强调：项目品牌森林绿，白字（与用户气泡/发送键同色板）
  accent: 'bg-[color:var(--primary-light)] text-white shadow-sm hover:bg-[color:var(--primary-hover)]',
  // 次要：透明底，悬停浮出 body 米色
  ghost: 'text-text-secondary hover:bg-[color:var(--bg-body)] hover:text-text-primary',
  // 描边：与 RouteComparison 的次要按钮一致
  outline: 'border border-[var(--border-color)] text-text-primary hover:bg-[color:var(--bg-body)]',
};

const SIZES: Record<ButtonSize, string> = {
  sm: 'h-8 px-3 text-[13px]',
  md: 'h-9 px-4 text-sm',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'ghost', size = 'sm', type = 'button', className = '', ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={`${BASE} ${VARIANTS[variant]} ${SIZES[size]} ${className}`.trim()}
      {...props}
    />
  );
});

export default Button;
