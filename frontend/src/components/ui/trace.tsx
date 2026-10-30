import * as React from "react"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@radix-ui/react-collapsible"
import { clsx, type ClassValue } from "clsx"
import { ChevronRight } from "lucide-react"
import { twMerge } from "tailwind-merge"

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

const TraceCollapsibleIdContext = React.createContext<string | undefined>(
  undefined,
)

function Trace({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="trace"
      className={cn("flex w-full min-w-0 flex-col gap-1", className)}
      {...props}
    />
  )
}

function TraceLine({
  className,
  status,
  ...props
}: React.ComponentProps<"span"> & {
  status?: "running" | "error" | "pending"
}) {
  return (
    <span
      data-slot="trace-line"
      data-status={status}
      className={cn(
        "group/trace-line flex w-fit max-w-full min-w-0 items-center gap-1.5 text-left text-text-secondary transition-colors data-[status=error]:text-red-600 data-[status=pending]:opacity-60 dark:data-[status=error]:text-red-400",
        className,
      )}
      {...props}
    />
  )
}

function TraceIcon({ className, ...props }: React.ComponentProps<"span">) {
  return (
    <span
      data-slot="trace-icon"
      aria-hidden="true"
      className={cn(
        "flex shrink-0 items-center justify-center leading-none [&_svg]:pointer-events-none [&_svg]:shrink-0",
        className,
      )}
      {...props}
    />
  )
}

function TraceTitle({ className, ...props }: React.ComponentProps<"span">) {
  return (
    <span
      data-slot="trace-title"
      className={cn(
        "min-w-0 truncate group-data-[status=running]/trace-line:motion-safe:animate-shimmer group-data-[status=running]/trace-line:motion-safe:bg-gradient-to-r group-data-[status=running]/trace-line:motion-safe:from-[#a3a3a3] group-data-[status=running]/trace-line:motion-safe:via-[#111] group-data-[status=running]/trace-line:motion-safe:to-[#a3a3a3] group-data-[status=running]/trace-line:motion-safe:bg-[length:200%_auto] group-data-[status=running]/trace-line:motion-safe:bg-clip-text group-data-[status=running]/trace-line:motion-safe:text-transparent dark:group-data-[status=running]/trace-line:motion-safe:from-[#5f6368] dark:group-data-[status=running]/trace-line:motion-safe:via-[#ffffff] dark:group-data-[status=running]/trace-line:motion-safe:to-[#5f6368]",
        className,
      )}
      {...props}
    />
  )
}

function TraceDescription({
  className,
  ...props
}: React.ComponentProps<"span">) {
  return (
    <span
      data-slot="trace-description"
      className={cn(
        "shrink-0 text-xs tabular-nums text-text-muted",
        className,
      )}
      {...props}
    />
  )
}

function TraceCollapsible({
  className,
  children,
  open,
  onOpenChange,
  ...props
}: Omit<React.ComponentProps<typeof Collapsible>, "asChild" | "open"> &
  (
    | { open?: undefined }
    | {
        open: boolean
        onOpenChange: NonNullable<
          React.ComponentProps<typeof Collapsible>["onOpenChange"]
        >
      }
  )) {
  const contentId = React.useId()
  // The collapse animation must not run on first mount: an already-expanded
  // trace should appear without moving. Latch animation on once the open
  // state changes, whether from a user toggle or a programmatic (status)
  // change. The CSS animation selector requires this attribute.
  const [animate, setAnimate] = React.useState(false)
  const previousOpen = React.useRef(open)
  if (open !== undefined && previousOpen.current !== open) {
    previousOpen.current = open
    setAnimate(true)
  }
  const handleOpenChange = React.useCallback(
    (nextOpen: boolean) => {
      setAnimate(true)
      onOpenChange?.(nextOpen)
    },
    [onOpenChange],
  )

  return (
    <TraceCollapsibleIdContext.Provider value={contentId}>
      <Collapsible
        data-slot="trace-collapsible"
        data-animate={animate ? "true" : undefined}
        className={cn("flex w-full min-w-0 flex-col", className)}
        {...props}
        open={open}
        onOpenChange={handleOpenChange}
      >
        {children}
      </Collapsible>
    </TraceCollapsibleIdContext.Provider>
  )
}

function TraceCollapsibleTrigger({
  className,
  children,
  ...props
}: Omit<React.ComponentProps<typeof CollapsibleTrigger>, "asChild">) {
  const contentId = React.useContext(TraceCollapsibleIdContext)

  return (
    <CollapsibleTrigger
      data-slot="trace-collapsible-trigger"
      {...(contentId ? { "aria-controls": contentId } : {})}
      className={cn(
        "group/trace-collapsible-trigger flex w-fit max-w-full min-w-0 cursor-pointer items-center text-left text-text-secondary transition-colors hover:[&_[data-slot=trace-line]:not([data-status=error])]:text-text-primary focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-light",
        className,
      )}
      {...props}
    >
      {children}
      <TraceIcon className="ml-1 transition-transform group-aria-expanded/trace-collapsible-trigger:rotate-90 motion-reduce:transition-none">
        <ChevronRight className="h-3.5 w-3.5" />
      </TraceIcon>
    </CollapsibleTrigger>
  )
}

function TraceCollapsibleContent({
  className,
  children,
  ...props
}: Omit<React.ComponentProps<typeof CollapsibleContent>, "asChild">) {
  const contentId = React.useContext(TraceCollapsibleIdContext)

  return (
    <CollapsibleContent
      data-slot="trace-collapsible-content"
      {...(contentId ? { id: contentId } : {})}
      className="min-w-0 overflow-hidden"
      {...props}
    >
      <div
        data-slot="trace-collapsible-content-inner"
        className={cn(
          "min-w-0 break-words text-text-secondary",
          className,
        )}
      >
        {children}
      </div>
    </CollapsibleContent>
  )
}

export {
  Trace,
  TraceLine,
  TraceIcon,
  TraceTitle,
  TraceDescription,
  TraceCollapsible,
  TraceCollapsibleTrigger,
  TraceCollapsibleContent,
}
