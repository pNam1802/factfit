import { AlertTriangle, Check, CircleHelp, Minus, X } from "lucide-react";
import type { ComponentProps, ReactNode } from "react";

import type { ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";

/** White card with a thin border: the one container style used across the page. */
export function Panel({ className, ...props }: ComponentProps<"section">) {
  return (
    <section
      className={cn("flex flex-col gap-3.5 rounded-xl border bg-card p-5", className)}
      {...props}
    />
  );
}

export function PanelTitle({ className, ...props }: ComponentProps<"h2">) {
  return <h2 className={cn("text-[15px] font-semibold", className)} {...props} />;
}

export type Tone = "ok" | "warn" | "bad" | "neutral";

const TONE: Record<Tone, { pill: string; icon: ReactNode }> = {
  ok: { pill: "bg-ok-soft text-ok-strong", icon: <Check strokeWidth={3} /> },
  warn: { pill: "bg-warn-soft text-warn-strong", icon: <Minus strokeWidth={3} /> },
  bad: { pill: "bg-bad-soft text-bad-strong", icon: <X strokeWidth={3} /> },
  neutral: { pill: "bg-muted text-muted-foreground", icon: <CircleHelp /> },
};

/** Status label: icon and word, so it reads the same without colour. */
export function StatusPill({
  tone,
  children,
  className,
}: {
  tone: Tone;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex h-6 w-fit shrink-0 items-center gap-1 rounded-full px-2.5 text-xs font-medium whitespace-nowrap [&_svg]:size-3",
        TONE[tone].pill,
        className,
      )}
    >
      {TONE[tone].icon}
      {children}
    </span>
  );
}

/** A soft box for one warning or problem, as in the ATS list. */
export function Note({ tone, children }: { tone: "warn" | "bad"; children: ReactNode }) {
  return (
    <div
      className={cn(
        "rounded-lg px-3.5 py-3 text-sm leading-relaxed",
        tone === "warn" ? "bg-warn-soft text-warn-strong" : "bg-bad-soft text-bad-strong",
      )}
    >
      {children}
    </div>
  );
}

export function ErrorBox({ error }: { error: ApiError }) {
  return (
    <div
      role="alert"
      className="flex gap-3 rounded-xl border border-bad-line bg-bad-soft p-4 text-sm text-bad-strong"
    >
      <AlertTriangle className="mt-0.5 size-4 shrink-0" />
      <div>
        <p className="font-medium">{error.message}</p>
        {error.issues.length > 0 && (
          <ul className="mt-2 list-disc pl-5">
            {error.issues.slice(0, 8).map((i) => (
              <li key={i}>{i}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={cn(
        "inline-block size-4 animate-spin rounded-full border-2 border-current border-t-transparent",
        className,
      )}
    />
  );
}
