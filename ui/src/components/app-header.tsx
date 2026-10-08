import { Check } from "lucide-react";
import Link from "next/link";

import { cn } from "@/lib/utils";

export const STAGES = [
  { key: "paste", label: "Paste JD" },
  { key: "match", label: "Match" },
  { key: "tailor", label: "Tailor" },
  { key: "review", label: "Review" },
  { key: "export", label: "Export" },
] as const;

export type Stage = (typeof STAGES)[number]["key"];

const PAGES = [
  { key: "tailor", label: "Tailor", href: "/" },
  { key: "applications", label: "Applications", href: "/applications" },
] as const;

/** Top bar: the product name, the two pages, and on the Tailor page where you are in its
 * five steps. Finished steps link to their section on the page. */
export function AppHeader({
  page,
  stage,
}: {
  page: (typeof PAGES)[number]["key"];
  stage?: Stage;
}) {
  return (
    <header className="sticky top-0 z-20 border-b bg-card/95 backdrop-blur">
      <div className="mx-auto flex max-w-[1120px] flex-wrap items-center justify-between gap-3 px-6 py-3">
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2 text-base font-semibold tracking-tight">
            <span aria-hidden className="size-2.5 rounded-full bg-primary" />
            factfit
          </Link>
          <nav aria-label="Pages" className="flex gap-1 text-sm">
            {PAGES.map((p) => (
              <Link
                key={p.key}
                href={p.href}
                aria-current={p.key === page ? "page" : undefined}
                className={cn(
                  "rounded-md px-2.5 py-1.5",
                  p.key === page
                    ? "font-medium text-foreground underline decoration-primary decoration-2 underline-offset-[10px]"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {p.label}
              </Link>
            ))}
          </nav>
        </div>
        {stage && <Steps stage={stage} />}
      </div>
    </header>
  );
}

function Steps({ stage }: { stage: Stage }) {
  const current = STAGES.findIndex((s) => s.key === stage);
  return (
    <nav aria-label="Progress" className="flex flex-wrap gap-1 text-[13px]">
      {STAGES.map((s, i) => {
        const done = i < current;
        const here = i === current;
        const body = (
          <>
            {done ? <Check className="size-3.5" strokeWidth={3} /> : <span>{i + 1}</span>}
            {s.label}
          </>
        );
        const cls = cn(
          "inline-flex items-center gap-1.5 rounded-full px-3 py-1.5",
          done && "text-ok hover:bg-ok-soft",
          here && "bg-primary font-medium text-primary-foreground",
          !done && !here && "text-muted-foreground",
        );
        return done ? (
          <a key={s.key} href={`#${s.key}`} className={cls}>
            {body}
          </a>
        ) : (
          <span key={s.key} className={cls} aria-current={here ? "step" : undefined}>
            {body}
          </span>
        );
      })}
    </nav>
  );
}
