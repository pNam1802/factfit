"use client";

import { ChevronRight } from "lucide-react";

import { Panel, PanelTitle, StatusPill, type Tone } from "@/components/parts";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { Job, Match, Requirement } from "@/lib/api";
import { cn } from "@/lib/utils";

const STATUS: Record<Requirement["status"], { label: string; tone: Tone; bar: string }> = {
  met: { label: "Met", tone: "ok", bar: "bg-ok" },
  partial: { label: "Partial", tone: "warn", bar: "bg-warn" },
  missing: { label: "Missing", tone: "bad", bar: "bg-bad" },
  unverifiable: { label: "Can't verify", tone: "neutral", bar: "bg-input" },
};

const ORDER: Requirement["status"][] = ["met", "partial", "missing", "unverifiable"];

export function MatchResults({ job, match }: { job: Job; match: Match }) {
  const groups = [
    { title: "Must have", rows: match.requirements.filter((r) => r.bucket === "must") },
    { title: "Nice to have", rows: match.requirements.filter((r) => r.bucket === "nice") },
  ].filter((g) => g.rows.length > 0);

  return (
    <section id="match" className="flex flex-col gap-5">
      <h2 className="text-xl font-semibold tracking-tight">Match</h2>
      <div className="flex flex-wrap items-start gap-6">
        <aside className="flex flex-[1_1_300px] flex-col gap-4 lg:sticky lg:top-20">
          <Panel className="gap-5">
            <div className="flex flex-col gap-1">
              <span className="text-sm text-muted-foreground">
                {job.company ?? "Unknown company"} · judged as {match.level ?? "any level"}
              </span>
              <span className="text-lg leading-snug font-semibold">{job.title}</span>
            </div>
            <div className="flex items-center gap-4">
              <ScoreRing score={match.score} />
              <p className="text-sm text-muted-foreground">
                Score from must-have and nice-to-have requirements, weighted in code. The model
                only judges each requirement.
              </p>
            </div>
            {groups.map((g) => (
              <Breakdown key={g.title} title={g.title} rows={g.rows} />
            ))}
          </Panel>
          <p className="font-mono text-xs text-muted-foreground">
            {job.cached && match.cached ? "Served from cache, no LLM call" : "Fresh result"} ·
            profile {match.profile_version}
          </p>
        </aside>

        <div className="flex min-w-0 flex-[999_1_560px] flex-col gap-4">
          {groups.map((g) => (
            <Panel key={g.title} className="gap-0 p-0">
              <div className="flex items-baseline justify-between px-5 pt-4 pb-3">
                <PanelTitle>{g.title}</PanelTitle>
                <span className="text-sm text-muted-foreground">{g.rows.length} requirements</span>
              </div>
              <div className="flex flex-col divide-y border-t">
                {g.rows.map((r) => (
                  <RequirementRow key={r.id} req={r} />
                ))}
              </div>
            </Panel>
          ))}
          {match.excluded.length > 0 && (
            <p className="text-sm text-muted-foreground">
              {match.excluded.length} requirement(s) apply only to another level and were not
              judged.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

function ScoreRing({ score }: { score: number }) {
  const r = 34;
  const c = 2 * Math.PI * r;
  return (
    <div className="relative size-22 shrink-0" role="img" aria-label={`Match score ${score} of 100`}>
      <svg viewBox="0 0 80 80" className="size-full -rotate-90">
        <circle cx="40" cy="40" r={r} fill="none" strokeWidth="7" className="stroke-muted" />
        <circle
          cx="40"
          cy="40"
          r={r}
          fill="none"
          strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - score / 100)}
          className="stroke-primary transition-[stroke-dashoffset] duration-700"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl leading-none font-semibold tabular-nums">{score}</span>
        <span className="font-mono text-[10px] text-muted-foreground">/ 100</span>
      </div>
    </div>
  );
}

function Breakdown({ title, rows }: { title: string; rows: Requirement[] }) {
  const counts = ORDER.map((s) => ({ s, n: rows.filter((r) => r.status === s).length })).filter(
    (x) => x.n > 0,
  );
  return (
    <div className="flex flex-col gap-2 text-sm">
      <div className="flex justify-between">
        <span className="font-medium">{title}</span>
        <span className="text-muted-foreground tabular-nums">{rows.length}</span>
      </div>
      <div className="flex h-2 overflow-hidden rounded-full bg-muted">
        {counts.map(({ s, n }) => (
          <div key={s} className={STATUS[s].bar} style={{ width: `${(n / rows.length) * 100}%` }} />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
        {counts.map(({ s, n }) => (
          <span key={s} className="inline-flex items-center gap-1.5">
            <span className={cn("size-2 rounded-full", STATUS[s].bar)} />
            {n} {STATUS[s].label.toLowerCase()}
          </span>
        ))}
      </div>
    </div>
  );
}

function RequirementRow({ req }: { req: Requirement }) {
  const status = STATUS[req.status];
  return (
    <Collapsible>
      <CollapsibleTrigger
        className="group flex w-full items-start gap-3 px-5 py-3.5 text-left enabled:hover:bg-muted/50"
        disabled={req.evidence.length === 0}
      >
        <StatusPill tone={status.tone} className="mt-0.5 w-28 justify-center">
          {status.label}
        </StatusPill>
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <span className="font-medium">{req.text}</span>
          {req.any_of.length > 0 && (
            <span className="text-xs text-muted-foreground">One of: {req.any_of.join(", ")}</span>
          )}
          <span className="text-sm text-muted-foreground">{req.note}</span>
        </div>
        {req.evidence.length > 0 && (
          <span className="mt-1 inline-flex shrink-0 items-center gap-1 text-xs text-muted-foreground">
            {req.evidence.length} proof
            <ChevronRight className="size-4 transition-transform group-data-[panel-open]:rotate-90" />
          </span>
        )}
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ul className="flex flex-col gap-2 px-5 pb-4 sm:pl-[9.25rem]">
          {req.evidence.map((e) => (
            <li key={e.id} className="rounded-lg border-l-2 border-primary bg-muted/60 px-3 py-2 text-sm">
              <div className="font-mono text-xs text-muted-foreground">
                {e.where} · {e.id}
              </div>
              <div>{e.text}</div>
            </li>
          ))}
        </ul>
      </CollapsibleContent>
    </Collapsible>
  );
}
