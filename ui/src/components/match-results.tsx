"use client";

import { ChevronRight } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { Job, Match, Requirement } from "@/lib/api";
import { cn } from "@/lib/utils";

const STATUS: Record<Requirement["status"], { label: string; className: string }> = {
  met: { label: "Met", className: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300" },
  partial: { label: "Partial", className: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300" },
  missing: { label: "Missing", className: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300" },
  unverifiable: {
    label: "Can't verify",
    className: "bg-muted text-muted-foreground",
  },
};

export function MatchResults({ job, match }: { job: Job; match: Match }) {
  const groups = [
    { title: "Must have", rows: match.requirements.filter((r) => r.bucket === "must") },
    { title: "Nice to have", rows: match.requirements.filter((r) => r.bucket === "nice") },
  ].filter((g) => g.rows.length > 0);

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <CardDescription>
            {job.company ?? "Unknown company"} · judged as {match.level ?? "any level"}
          </CardDescription>
          <CardTitle className="text-xl">{job.title}</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap items-end gap-6">
          <div>
            <div className="text-4xl font-semibold tabular-nums">{match.score}</div>
            <div className="text-sm text-muted-foreground">match score / 100</div>
          </div>
          {groups.map((g) => (
            <Summary key={g.title} title={g.title} rows={g.rows} />
          ))}
        </CardContent>
      </Card>

      {groups.map((g) => (
        <section key={g.title} className="flex flex-col gap-2">
          <h2 className="text-sm font-medium text-muted-foreground">{g.title}</h2>
          <div className="flex flex-col divide-y rounded-lg border">
            {g.rows.map((r) => (
              <RequirementRow key={r.id} req={r} />
            ))}
          </div>
        </section>
      ))}

      {match.excluded.length > 0 && (
        <p className="text-sm text-muted-foreground">
          {match.excluded.length} requirement(s) apply only to another level and were not judged.
        </p>
      )}
      <p className="text-xs text-muted-foreground">
        {job.cached && match.cached ? "Served from cache, no LLM call." : "Fresh result."} Profile
        version {match.profile_version}.
      </p>
    </div>
  );
}

function Summary({ title, rows }: { title: string; rows: Requirement[] }) {
  const count = (s: Requirement["status"]) => rows.filter((r) => r.status === s).length;
  return (
    <div className="text-sm">
      <div className="font-medium">{title}</div>
      <div className="text-muted-foreground">
        {count("met")} met · {count("partial")} partial · {count("missing")} missing
        {count("unverifiable") > 0 && ` · ${count("unverifiable")} can't verify`}
      </div>
    </div>
  );
}

function RequirementRow({ req }: { req: Requirement }) {
  const status = STATUS[req.status];
  return (
    <Collapsible>
      <CollapsibleTrigger
        className="group flex w-full items-start gap-3 p-3 text-left hover:bg-muted/50"
        disabled={req.evidence.length === 0}
      >
        <Badge className={cn("mt-0.5 w-24 shrink-0", status.className)}>{status.label}</Badge>
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <span>{req.text}</span>
          {req.any_of.length > 0 && (
            <span className="text-xs text-muted-foreground">One of: {req.any_of.join(", ")}</span>
          )}
          <span className="text-sm text-muted-foreground">{req.note}</span>
        </div>
        {req.evidence.length > 0 && (
          <ChevronRight className="mt-1 size-4 shrink-0 text-muted-foreground transition-transform group-data-[panel-open]:rotate-90" />
        )}
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ul className="flex flex-col gap-2 px-3 pb-3 pl-[7.75rem]">
          {req.evidence.map((e) => (
            <li key={e.id} className="rounded-md bg-muted/50 p-2 text-sm">
              <div className="text-xs text-muted-foreground">
                {e.where} · <code>{e.id}</code>
              </div>
              <div>{e.text}</div>
            </li>
          ))}
        </ul>
      </CollapsibleContent>
    </Collapsible>
  );
}
