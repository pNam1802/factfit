"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import {
  ApiError,
  api,
  type Draft,
  type Level,
  type Rendered,
  type ReviewDecision,
  type RunStatus,
  toApiError,
} from "@/lib/api";
import { type Token, wordDiff } from "@/lib/diff";
import { cn } from "@/lib/utils";

type Phase = "idle" | "running" | "review" | "submitting" | "done";
type Choice = { action: "accept" | "edit" | "reject"; text?: string };

const STEPS = ["parse", "match", "select", "rewrite", "check", "review"];
const STEP_LABEL: Record<string, string> = {
  parse: "Read the job",
  match: "Match your profile",
  select: "Pick bullets",
  rewrite: "Rewrite",
  check: "Check for invented claims",
  fallback: "Fall back to originals",
  review: "Ready for your review",
};

export function TailorReview({ jobId, level }: { jobId: number; level: Level | null }) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [runId, setRunId] = useState<string | null>(null);
  const [steps, setSteps] = useState<string[]>([]);
  const [drafts, setDrafts] = useState<Draft[]>([]);
  const [reviewErrors, setReviewErrors] = useState<Record<string, string[]>>({});
  const [choices, setChoices] = useState<Record<string, Choice>>({});
  const [focus, setFocus] = useState(0);
  const [editing, setEditing] = useState<string | null>(null);
  const [editText, setEditText] = useState("");
  const [rendered, setRendered] = useState<Rendered | null>(null);
  const [rendering, setRendering] = useState(false);
  const [pdfVersion, setPdfVersion] = useState(0); // bumps on each build, reloads the preview
  const [error, setError] = useState<ApiError | null>(null);
  const cards = useRef<(HTMLDivElement | null)[]>([]);

  // --- running ----------------------------------------------------------------------------

  async function start() {
    setError(null);
    setRendered(null);
    setChoices({});
    setSteps([]);
    const res = await api.POST("/jobs/{job_id}/tailor", {
      params: { path: { job_id: jobId } },
      body: { level: level ?? null, use_judge: true },
    });
    if (!res.data) {
      setError(toApiError(res.error, res.response.status));
      return;
    }
    setRunId(res.data.run_id);
    setPhase("running");
  }

  // Progress: one event per finished step (SSE), plus polling the status as a fallback.
  useEffect(() => {
    if (phase !== "running" || !runId) return;
    const events = new EventSource(`/api/runs/${runId}/events`);
    events.onmessage = (e) => {
      const node = JSON.parse(e.data).node as string;
      if (node !== "__end__") setSteps((s) => (s.includes(node) ? s : [...s, node]));
    };
    const timer = setInterval(async () => {
      const res = await api.GET("/runs/{run_id}", { params: { path: { run_id: runId } } });
      if (res.data && res.data.status !== "running" && res.data.status !== "not_found") {
        clearInterval(timer);
        events.close();
        applyStatus(res.data);
      }
    }, 1000);
    return () => {
      clearInterval(timer);
      events.close();
    };
  }, [phase, runId]);

  function applyStatus(status: RunStatus) {
    if (status.status === "error") {
      setError(new ApiError(status.error ?? "The run failed."));
      setPhase("idle");
      return;
    }
    setDrafts(status.drafts);
    setReviewErrors(status.review_errors);
    setPhase(status.status === "done" ? "done" : "review");
  }

  async function submit() {
    if (!runId) return;
    setPhase("submitting");
    const decisions: ReviewDecision[] = drafts.map((d) => {
      const c = choices[d.source_id] ?? { action: "accept" };
      return { source_id: d.source_id, action: c.action, text: c.text ?? null };
    });
    const res = await api.POST("/runs/{run_id}/review", {
      params: { path: { run_id: runId } },
      body: { decisions },
    });
    if (!res.data) {
      setError(toApiError(res.error, res.response.status));
      setPhase("review");
      return;
    }
    applyStatus(res.data);
    if (res.data.status === "waiting_review") {
      // Refused edits come back with reasons; keep the person's other choices.
      setChoices((c) => {
        const next = { ...c };
        for (const id of Object.keys(res.data.review_errors)) delete next[id];
        return next;
      });
    }
  }

  async function render() {
    if (!runId) return;
    setRendering(true);
    setError(null);
    const res = await api.POST("/runs/{run_id}/render", {
      params: { path: { run_id: runId } },
      body: {},
    });
    setRendering(false);
    if (!res.data) setError(toApiError(res.error, res.response.status));
    else {
      setRendered(res.data);
      setPdfVersion((v) => v + 1);
    }
  }

  // --- review choices and keyboard ---------------------------------------------------------

  const choose = useCallback((id: string, choice: Choice) => {
    setChoices((c) => ({ ...c, [id]: choice }));
  }, []);

  const startEdit = useCallback(
    (d: Draft) => {
      setEditing(d.source_id);
      setEditText(choices[d.source_id]?.text ?? d.text);
    },
    [choices],
  );

  function saveEdit(id: string) {
    const text = editText.trim();
    if (text) choose(id, { action: "edit", text });
    setEditing(null);
  }

  useEffect(() => {
    if (phase !== "review") return;
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement;
      if (editing || target.tagName === "TEXTAREA" || target.tagName === "INPUT") return;
      const d = drafts[focus];
      if (!d) return;
      const key = e.key.toLowerCase();
      if (key === "j") setFocus((f) => Math.min(f + 1, drafts.length - 1));
      else if (key === "k") setFocus((f) => Math.max(f - 1, 0));
      else if (key === "a") choose(d.source_id, { action: "accept" });
      else if (key === "r") choose(d.source_id, { action: "reject" });
      else if (key === "e") startEdit(d);
      else return;
      e.preventDefault();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase, drafts, focus, editing, choose, startEdit]);

  useEffect(() => {
    cards.current[focus]?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [focus]);

  // --- view ---------------------------------------------------------------------------------

  if (phase === "idle") {
    return (
      <div className="flex flex-col gap-2">
        <Button onClick={start} className="self-start">
          Tailor my CV for this job
        </Button>
        {error && <ErrorBox error={error} />}
      </div>
    );
  }

  if (phase === "running") {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Tailoring your CV…</CardTitle>
          <CardDescription>Usually 15–60 seconds.</CardDescription>
        </CardHeader>
        <CardContent>
          <ol className="flex flex-col gap-1 text-sm" aria-live="polite">
            {STEPS.map((s) => (
              <li key={s} className={cn(steps.includes(s) ? "text-foreground" : "text-muted-foreground")}>
                {steps.includes(s) ? "✓" : "○"} {STEP_LABEL[s]}
              </li>
            ))}
          </ol>
        </CardContent>
      </Card>
    );
  }

  const decided = drafts.filter((d) => choices[d.source_id]).length;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-lg font-semibold">
          {phase === "done" ? "Reviewed CV" : "Review each bullet"}
        </h2>
        {phase !== "done" && (
          <p className="text-xs text-muted-foreground">
            J / K move · A accept · E edit · R keep original · {decided}/{drafts.length} decided
            (the rest are accepted)
          </p>
        )}
      </div>

      {drafts.map((d, i) => (
        <DraftCard
          key={d.source_id}
          cardRef={(el) => {
            cards.current[i] = el;
          }}
          draft={d}
          focused={phase === "review" && i === focus}
          choice={choices[d.source_id]}
          errors={reviewErrors[d.source_id]}
          readOnly={phase !== "review"}
          editing={editing === d.source_id}
          editText={editText}
          onFocus={() => setFocus(i)}
          onChoose={(c) => choose(d.source_id, c)}
          onEdit={() => startEdit(d)}
          onEditText={setEditText}
          onSave={() => saveEdit(d.source_id)}
          onCancel={() => setEditing(null)}
        />
      ))}

      {error && <ErrorBox error={error} />}

      {phase !== "done" ? (
        <Button onClick={submit} disabled={phase === "submitting" || editing !== null} className="self-start">
          {phase === "submitting" ? "Checking your edits…" : "Finish review"}
        </Button>
      ) : (
        <div className="flex flex-col gap-3">
          <Button onClick={render} disabled={rendering} className="self-start">
            {rendering ? "Building the PDF…" : rendered ? "Rebuild PDF" : "Build PDF"}
          </Button>
          {rendered && <RenderedView rendered={rendered} version={pdfVersion} />}
        </div>
      )}
    </div>
  );
}

// --- pieces ------------------------------------------------------------------------------------

type DraftCardProps = {
  cardRef: (el: HTMLDivElement | null) => void;
  draft: Draft;
  focused: boolean;
  choice?: Choice;
  errors?: string[];
  readOnly: boolean;
  editing: boolean;
  editText: string;
  onFocus: () => void;
  onChoose: (c: Choice) => void;
  onEdit: () => void;
  onEditText: (t: string) => void;
  onSave: () => void;
  onCancel: () => void;
};

function DraftCard({
  cardRef,
  draft,
  focused,
  choice,
  errors,
  readOnly,
  editing,
  editText,
  onFocus,
  onChoose,
  onEdit,
  onEditText,
  onSave,
  onCancel,
}: DraftCardProps) {
  const d = draft;
  const shown = choice?.action === "reject" ? d.original : (choice?.text ?? d.text);
  const diff = wordDiff(d.original, shown);
  const status = !d.passed
    ? { label: "Your edit failed checks", tone: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300" }
    : d.fallback
    ? { label: "Original kept: rewrites failed checks", tone: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300" }
    : d.attempts > 1
      ? { label: "Passed after one retry", tone: "bg-muted text-muted-foreground" }
      : { label: "Passed checks", tone: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300" };

  return (
    <div
      ref={cardRef}
      onClick={onFocus}
      className={cn(
        "flex flex-col gap-2 rounded-lg border p-3 transition-colors",
        focused && "border-foreground/40 ring-2 ring-ring/30",
        errors && "border-red-400",
      )}
    >
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <code>{d.source_id}</code>
        <span>· {d.entry_id}</span>
        <Badge className={status.tone}>{status.label}</Badge>
        {choice && <Badge variant="outline">{choiceLabel(choice)}</Badge>}
      </div>

      <p className="text-sm text-muted-foreground">
        <Words tokens={diff.before} />
      </p>
      {editing ? (
        <div className="flex flex-col gap-2">
          <Textarea
            autoFocus
            value={editText}
            onChange={(e) => onEditText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) onSave();
              if (e.key === "Escape") onCancel();
            }}
          />
          <p className="text-xs text-muted-foreground">
            Ctrl+Enter to save, Esc to cancel. Your edit goes through the same checks.
          </p>
        </div>
      ) : (
        <p>
          <Words tokens={diff.after} />
        </p>
      )}

      {errors && (
        <div role="alert" className="text-sm text-red-700 dark:text-red-300">
          Your edit was refused:
          <ul className="list-disc pl-5">
            {errors.map((e) => (
              <li key={e}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      {d.history.length > 0 && (
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer">Rejected rewrites ({d.history.length})</summary>
          <ul className="mt-1 flex flex-col gap-1">
            {d.history.map(([text, problems], i) => (
              <li key={i}>
                “{String(text)}” — {(problems as string[]).join("; ")}
              </li>
            ))}
          </ul>
        </details>
      )}

      {!readOnly && !editing && (
        <div className="flex gap-2">
          <Button size="sm" variant="outline" onClick={() => onChoose({ action: "accept" })}>
            Accept (A)
          </Button>
          <Button size="sm" variant="outline" onClick={onEdit}>
            Edit (E)
          </Button>
          <Button size="sm" variant="outline" onClick={() => onChoose({ action: "reject" })}>
            Keep original (R)
          </Button>
        </div>
      )}
    </div>
  );
}

function Words({ tokens }: { tokens: Token[] }) {
  return (
    <>
      {tokens.map((t, i) => (
        <span
          key={i}
          className={cn(
            t.kind === "added" && "rounded bg-emerald-100 dark:bg-emerald-950",
            t.kind === "removed" && "line-through decoration-red-500/70",
          )}
        >
          {t.word}{" "}
        </span>
      ))}
    </>
  );
}

function choiceLabel(c: Choice): string {
  return c.action === "accept" ? "accepted" : c.action === "reject" ? "keep original" : "edited";
}

function RenderedView({ rendered, version }: { rendered: Rendered; version: number }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <Badge
          className={
            rendered.ats_ok
              ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
              : "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300"
          }
        >
          {rendered.ats_ok ? "Ready to send" : "Do not send yet"}
        </Badge>
        <span className="text-muted-foreground">{rendered.pages} page(s)</span>
        <a className="underline" href={`/api${rendered.pdf_url}`} download>
          Download PDF
        </a>
        <a className="underline" href={`/api${rendered.tex_url}`} download>
          Download .tex
        </a>
      </div>
      {rendered.issues.length > 0 && (
        <ul className="list-disc pl-5 text-sm text-red-700 dark:text-red-300">
          {rendered.issues.map((i) => (
            <li key={i}>{i}</li>
          ))}
        </ul>
      )}
      {rendered.warnings.length > 0 && (
        <ul className="list-disc pl-5 text-xs text-muted-foreground">
          {rendered.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}
      <iframe
        title="CV preview"
        src={`/api${rendered.pdf_url}?v=${version}`}
        className="h-[85vh] w-full rounded-lg border"
      />
    </div>
  );
}

function ErrorBox({ error }: { error: ApiError }) {
  return (
    <div
      role="alert"
      className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200"
    >
      {error.message}
      {error.issues.length > 0 && (
        <ul className="mt-1 list-disc pl-5">
          {error.issues.slice(0, 8).map((i) => (
            <li key={i}>{i}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
