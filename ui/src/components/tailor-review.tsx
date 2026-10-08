"use client";

import { type ReactNode, useCallback, useEffect, useRef, useState } from "react";

import {
  ArrowRight,
  Check,
  Circle,
  Download,
  FileCode,
  FileText,
  RefreshCw,
  Sparkles,
} from "lucide-react";

import type { Stage } from "@/components/app-header";
import {
  ErrorBox,
  Note,
  Panel,
  PanelTitle,
  Spinner,
  StatusPill,
  type Tone,
} from "@/components/parts";
import { Button } from "@/components/ui/button";
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
import { useElapsed } from "@/lib/use-elapsed";
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

type Props = {
  jobId: number;
  level: Level | null;
  onStage: (stage: Stage) => void; // tells the header which step is current
};

export function TailorReview({ jobId, level, onStage }: Props) {
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
  const [built, setBuilt] = useState<Built | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const cards = useRef<(HTMLDivElement | null)[]>([]);
  const seconds = useElapsed(phase === "running");

  const stage: Stage =
    phase === "done" ? "export" : phase === "review" || phase === "submitting" ? "review" : "tailor";
  useEffect(() => onStage(stage), [stage, onStage]);

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
    const started = Date.now();
    const res = await api.POST("/runs/{run_id}/render", {
      params: { path: { run_id: runId } },
      body: {},
    });
    setRendering(false);
    if (!res.data) setError(toApiError(res.error, res.response.status));
    else {
      setRendered(res.data);
      setPdfVersion((v) => v + 1);
      setBuilt({ at: new Date(), seconds: (Date.now() - started) / 1000 });
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
      <section id="tailor" className="flex flex-col gap-4">
        <Panel className="flex-row flex-wrap items-center justify-between gap-5 p-6">
          <div className="flex items-start gap-4">
            <span className="flex size-10 shrink-0 items-center justify-center rounded-full bg-accent text-accent-foreground">
              <Sparkles className="size-5" />
            </span>
            <div className="flex max-w-xl flex-col gap-1">
              <PanelTitle className="text-lg">Tailor the CV for this job</PanelTitle>
              <p className="text-sm text-muted-foreground">
                Picks your strongest bullets, rewrites them toward this job, and checks every
                rewrite against your profile. Anything that fails keeps your original words.
              </p>
            </div>
          </div>
          <Button size="lg" onClick={start}>
            Tailor my CV
            <ArrowRight />
          </Button>
        </Panel>
        {error && <ErrorBox error={error} />}
      </section>
    );
  }

  if (phase === "running") {
    const current = STEPS.find((s) => !steps.includes(s));
    return (
      <section id="tailor">
        <Panel className="p-6">
          <div className="flex items-baseline justify-between gap-3">
            <PanelTitle className="text-lg">Tailoring your CV…</PanelTitle>
            <span className="font-mono text-sm text-muted-foreground">
              {seconds}s · usually 15–60s
            </span>
          </div>
          <ol className="flex flex-col gap-2.5 text-sm" aria-live="polite">
            {STEPS.map((s) => {
              const done = steps.includes(s);
              const now = s === current;
              return (
                <li
                  key={s}
                  className={cn(
                    "flex items-center gap-3",
                    done
                      ? "text-foreground"
                      : now
                        ? "font-medium text-foreground"
                        : "text-muted-foreground",
                  )}
                >
                  <span className="flex size-5 items-center justify-center">
                    {done ? (
                      <Check className="size-4 text-ok" strokeWidth={3} />
                    ) : now ? (
                      <Spinner className="size-3.5 text-primary" />
                    ) : (
                      <Circle className="size-3" />
                    )}
                  </span>
                  {STEP_LABEL[s]}
                </li>
              );
            })}
          </ol>
          {steps.includes("fallback") && (
            <p className="text-sm text-warn">
              Some rewrites failed the checks twice and fell back to your original words.
            </p>
          )}
        </Panel>
      </section>
    );
  }

  const decided = drafts.filter((d) => choices[d.source_id]).length;
  const reviewing = phase !== "done";
  const counts = {
    passed: drafts.filter((d) => d.passed && !d.fallback && d.attempts <= 1).length,
    retried: drafts.filter((d) => d.passed && !d.fallback && d.attempts > 1).length,
    kept: drafts.filter((d) => d.fallback).length,
    failed: drafts.filter((d) => !d.passed).length,
  };

  return (
    <>
      <section id="review" className="flex flex-col gap-5">
        <div className="flex flex-col gap-1">
          <h2 className="text-xl font-semibold tracking-tight">
            {reviewing ? "Review each bullet" : "Reviewed CV"}
          </h2>
          <p className="text-sm text-muted-foreground">
            {reviewing
              ? "Green words are new, struck-through words are gone. Bullets you leave undecided are accepted."
              : "Your decisions are saved. Build the PDF below."}
          </p>
        </div>

        <div className="flex flex-wrap items-start gap-6">
          <aside className="flex flex-[1_1_280px] flex-col gap-4 lg:sticky lg:top-20">
            <Panel>
              <PanelTitle>Progress</PanelTitle>
              <div className="flex flex-col gap-2">
                <div className="flex justify-between text-sm">
                  <span>{reviewing ? "Decided" : "Bullets"}</span>
                  <span className="font-medium tabular-nums">
                    {reviewing ? `${decided} of ${drafts.length}` : drafts.length}
                  </span>
                </div>
                {reviewing && (
                  <div className="h-2 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full bg-primary transition-[width]"
                      style={{ width: `${drafts.length ? (decided / drafts.length) * 100 : 0}%` }}
                    />
                  </div>
                )}
              </div>
              <CountRow label="Passed checks" n={counts.passed} tone="ok" />
              <CountRow label="Passed after a retry" n={counts.retried} tone="ok" />
              <CountRow label="Original kept" n={counts.kept} tone="warn" />
              {counts.failed > 0 && <CountRow label="Edits refused" n={counts.failed} tone="bad" />}
            </Panel>

            {reviewing && (
              <>
                <Panel className="gap-2.5 text-sm">
                  <PanelTitle>Keys</PanelTitle>
                  <KeyRow keys={["J", "K"]} label="Next / previous bullet" />
                  <KeyRow keys={["A"]} label="Accept the rewrite" />
                  <KeyRow keys={["E"]} label="Edit it yourself" />
                  <KeyRow keys={["R"]} label="Keep the original" />
                </Panel>
                <Button
                  size="lg"
                  onClick={submit}
                  disabled={phase === "submitting" || editing !== null}
                >
                  {phase === "submitting" ? <Spinner /> : <Check />}
                  {phase === "submitting" ? "Checking your edits…" : "Finish review"}
                </Button>
              </>
            )}
          </aside>

          <div className="flex min-w-0 flex-[999_1_560px] flex-col gap-3">
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
          </div>
        </div>
        {reviewing && error && <ErrorBox error={error} />}
      </section>

      {!reviewing && (
        <section id="export" className="flex flex-col gap-5">
          <h2 className="text-xl font-semibold tracking-tight">Export</h2>
          {error && <ErrorBox error={error} />}
          {rendered ? (
            <ExportView
              rendered={rendered}
              version={pdfVersion}
              built={built}
              rendering={rendering}
              onRebuild={render}
            />
          ) : (
            <Panel className="flex-row flex-wrap items-center justify-between gap-5 p-6">
              <div className="flex items-start gap-4">
                <span className="flex size-10 shrink-0 items-center justify-center rounded-full bg-accent text-accent-foreground">
                  <FileText className="size-5" />
                </span>
                <div className="flex max-w-xl flex-col gap-1">
                  <PanelTitle className="text-lg">Build the PDF</PanelTitle>
                  <p className="text-sm text-muted-foreground">
                    Fills your Overleaf template, compiles it, then reads the PDF back the way an
                    ATS would to check nothing got lost.
                  </p>
                </div>
              </div>
              <Button size="lg" onClick={render} disabled={rendering}>
                {rendering ? <Spinner /> : <FileText />}
                {rendering ? "Building the PDF…" : "Build PDF"}
              </Button>
            </Panel>
          )}
        </section>
      )}
    </>
  );
}

// --- pieces ------------------------------------------------------------------------------------

function CountRow({ label, n, tone }: { label: string; n: number; tone: Tone }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span>{label}</span>
      <span
        className={cn(
          "font-medium tabular-nums",
          n === 0 && "text-muted-foreground",
          n > 0 && tone === "ok" && "text-ok",
          n > 0 && tone === "warn" && "text-warn",
          n > 0 && tone === "bad" && "text-bad",
        )}
      >
        {n}
      </span>
    </div>
  );
}

function KeyRow({ keys, label }: { keys: string[]; label: string }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-muted-foreground">{label}</span>
      <span className="flex gap-1">
        {keys.map((k) => (
          <kbd key={k}>{k}</kbd>
        ))}
      </span>
    </div>
  );
}

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

function draftStatus(d: Draft): { label: string; tone: Tone; edge: string } {
  if (!d.passed) return { label: "Your edit failed checks", tone: "bad", edge: "border-l-bad" };
  if (d.fallback)
    return { label: "Original kept: rewrites failed checks", tone: "warn", edge: "border-l-warn" };
  if (d.attempts > 1) return { label: "Passed after one retry", tone: "ok", edge: "border-l-ok" };
  return { label: "Passed checks", tone: "ok", edge: "border-l-ok" };
}

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
  const status = draftStatus(d);
  const unchanged = d.original === shown;

  return (
    <div
      ref={cardRef}
      onClick={onFocus}
      className={cn(
        "flex flex-col gap-3 rounded-xl border border-l-4 bg-card p-4 transition-shadow",
        status.edge,
        focused && "ring-2 ring-primary/60",
        errors && "border-bad-line",
      )}
    >
      <div className="flex flex-wrap items-center gap-2">
        <StatusPill tone={status.tone}>{status.label}</StatusPill>
        {choice && <ChoiceTag choice={choice} />}
        <span className="ml-auto font-mono text-xs text-muted-foreground">
          {d.entry_id} · {d.source_id}
        </span>
      </div>

      {!unchanged && (
        <div className="flex flex-col gap-1">
          <Label>Original</Label>
          <p className="text-sm text-muted-foreground">
            <Words tokens={diff.before} />
          </p>
        </div>
      )}
      <div className="flex flex-col gap-1">
        <Label>
          {unchanged ? "Bullet (unchanged)" : choice?.action === "edit" ? "Your edit" : "Rewrite"}
        </Label>
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
              className="text-[15px]"
            />
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" onClick={onSave}>
                Save
              </Button>
              <Button size="sm" variant="outline" onClick={onCancel}>
                Cancel
              </Button>
              <span className="text-xs text-muted-foreground">
                <kbd>Ctrl</kbd> + <kbd>Enter</kbd> saves, <kbd>Esc</kbd> cancels. Your edit goes
                through the same checks.
              </span>
            </div>
          </div>
        ) : (
          <p className="text-[15px] leading-relaxed">
            <Words tokens={diff.after} />
          </p>
        )}
      </div>

      {errors && (
        <Note tone="bad">
          <strong className="font-semibold">Your edit was refused</strong>
          <ul className="mt-1 list-disc pl-5">
            {errors.map((e) => (
              <li key={e}>{e}</li>
            ))}
          </ul>
        </Note>
      )}

      {d.history.length > 0 && (
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer select-none">
            Rejected rewrites ({d.history.length})
          </summary>
          <ul className="mt-2 flex flex-col gap-1.5">
            {d.history.map(([text, problems], i) => (
              <li key={i} className="rounded-md bg-muted/60 px-2.5 py-1.5">
                “{String(text)}”
                <span className="block text-bad">{(problems as string[]).join("; ")}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      {!readOnly && !editing && (
        <div className="flex flex-wrap gap-2">
          <ActionButton
            active={choice?.action === "accept"}
            onClick={() => onChoose({ action: "accept" })}
            k="A"
          >
            Accept
          </ActionButton>
          <ActionButton active={choice?.action === "edit"} onClick={onEdit} k="E">
            Edit
          </ActionButton>
          <ActionButton
            active={choice?.action === "reject"}
            onClick={() => onChoose({ action: "reject" })}
            k="R"
          >
            Keep original
          </ActionButton>
        </div>
      )}
    </div>
  );
}

function Label({ children }: { children: ReactNode }) {
  return (
    <span className="text-[11px] font-medium tracking-wider text-muted-foreground uppercase">
      {children}
    </span>
  );
}

function ActionButton({
  active,
  onClick,
  k,
  children,
}: {
  active: boolean;
  onClick: () => void;
  k: string;
  children: ReactNode;
}) {
  return (
    <Button
      size="sm"
      variant="outline"
      onClick={onClick}
      className={cn("h-8 gap-2", active && "border-primary bg-accent text-accent-foreground")}
    >
      {children}
      <kbd className="h-4 min-w-4 text-[10px]">{k}</kbd>
    </Button>
  );
}

function ChoiceTag({ choice }: { choice: Choice }) {
  const label =
    choice.action === "accept"
      ? "Accepted"
      : choice.action === "reject"
        ? "Keeping original"
        : "Edited";
  return (
    <span className="inline-flex h-6 items-center rounded-full border border-primary/40 px-2.5 text-xs font-medium text-accent-foreground">
      {label}
    </span>
  );
}

function Words({ tokens }: { tokens: Token[] }) {
  return (
    <>
      {tokens.map((t, i) => (
        <span
          key={i}
          className={cn(
            t.kind === "added" && "rounded bg-ok-soft px-0.5 text-ok-strong",
            t.kind === "removed" && "line-through decoration-bad/70",
          )}
        >
          {t.word}{" "}
        </span>
      ))}
    </>
  );
}

type Built = { at: Date; seconds: number };

function ExportView({
  rendered,
  version,
  built,
  rendering,
  onRebuild,
}: {
  rendered: Rendered;
  version: number;
  built: Built | null;
  rendering: boolean;
  onRebuild: () => void;
}) {
  const ok = rendered.ats_ok;
  const pdf = `/api${rendered.pdf_url}`;
  const summary = [
    `${rendered.pages} page${rendered.pages === 1 ? "" : "s"}`,
    rendered.issues.length ? `${rendered.issues.length} problem(s) to fix` : "no errors",
    rendered.warnings.length
      ? `${rendered.warnings.length} ATS warning(s) worth a look`
      : "no ATS warnings",
  ].join(" · ");

  return (
    <div className="flex flex-col gap-6">
      <div
        className={cn(
          "flex flex-wrap items-center justify-between gap-4 rounded-xl border px-6 py-5",
          ok ? "border-ok-line bg-ok-soft" : "border-bad-line bg-bad-soft",
        )}
      >
        <div className="flex items-center gap-3.5">
          <span
            className={cn(
              "flex size-9 shrink-0 items-center justify-center rounded-full text-white",
              ok ? "bg-ok" : "bg-bad",
            )}
          >
            {ok ? (
              <Check className="size-[18px]" strokeWidth={3} />
            ) : (
              <span className="text-lg font-bold">!</span>
            )}
          </span>
          <div className="flex flex-col gap-0.5">
            <span className={cn("text-xl font-semibold", ok ? "text-ok-strong" : "text-bad-strong")}>
              {ok ? "Ready to send" : "Do not send yet"}
            </span>
            <span className={cn("text-sm", ok ? "text-ok" : "text-bad")}>{summary}</span>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <a
            href={pdf}
            download
            className={cn(
              "inline-flex h-11 items-center gap-2 rounded-lg px-4.5 text-sm font-semibold text-white",
              ok ? "bg-ok hover:bg-ok-strong" : "bg-bad hover:bg-bad-strong",
            )}
          >
            <Download className="size-4" />
            Download .pdf
          </a>
          <a
            href={`/api${rendered.tex_url}`}
            download
            className={cn(
              "inline-flex h-11 items-center gap-2 rounded-lg border bg-card px-4.5 text-sm font-medium",
              ok ? "border-ok-line text-ok-strong" : "border-bad-line text-bad-strong",
            )}
          >
            <FileCode className="size-4" />
            .tex source
          </a>
        </div>
      </div>

      <div className="flex flex-wrap items-start gap-6">
        <aside className="flex flex-[1_1_320px] flex-col gap-4">
          <Panel>
            <PanelTitle>Checks</PanelTitle>
            <CheckRow label="Page count" ok={rendered.pages === 1}>
              {rendered.pages} of 1
            </CheckRow>
            <CheckRow label="Compile (tectonic)" ok>
              OK{built && ` · ${built.seconds.toFixed(1)}s`}
            </CheckRow>
            <CheckRow label="ATS errors" ok={rendered.issues.length === 0}>
              {rendered.issues.length}
            </CheckRow>
            <CheckRow label="ATS warnings" ok={rendered.warnings.length === 0} soft>
              {rendered.warnings.length}
            </CheckRow>
          </Panel>

          {rendered.issues.length > 0 && (
            <Panel className="gap-3">
              <PanelTitle>Fix before sending</PanelTitle>
              {rendered.issues.map((i) => (
                <Note key={i} tone="bad">
                  {i}
                </Note>
              ))}
            </Panel>
          )}
          {rendered.warnings.length > 0 && (
            <Panel className="gap-3">
              <PanelTitle>ATS warnings</PanelTitle>
              {rendered.warnings.map((w) => (
                <Note key={w} tone="warn">
                  {w}
                </Note>
              ))}
            </Panel>
          )}

          <Button size="lg" variant="outline" onClick={onRebuild} disabled={rendering}>
            {rendering ? <Spinner /> : <RefreshCw />}
            {rendering ? "Building…" : "Rebuild PDF"}
          </Button>
          <p className="font-mono text-xs text-muted-foreground">
            Template: your Overleaf CV
            {built &&
              ` · built ${built.at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`}
          </p>
        </aside>

        <section
          aria-label="PDF preview"
          className="flex min-w-0 flex-[999_1_560px] justify-center rounded-xl bg-stage p-4 sm:p-8"
        >
          <PdfFrame src={`${pdf}?v=${version}`} />
        </section>
      </div>
    </div>
  );
}

const A4_WIDTH_PX = 794; // 210 mm at 96 dpi: the PDF viewer's width at 100% zoom

/** The PDF page, scaled to fill the frame. Chrome's viewer ignores "fit" hints in the URL
 * but honours a numeric zoom, so the zoom follows the frame's width. */
function PdfFrame({ src }: { src: string }) {
  const [zoom, setZoom] = useState<number | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const fit = () => setZoom(Math.floor(((el.clientWidth - 8) / A4_WIDTH_PX) * 100));
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return (
    <div
      ref={ref}
      className="aspect-[210/297] w-full max-w-[640px] bg-white shadow-[0_4px_16px_rgba(0,0,0,0.12)]"
    >
      {zoom && (
        <iframe
          title="CV preview"
          src={`${src}#toolbar=0&navpanes=0&zoom=${zoom}`}
          className="size-full border-0"
        />
      )}
    </div>
  );
}

function CheckRow({
  label,
  ok,
  soft,
  children,
}: {
  label: string;
  ok: boolean;
  soft?: boolean; // a failure here is a warning, not a blocker
  children: ReactNode;
}) {
  return (
    <div className="flex justify-between text-sm">
      <span>{label}</span>
      <span
        className={cn("font-medium", ok ? "text-ok" : soft ? "text-warn" : "text-bad")}
      >
        {ok ? "✓" : soft ? "!" : "✕"} {children}
      </span>
    </div>
  );
}
