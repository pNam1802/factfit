"use client";

import { useEffect, useState } from "react";

import { MatchResults } from "@/components/match-results";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { ApiError, api, type Job, type Level, type Match, toApiError } from "@/lib/api";

type Step = "idle" | "parsing" | "matching" | "done";

const LEVELS: { value: "" | Level; label: string }[] = [
  { value: "", label: "Lowest level the job accepts" },
  { value: "intern", label: "Intern" },
  { value: "fresher", label: "Fresher" },
  { value: "junior", label: "Junior" },
  { value: "mid", label: "Mid" },
  { value: "senior", label: "Senior" },
];

export default function Home() {
  const [text, setText] = useState("");
  const [company, setCompany] = useState("");
  const [level, setLevel] = useState<"" | Level>("");
  const [step, setStep] = useState<Step>("idle");
  const [job, setJob] = useState<Job | null>(null);
  const [match, setMatch] = useState<Match | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const seconds = useElapsed(step === "parsing" || step === "matching");

  async function analyze() {
    setError(null);
    setMatch(null);
    setJob(null);
    try {
      setStep("parsing");
      const parsed = await api.POST("/jobs", {
        body: { text, company: company.trim() || null },
      });
      if (!parsed.data) throw toApiError(parsed.error, parsed.response.status);
      setJob(parsed.data);

      setStep("matching");
      const matched = await api.POST("/jobs/{job_id}/match", {
        params: { path: { job_id: parsed.data.id } },
        body: { level: level || null },
      });
      if (!matched.data) throw toApiError(matched.error, matched.response.status);
      setMatch(matched.data);
      setStep("done");
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError(String(e)));
      setStep("idle");
    }
  }

  const busy = step === "parsing" || step === "matching";

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-10">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold">factfit</h1>
        <p className="text-muted-foreground">
          Paste a job description to see which requirements your verified profile meets, and
          which bullets prove it.
        </p>
      </header>

      <form
        className="flex flex-col gap-3"
        onSubmit={(e) => {
          e.preventDefault();
          if (!busy && text.trim()) analyze();
        }}
      >
        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste the job description here (English or Vietnamese)"
          className="min-h-56"
          disabled={busy}
        />
        <div className="flex flex-col gap-3 sm:flex-row">
          <Input
            value={company}
            onChange={(e) => setCompany(e.target.value)}
            placeholder="Company (optional)"
            disabled={busy}
          />
          <select
            value={level}
            onChange={(e) => setLevel(e.target.value as "" | Level)}
            disabled={busy}
            aria-label="Your level"
            className="h-9 rounded-md border bg-transparent px-3 text-sm"
          >
            {LEVELS.map((l) => (
              <option key={l.value} value={l.value}>
                {l.label}
              </option>
            ))}
          </select>
          <Button type="submit" disabled={busy || !text.trim()}>
            {busy ? "Working…" : "Analyze"}
          </Button>
        </div>
      </form>

      {busy && (
        <p className="text-sm text-muted-foreground" aria-live="polite">
          {step === "parsing"
            ? "Reading the job description…"
            : `Checking your profile against ${job?.jd.must_have.length ?? 0} must-have requirements…`}{" "}
          {seconds}s
        </p>
      )}

      {error && (
        <div role="alert" className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          <p className="font-medium">{error.message}</p>
          {error.issues.length > 0 && (
            <ul className="mt-2 list-disc pl-5">
              {error.issues.slice(0, 8).map((i) => (
                <li key={i}>{i}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {job && match && <MatchResults job={job} match={match} />}
    </main>
  );
}

/** Seconds since `running` became true; resets when it turns false. */
function useElapsed(running: boolean): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!running) return;
    const started = Date.now();
    const id = setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 500);
    return () => {
      clearInterval(id);
      setSeconds(0);
    };
  }, [running]);
  return seconds;
}
