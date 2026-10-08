"use client";

import { useState } from "react";

import { Search } from "lucide-react";

import { AppHeader, type Stage } from "@/components/app-header";
import { MatchResults } from "@/components/match-results";
import { ErrorBox, Panel, PanelTitle, Spinner } from "@/components/parts";
import { TailorReview } from "@/components/tailor-review";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { ApiError, api, type Job, type Level, type Match, toApiError } from "@/lib/api";
import { useElapsed } from "@/lib/use-elapsed";

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
  const [tailorStage, setTailorStage] = useState<Stage>("tailor");
  const seconds = useElapsed(step === "parsing" || step === "matching");

  async function analyze() {
    setError(null);
    setMatch(null);
    setJob(null);
    setTailorStage("tailor");
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
  const stage: Stage = match ? tailorStage : busy ? "match" : "paste";

  return (
    <>
      <AppHeader page="tailor" stage={stage} />
      <main className="mx-auto flex w-full max-w-[1120px] flex-col gap-10 px-6 pt-10 pb-16">
        <section id="paste" className="flex flex-col gap-5">
          <div className="flex flex-col gap-1.5">
            <h1 className="text-2xl font-semibold tracking-tight">Tailor your CV to a job</h1>
            <p className="max-w-2xl text-muted-foreground">
              Paste a job description to see which requirements your verified profile meets, and
              which bullets prove it. Every line on the CV traces back to your profile.
            </p>
          </div>

          <form
            className="flex flex-wrap items-start gap-6"
            onSubmit={(e) => {
              e.preventDefault();
              if (!busy && text.trim()) analyze();
            }}
          >
            <Panel className="min-w-0 flex-[999_1_560px] gap-2">
              <label htmlFor="jd" className="text-[15px] font-semibold">
                Job description
              </label>
              <Textarea
                id="jd"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste the job description here (English or Vietnamese)"
                className="min-h-64 bg-background/40 text-[15px] leading-relaxed"
                disabled={busy}
              />
              <p className="font-mono text-xs text-muted-foreground">
                {text.trim() ? `${words(text)} words` : "Nothing pasted yet"}
              </p>
            </Panel>

            <aside className="flex flex-[1_1_300px] flex-col gap-4">
              <Panel>
                <PanelTitle>Options</PanelTitle>
                <label className="flex flex-col gap-1.5 text-sm">
                  <span className="text-muted-foreground">Company</span>
                  <Input
                    value={company}
                    onChange={(e) => setCompany(e.target.value)}
                    placeholder="Optional"
                    disabled={busy}
                    className="h-10"
                  />
                </label>
                <label className="flex flex-col gap-1.5 text-sm">
                  <span className="text-muted-foreground">Your level</span>
                  <select
                    value={level}
                    onChange={(e) => setLevel(e.target.value as "" | Level)}
                    disabled={busy}
                    className="h-10 rounded-lg border border-input bg-card px-3 text-sm"
                  >
                    {LEVELS.map((l) => (
                      <option key={l.value} value={l.value}>
                        {l.label}
                      </option>
                    ))}
                  </select>
                </label>
              </Panel>
              <Button type="submit" size="lg" disabled={busy || !text.trim()}>
                {busy ? <Spinner /> : <Search />}
                {busy ? "Working…" : "Analyze"}
              </Button>
              {busy && (
                <p className="text-sm text-muted-foreground" aria-live="polite">
                  {step === "parsing"
                    ? "Reading the job description…"
                    : `Checking your profile against ${job?.jd.must_have.length ?? 0} must-have requirements…`}{" "}
                  <span className="font-mono">{seconds}s</span>
                </p>
              )}
              <p className="font-mono text-xs text-muted-foreground">
                Parse and match are cached per job text: pasting the same JD again is free.
              </p>
            </aside>
          </form>

          {error && <ErrorBox error={error} />}
        </section>

        {job && match && <MatchResults job={job} match={match} />}
        {job && match && (
          <TailorReview
            key={job.id}
            jobId={job.id}
            level={level || null}
            onStage={setTailorStage}
          />
        )}
      </main>
    </>
  );
}

function words(text: string): number {
  return text.trim().split(/\s+/).length;
}
