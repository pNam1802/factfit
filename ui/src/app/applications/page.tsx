"use client";

import { ExternalLink, FileText, Plus, X } from "lucide-react";
import { useEffect, useState } from "react";

import { AppHeader } from "@/components/app-header";
import { ErrorBox, Note, Panel, PanelTitle, Spinner, StatusPill } from "@/components/parts";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  type Application,
  ApiError,
  type ApplicationStatus,
  type ManualApplication,
  api,
  toApiError,
} from "@/lib/api";
import { CHANNELS, STATUSES } from "@/lib/statuses";
import { cn } from "@/lib/utils";

const REPLIED: ApplicationStatus[] = ["screening", "interview", "offer", "rejected"];

export default function ApplicationsPage() {
  const [apps, setApps] = useState<Application[] | null>(null);
  const [filter, setFilter] = useState<ApplicationStatus | null>(null);
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const [version, setVersion] = useState(0); // bump to load the list again
  const reload = () => setVersion((v) => v + 1);

  useEffect(() => {
    let current = true;
    api.GET("/applications").then((res) => {
      if (!current) return;
      if (res.data) setApps(res.data);
      else setError(toApiError(res.error, res.response.status));
    });
    return () => {
      current = false;
    };
  }, [version]);

  async function update(id: number, body: { status?: ApplicationStatus; notes?: string }) {
    const res = await api.PATCH("/applications/{application_id}", {
      params: { path: { application_id: id } },
      body,
    });
    if (!res.data) return setError(toApiError(res.error, res.response.status));
    const updated = res.data;
    setApps((list) => list?.map((a) => (a.id === id ? updated : a)) ?? null);
  }

  const shown = apps?.filter((a) => !filter || a.status === filter) ?? [];
  const sent = apps?.filter((a) => a.applied_at) ?? [];
  const replied = sent.filter((a) => REPLIED.includes(a.status));

  return (
    <>
      <AppHeader page="applications" />
      <main className="mx-auto flex w-full max-w-[1120px] flex-col gap-6 px-6 pt-10 pb-16">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div className="flex flex-col gap-1.5">
            <h1 className="text-2xl font-semibold tracking-tight">Applications</h1>
            <p className="max-w-2xl text-muted-foreground">
              Every application with the exact CV that was sent. Move each one along as you hear
              back; the dates are kept to measure response times.
            </p>
          </div>
          <Button size="lg" variant={adding ? "outline" : "default"} onClick={() => setAdding(!adding)}>
            {adding ? <X /> : <Plus />}
            {adding ? "Close" : "Add a manual application"}
          </Button>
        </div>

        {adding && (
          <ManualForm
            onSaved={() => {
              setAdding(false);
              reload();
            }}
          />
        )}
        {error && <ErrorBox error={error} />}

        {apps === null ? (
          <p className="flex items-center gap-2 text-muted-foreground">
            <Spinner /> Loading…
          </p>
        ) : apps.length === 0 ? (
          <Panel className="items-center gap-2 py-12 text-center">
            <PanelTitle>No applications yet</PanelTitle>
            <p className="max-w-md text-sm text-muted-foreground">
              Tailor a CV and press “Mark as applied” on the Export step, or add one you sent
              without factfit.
            </p>
          </Panel>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="Applications" value={apps.length} />
              <Stat label="Sent" value={sent.length} />
              <Stat
                label="Heard back"
                value={sent.length ? `${Math.round((100 * replied.length) / sent.length)}%` : "–"}
                hint={`${replied.length} of ${sent.length} sent`}
              />
              <Stat
                label="Interviews"
                value={apps.filter((a) => a.status === "interview" || a.status === "offer").length}
              />
            </div>

            <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by status">
              <FilterChip active={!filter} onClick={() => setFilter(null)}>
                All <span className="tabular-nums">{apps.length}</span>
              </FilterChip>
              {STATUSES.map((s) => {
                const n = apps.filter((a) => a.status === s.value).length;
                if (!n) return null;
                return (
                  <FilterChip key={s.value} active={filter === s.value} onClick={() => setFilter(s.value)}>
                    {s.label} <span className="tabular-nums">{n}</span>
                  </FilterChip>
                );
              })}
            </div>

            <Panel className="gap-0 p-0">
              <div className="flex flex-col divide-y">
                {shown.map((a) => (
                  <Row key={a.id} app={a} onUpdate={(body) => update(a.id, body)} />
                ))}
              </div>
            </Panel>
          </>
        )}
      </main>
    </>
  );
}

function Stat({ label, value, hint }: { label: string; value: number | string; hint?: string }) {
  return (
    <Panel className="gap-1 p-4">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-2xl font-semibold tabular-nums">{value}</span>
      {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
    </Panel>
  );
}

function FilterChip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-sm",
        active
          ? "border-primary bg-primary font-medium text-primary-foreground"
          : "bg-card text-muted-foreground hover:text-foreground",
      )}
    >
      {children}
    </button>
  );
}

function daysAgo(iso: string): number {
  return Math.max(0, Math.floor((Date.now() - new Date(iso).getTime()) / 86_400_000));
}

function Row({
  app,
  onUpdate,
}: {
  app: Application;
  onUpdate: (body: { status?: ApplicationStatus; notes?: string }) => void;
}) {
  const [notes, setNotes] = useState(app.notes);
  const status = STATUSES.find((s) => s.value === app.status)!;
  const waiting = app.status === "applied" ? daysAgo(app.last_change) : null;

  return (
    <div className="flex flex-wrap items-start gap-x-6 gap-y-3 px-5 py-4">
      <div className="flex min-w-[220px] flex-[2_1_260px] flex-col gap-0.5">
        <span className="text-sm text-muted-foreground">{app.company ?? "Unknown company"}</span>
        <span className="flex items-center gap-1.5 font-medium">
          {app.title}
          {app.url && (
            <a href={app.url} target="_blank" rel="noreferrer" aria-label="Open the job posting">
              <ExternalLink className="size-3.5 text-muted-foreground hover:text-foreground" />
            </a>
          )}
        </span>
        <Input
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          onBlur={() => notes !== app.notes && onUpdate({ notes })}
          placeholder="Add a note"
          className="mt-1.5 h-8 border-transparent bg-transparent px-1.5 text-sm shadow-none hover:border-input focus-visible:border-input"
        />
      </div>

      <div className="flex min-w-[140px] flex-[1_1_140px] flex-col gap-0.5 text-sm">
        <span className="text-muted-foreground">
          {app.applied_at ? `Sent ${new Date(app.applied_at).toLocaleDateString()}` : "Not sent yet"}
        </span>
        <span>{app.channel ?? <span className="text-muted-foreground">–</span>}</span>
        {waiting !== null && waiting >= 14 && (
          <span className="text-xs text-warn">No reply for {waiting} days</span>
        )}
      </div>

      <div className="flex min-w-[140px] flex-[1_1_140px] flex-col gap-2">
        <label className="sr-only" htmlFor={`status-${app.id}`}>
          Status
        </label>
        <div className="relative w-fit">
          <StatusPill tone={status.tone} icon={false} className="pointer-events-none h-8 pr-7 pl-3 text-sm">
            {status.label}
          </StatusPill>
          <select
            id={`status-${app.id}`}
            value={app.status}
            onChange={(e) => onUpdate({ status: e.target.value as ApplicationStatus })}
            className="absolute inset-0 cursor-pointer opacity-0"
          >
            {STATUSES.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
          <span aria-hidden className="pointer-events-none absolute top-1/2 right-2.5 -translate-y-1/2 text-[10px]">
            ▾
          </span>
        </div>
        {app.cv_pdf_url ? (
          <span className="flex gap-3 text-sm">
            <a
              href={`/api${app.cv_pdf_url}`}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 font-medium text-accent-foreground hover:underline"
            >
              <FileText className="size-3.5" /> {app.applied_at ? "Sent CV" : "Saved CV"}
            </a>
            <a href={`/api${app.cv_tex_url}`} className="text-muted-foreground hover:underline" download>
              .tex
            </a>
          </span>
        ) : (
          <span className="text-xs text-muted-foreground">CV made by hand</span>
        )}
      </div>
    </div>
  );
}

function ManualForm({ onSaved }: { onSaved: () => void }) {
  const [form, setForm] = useState<Omit<ManualApplication, "confirm_duplicate">>({
    company: "",
    title: "",
    url: null,
    status: "applied",
    applied_on: new Date().toISOString().slice(0, 10),
    channel: null,
    notes: "",
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [duplicate, setDuplicate] = useState<ApiError | null>(null);

  const set = (patch: Partial<ManualApplication>) => setForm((f) => ({ ...f, ...patch }));


  async function save(confirmDuplicate = false) {
    setBusy(true);
    setError(null);
    const res = await api.POST("/applications", {
      body: { ...form, confirm_duplicate: confirmDuplicate },
    });
    setBusy(false);
    if (res.data) return onSaved();
    const err = toApiError(res.error, res.response.status);
    if (err.code === "duplicate") setDuplicate(err);
    else setError(err);
  }

  return (
    <Panel>
      <PanelTitle>Add an application sent without factfit</PanelTitle>
      <p className="-mt-2 text-sm text-muted-foreground">
        These are the baseline: they show how hand-made applications do next to tailored ones.
      </p>
      <form
        className="grid gap-3 sm:grid-cols-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (form.company.trim() && form.title.trim()) save();
        }}
      >
        <Field label="Company">
          <Input required value={form.company} onChange={(e) => set({ company: e.target.value })} className="h-10" />
        </Field>
        <Field label="Job title">
          <Input required value={form.title} onChange={(e) => set({ title: e.target.value })} className="h-10" />
        </Field>
        <Field label="Posting link">
          <Input
            type="url"
            value={form.url ?? ""}
            onChange={(e) => set({ url: e.target.value || null })}
            placeholder="Optional"
            className="h-10"
          />
        </Field>
        <Field label="Sent on">
          <Input
            type="date"
            value={form.applied_on ?? ""}
            onChange={(e) => set({ applied_on: e.target.value || null })}
            className="h-10"
          />
        </Field>
        <Field label="Status">
          <select
            value={form.status}
            onChange={(e) => set({ status: e.target.value as ApplicationStatus })}
            className="h-10 rounded-lg border border-input bg-card px-3 text-sm"
          >
            {STATUSES.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Where you sent it">
          <Input
            list="manual-channels"
            value={form.channel ?? ""}
            onChange={(e) => set({ channel: e.target.value || null })}
            placeholder="Optional"
            className="h-10"
          />
          <datalist id="manual-channels">
            {CHANNELS.map((c) => (
              <option key={c} value={c} />
            ))}
          </datalist>
        </Field>
        <div className="flex flex-wrap items-center gap-2 sm:col-span-2">
          <Button type="submit" size="lg" disabled={busy || !form.company.trim() || !form.title.trim()}>
            {busy ? <Spinner /> : <Plus />}
            Add application
          </Button>
        </div>
      </form>
      {duplicate && (
        <div className="flex flex-col gap-2">
          <Note tone="warn">
            <strong className="font-semibold">{duplicate.message}</strong>
            <ul className="mt-1 list-disc pl-5">
              {duplicate.issues.map((i) => (
                <li key={i}>{i}</li>
              ))}
            </ul>
          </Note>
          <Button variant="outline" className="self-start" onClick={() => save(true)} disabled={busy}>
            Add anyway
          </Button>
        </div>
      )}
      {error && <ErrorBox error={error} />}
    </Panel>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5 text-sm">
      <span className="text-muted-foreground">{label}</span>
      {children}
    </label>
  );
}
