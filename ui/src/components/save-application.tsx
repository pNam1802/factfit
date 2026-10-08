"use client";

import { ArrowRight, Check, Send } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { ErrorBox, Note, Panel, PanelTitle, Spinner } from "@/components/parts";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { type Application, ApiError, api, toApiError } from "@/lib/api";
import { CHANNELS } from "@/lib/statuses";

type Kind = "applied" | "saved";

/** Save the finished run as an application. The CV is frozen on the server, so what this
 * application links to stays what was sent, even after a rebuild. */
export function SaveApplication({ runId }: { runId: string }) {
  const [channel, setChannel] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState<Kind | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [confirm, setConfirm] = useState<{ kind: Kind; error: ApiError } | null>(null);
  const [saved, setSaved] = useState<Application | null>(null);

  async function save(kind: Kind, confirmed: { duplicate?: boolean; ats?: boolean } = {}) {
    setBusy(kind);
    setError(null);
    const res = await api.POST("/runs/{run_id}/application", {
      params: { path: { run_id: runId } },
      body: {
        status: kind,
        channel: channel.trim() || null,
        notes: notes.trim(),
        confirm_duplicate: confirmed.duplicate ?? false,
        confirm_ats: confirmed.ats ?? false,
      },
    });
    setBusy(null);
    if (res.data) {
      setSaved(res.data);
      setConfirm(null);
      return;
    }
    const err = toApiError(res.error, res.response.status);
    if (err.code === "duplicate" || err.code === "ats") setConfirm({ kind, error: err });
    else setError(err);
  }

  if (saved) {
    return (
      <Panel className="border-ok-line bg-ok-soft text-ok-strong">
        <div className="flex items-center gap-2 font-semibold">
          <Check className="size-4" strokeWidth={3} />
          {saved.status === "applied" ? "Marked as applied" : "Saved for later"}
        </div>
        <p className="text-sm">
          This CV is frozen with application #{saved.id}: rebuilding here will not change it.
        </p>
        <Link
          href="/applications"
          className="inline-flex items-center gap-1 text-sm font-medium underline-offset-4 hover:underline"
        >
          Open Applications <ArrowRight className="size-3.5" />
        </Link>
      </Panel>
    );
  }

  return (
    <Panel>
      <PanelTitle>Save as application</PanelTitle>
      <label className="flex flex-col gap-1.5 text-sm">
        <span className="text-muted-foreground">Where you send it</span>
        <Input
          list="channels"
          value={channel}
          onChange={(e) => setChannel(e.target.value)}
          placeholder="LinkedIn, email, company site…"
          className="h-10"
        />
        <datalist id="channels">
          {CHANNELS.map((c) => (
            <option key={c} value={c} />
          ))}
        </datalist>
      </label>
      <label className="flex flex-col gap-1.5 text-sm">
        <span className="text-muted-foreground">Notes</span>
        <Input
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Optional"
          className="h-10"
        />
      </label>

      {confirm ? (
        <div className="flex flex-col gap-3">
          <Note tone={confirm.error.code === "ats" ? "bad" : "warn"}>
            <strong className="font-semibold">{confirm.error.message}</strong>
            <ul className="mt-1 list-disc pl-5">
              {confirm.error.issues.map((i) => (
                <li key={i}>{i}</li>
              ))}
            </ul>
          </Note>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              onClick={() =>
                save(confirm.kind, {
                  duplicate: true,
                  ats: confirm.error.code === "ats",
                })
              }
              disabled={busy !== null}
            >
              {busy ? <Spinner /> : null}
              {confirm.error.code === "ats" ? "Mark as applied anyway" : "Save anyway"}
            </Button>
            <Button variant="ghost" onClick={() => setConfirm(null)} disabled={busy !== null}>
              Cancel
            </Button>
          </div>
        </div>
      ) : (
        <div className="flex flex-wrap gap-2">
          <Button size="lg" className="flex-1" onClick={() => save("applied")} disabled={busy !== null}>
            {busy === "applied" ? <Spinner /> : <Send />}
            Mark as applied
          </Button>
          <Button size="lg" variant="outline" onClick={() => save("saved")} disabled={busy !== null}>
            {busy === "saved" && <Spinner />}
            Save for later
          </Button>
        </div>
      )}
      {error && <ErrorBox error={error} />}
    </Panel>
  );
}
