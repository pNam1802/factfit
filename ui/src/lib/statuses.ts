import type { Tone } from "@/components/parts";
import type { ApplicationStatus } from "@/lib/api";

/** Application statuses in pipeline order, with how each one is shown. */
export const STATUSES: { value: ApplicationStatus; label: string; tone: Tone }[] = [
  { value: "saved", label: "Saved", tone: "neutral" },
  { value: "applied", label: "Applied", tone: "neutral" },
  { value: "screening", label: "Screening", tone: "warn" },
  { value: "interview", label: "Interview", tone: "warn" },
  { value: "offer", label: "Offer", tone: "ok" },
  { value: "rejected", label: "Rejected", tone: "bad" },
  { value: "ghosted", label: "No reply", tone: "bad" },
];

export const CHANNELS = ["Company site", "LinkedIn", "Email", "Referral", "TopCV", "ITviec"];
