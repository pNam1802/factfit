import createClient from "openapi-fetch";

import type { components, paths } from "./api-types";

// Types are generated from the FastAPI schema: `uv run factfit export-openapi`
// then `npm run gen:api`. Never edit api-types.ts by hand.
export type Job = components["schemas"]["JobOut"];
export type Match = components["schemas"]["MatchOut"];
export type Requirement = components["schemas"]["RequirementOut"];
export type Level = NonNullable<components["schemas"]["MatchRequest"]["level"]>;

// "/api" is forwarded to FastAPI by next.config.ts.
export const api = createClient<paths>({ baseUrl: "/api" });

export class ApiError extends Error {
  constructor(
    message: string,
    public issues: string[] = [],
  ) {
    super(message);
  }
}

/** Turn any error body from FastAPI into one readable message. */
export function toApiError(error: unknown, status: number): ApiError {
  const body = error as { detail?: unknown; issues?: string[] } | undefined;
  if (typeof body?.detail === "string") return new ApiError(body.detail, body.issues ?? []);
  if (Array.isArray(body?.detail)) {
    // Request validation errors: a list of { loc, msg }.
    const msgs = body.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d));
    return new ApiError(msgs.join("; "));
  }
  return new ApiError(`Request failed (HTTP ${status}). Is the API running?`);
}
