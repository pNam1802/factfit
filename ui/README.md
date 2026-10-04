# factfit UI

Next.js app (App Router, TypeScript, Tailwind, shadcn/ui). It holds no business logic: every
action calls the Python API, forwarded from `/api/...` by [`next.config.ts`](next.config.ts).

Run it together with the API from the repo root:

```bash
uv run factfit dev        # UI on http://localhost:3000, API on http://localhost:8000
```

API types are generated from the FastAPI schema, never edited by hand:

```bash
uv run factfit export-openapi   # from the repo root: writes ui/openapi.json
cd ui && npm run gen:api        # writes src/lib/api-types.ts
```

Why Next.js rather than Streamlit: [docs/adr/0004-nextjs-ui.md](../docs/adr/0004-nextjs-ui.md).
