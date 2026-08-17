# Echora web interface

The React, TypeScript, and Vite interface for Echora's local-first communication demo. It records or uploads speech, lets the speaker select a session setting, displays post-chain message suggestions with attached literal ASR evidence, and only enables Copy or Speak after explicit confirmation.

The browser keeps session history in memory only. Refreshing the page clears it.

## Run locally

From the repository root, use the combined launcher:

```bash
./scripts/dev.sh
```

Or run only the interface:

```bash
cd frontend
npm install
npm run dev
```

The interface expects the FastAPI backend at `http://127.0.0.1:8000` by default. Override it at build time with `NEXT_PUBLIC_ECHORA_API_URL`.

## Verify

```bash
npm test
```

This runs the production build and the interface source checks. No account, database, personal profile, or persistent history is used.
