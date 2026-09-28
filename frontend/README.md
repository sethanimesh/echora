# Echora web app

The primary web client for the unified Echora backend. Run `../scripts/dev.sh` from this directory, or `./scripts/dev.sh` from the repository root, then open http://localhost:3000.

This is a React/Vinext app. Vite proxies `/api` to the shared service on port 8000 and `/gaze-api` to the optional local gaze service on 8767. The backend must be running for communication features.

```sh
npm test
npm run typecheck
npm run build
```

The main communication screen is `app/page.tsx`. Its helpers own recording, places, accessibility, delivery suggestions, and revision-bound speech. Automatic speech is allowed only for the current local operation, never merely because an old session or event snapshot has a suggestion.

Local adapted English uses a separate validated acoustic decision for automatic selection. A single generated suggestion cannot resolve uncertain audio; all literal alternatives remain selectable. Explicit **Remember this message** stores only the displayed revision under the selected profile. Memory management offers individual deletion, clear and writable-profile deletion. Server invalidation events cancel dependent queued or active playback, including delayed confirmation responses.

The original Sites/Cloudflare scaffold is not part of the running app. See the repository README for provider configuration, profile migration, and the chosen immediate-speech and ten-minute follow-up policies.
