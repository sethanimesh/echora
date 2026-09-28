# Echora for iOS

This is a second client for the existing Echora FastAPI backend. It does not replace or share build output with `frontend/`.

## Run in the iOS simulator

Start the existing backend in one terminal:

```bash
cd ..
PYTHONPATH="$PWD/backend:$PWD" .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then start Expo in another terminal:

```bash
cd mobile
npm install
npm run ios
```

The app infers the development computer’s address from the Expo bundle and uses port 8000. Set `EXPO_PUBLIC_ECHORA_API_URL` for an installed build or a different server; the simulator fallback is `http://127.0.0.1:8000`.

## Run on a physical iPhone

The phone and Mac must be on the same local network. Expose the unified API to the LAN:

```bash
cd ..
PYTHONPATH="$PWD/backend:$PWD" .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then run `npm start` in `mobile/` and open the project with Expo Go. iOS will ask for local-network, microphone, and—only when place detection or tagging is used—location permission. Tagged coordinates are stored only by the local Echora service on the Mac. Matching happens in the app, and coordinates never enter a transcription request. Set `EXPO_PUBLIC_ECHORA_API_URL` in `.env.local` to override automatic address detection.

## API and message-flow guarantees

- A recording enters the same session and revision-bound message workflow as the web client through `/api/v1/communication/audio`. The profile revision, place, audience, and declared listener are frozen before processing. Native requests use an opaque in-memory session token; no provider credentials enter the app.
- Literal ASR hypotheses remain immutable evidence. The mobile client displays them and never rewrites them with the suggested message.
- A resolved recommendation speaks immediately. Local adapted English uses the separate acoustic-verification decision; missing or unvalidated artifacts require a choice even if only one expanded suggestion survives. All literal alternatives remain selectable. A tap selects the server candidate and authorizes its exact revision before speech. There is no extra confirmation screen. Edits invalidate the old speech authorization.
- Groq TTS is an enhancement. If it or playback fails, `expo-speech` is the quiet fallback.
- The last approved message can support a narrow follow-up in the same context for 10 minutes. That reference stays in server memory. Only **Remember this message** persists displayed wording to the selected profile; confirmation and playback never save it. Remembered messages can be listed, deleted or cleared, and writable profiles can be deleted with their vectors. Explicit profile and place edits use the same stores as the web client.
- Deleting a referenced source revokes dependent wording and speech. While displaying a result, the native client checks for cross-session invalidation every 500 ms, subject to network delay, and rejects delayed stale speech responses.
- The selected profile's language also applies to native recordings. A profile changed in another client must be reloaded before recording again.
- Places come from the existing `/api/v1/places` store. Coordinates are matched locally and never added to transcription requests.

## Verify

```bash
npm test
npm run export:ios
```

Native transport integration tests run from the repository root with `PYTHONPATH="$PWD/backend:$PWD" .venv/bin/python -m pytest mobile/tests/test_native.py`. Device microphone and playback checks still require an iOS simulator or phone. The native UI currently exposes adapted recognition and Groq/device speech; web-only camera, gaze, and Fish controls are not presented as native features.
