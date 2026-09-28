# ADR 0003: One revision lifecycle and explicit memory

Status: implemented; recorded retrospectively on 28 September 2026.

## Context and alternatives

Web/native divergence and delayed provider responses can authorize stale speech. Automatically saving approved messages also confuses a communication action with persistence consent. Alternatives were independent client stacks, an extra user confirmation step, and a shared revision-bound backend.

## Decision

Use one shared backend lifecycle. Internal confirmation authorizes exact current text, pronunciation and delivery; a resolved arrival or explicit choice supplies the user-facing action. Edits, Stop, new jobs and relevant context changes revoke pending playback. Reconnection never replays an old result.

Keep a visible same-context reference for at most ten minutes. Persist exact wording only on explicit Remember in the selected profile's SQLite store. Additive migration preserves source stores. Remembered wording is not acoustic training truth.

## Trade-offs and evidence

Shared contracts reduce client drift but couple both clients to the lifecycle. SQLite supports local ownership and transactional updates; this is not a production multi-user architecture. Native cross-session invalidation uses polling and is subject to network delay.

[Architecture](../architecture.md), [memory contract](../verification-and-memory.md), [native contract tests](../../mobile/tests/test_native.py) and the isolated unified test runner document and exercise these choices. Revisit if multi-device persistence or measured interaction costs require a different design.
