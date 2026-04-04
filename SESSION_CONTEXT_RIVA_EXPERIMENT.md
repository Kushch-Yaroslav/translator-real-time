# Session Context For Next Codex Turn

## What This Project Is

Linux desktop realtime speech translator with:
- local audio routing via PulseAudio/PipeWire
- STT via NVIDIA NIM realtime websocket
- translation via Marian/OPUS
- TTS via Piper
- output to virtual mic `translator_mic`

Main current direction of work:
- `RU -> EN` is the priority path
- `EN -> RU` exists and is already in a more stable state

## Current Repo State

Important files:
- `core/audio_engine.py`
- `core/main_window.py`
- `core/app_config.py`
- `app_config.json`
- `core/translation/translation_service.py`
- `core/nim_runtime.py`
- `tests/record_test_phrases.py`
- `tests/record_test_phrases_ru_to_en.py`
- `tests/evaluate_test_phrases.py`
- `tests/evaluate_test_phrases_ru_to_en.py`

Logs:
- runtime app log: `logs/app.log`
- startup log: `logs/startup.log`

Test recordings are intentionally gitignored:
- `recordings/test_phrases/`
- `recordings/test_phrases_ru_to_en/`

## What Was Already Done

### Stable base / infra
- README in Russian was added earlier.
- Config is stored in `app_config.json`.
- UI is Russian.
- Profiles/presets exist in UI.
- File logging exists in `logs/app.log`.
- Docker/NIM autostart exists through `core/nim_runtime.py`.
- App does **not** stop Docker container on close.

### EN -> RU
- This branch was stabilized earlier.
- Major issues like phantom finals, restart-per-final, TTS-on-CPU were handled.

### RU -> EN groundwork
- Branch-aware config model exists in `core/app_config.py`.
- `branches.primary` / `branches.secondary` exist.
- `RU -> EN` was added as isolated direction, not full dual-branch runtime.
- Current active branch in `app_config.json` is RU->EN-oriented.

### RU -> EN quality changes already done
- Anti-duplicate collapsing for repeated finals in `core/audio_engine.py`
- Test evaluator also updated for repeated sentence collapsing
- Lightweight translation normalization in `core/translation/translation_service.py`
  - `инженером программистом` -> `инженером-программистом`
  - `programmer engineer` -> `software engineer`
  - `Short sentence` -> `Short phrase`

### RU -> EN live-latency experiments already attempted

We tried multiple iterations on partial/final logic in `core/audio_engine.py`:

1. Disable partial emit entirely for RU->EN
- Quality improved
- Latency became too high (`~9 sec` in Telegram on short phrases)

2. Re-enable partial emit with stricter rules
- Earlier start
- But broken chunks appeared (`my name is me`, `I'm 26...`, etc.)

3. Add `last_emitted_source_text`
- Attempt to make final only emit the remaining tail after early partial

4. Limit RU->EN to only one early partial per utterance
- Reduced some chaos

5. Add short merge window for partial before enqueueing to translation/TTS
- Intended to soften split between early partial and final tail

6. Add utterance-state reset after inactivity
- Intended to make long open sessions behave better

7. Restrict duplicate-skip by time window
- Intended to allow repeating same phrase later in conversation

8. Add specific guard against partial ending as `меня зовут я`
- Intended to avoid `my name is me`

## Current Observed Behavior

Latest live tests are better than before, but still not solved:

### Good
- Long session behavior improved compared to older state.
- Some phrases now start around `3.5s - 5s` instead of `9s - 10s`.
- RU->EN is usable for some phrases.

### Still bad / important remaining issues
- Early partial sometimes cuts proper nouns or names:
  - `Привет , меня зовут Яросла .` -> `Hi, my name is Jarosla.`
  - `Привет , меня зовут я .` -> `Hey, my name's me.`
- Incremental extraction after early partial can still cut the wrong tail.
- Long session still has contextual bleed across phrases sometimes.
- Toponyms like `Запорожье` are poorly recognized by STT and then translate into nonsense.
- `Move recording stream ... FAILED` appears often in logs, though pipeline still continues.

### Important product conclusion from last session

User idea:
- detect sentence boundary / point / pause
- speak first finished sentence immediately
- while user continues speaking next sentence in Russian

Example:
- User says: `Всем привет, меня зовут Ярослав, мне 26 лет. Я люблю программировать и я из города Запорожье.`
- Desired behavior:
  - first sentence should start English TTS immediately after sentence boundary
  - second sentence should still be captured while first is being spoken

This was considered the right direction.

## Decision Reached At End Of Session

We do **not** want to keep endlessly tuning the current NIM partial heuristics in this same branch.

Next step should be:
- create a **separate git branch**
- run an experiment with **NVIDIA Riva**

Reason:
- user asked whether NVIDIA already has tools for this
- current NIM realtime partial/final heuristics are getting too messy
- Riva may provide stronger streaming ASR features, VAD / end-of-utterance behavior, and possibly better segmentation support for this use case

## What Next Session Should Do

The next Codex session should focus on an experiment branch for Riva.

Goal:
- compare current NIM-based RU->EN pipeline with a Riva-based streaming ASR path
- especially test whether Riva can produce earlier and cleaner sentence-level boundaries
- do **not** break the current working baseline branch while experimenting

Suggested first actions in the next session:
1. Read this file.
2. Inspect current `core/sst/nim_realtime_stt_service.py`.
3. Determine whether to create a parallel `riva_realtime_stt_service.py`.
4. Check what Riva container / docs / local install assumptions are needed on this machine.
5. Build the Riva experiment in isolation behind config, not as a destructive replacement.

## Important User Preferences

- User prefers concise practical work.
- User is fine with separate git branch for experiments.
- User wants local-only and free components.
- User hardware: RTX 3090 24GB.
- User is in Russian language workflow.
- User cares more about `RU -> EN` than `EN -> RU`.

## Notes For Future Codex

- Do not revert unrelated changes.
- Do not stop Docker container on app close.
- Be careful with anti-duplicate logic: user explicitly noted that if someone asks the same question again, the phrase still must be spoken.
- Live-session stability matters more now than isolated one-shot phrase success.
- Current branch can likely be treated as `RU->EN baseline`, not final product.

