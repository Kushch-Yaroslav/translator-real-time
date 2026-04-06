# Session Context For Next Codex Turn

## What This Project Is

Linux desktop realtime speech translator:
- microphone input on Linux
- STT
- translation
- TTS
- output into virtual mic `translator_mic`

Main real user scenario:
- user speaks Russian
- app should speak English into Telegram as fast as possible
- target path is `RU -> EN`

Current product goal:
- good translation quality
- stable long session behavior
- first audible English ideally near `2-3 sec`

User hardware:
- RTX 3090 24GB

## Current Best State

Current best practical state is **not** Riva, **not** Canary, **not** NIM hybrid.

Current best backend:
- `faster_whisper`

Current best observed behavior:
- startup of speech often begins around `2-3 sec`
- translation quality is generally good enough to use
- first-run cold start after app open was significantly improved by background prewarm and service caching

Current known remaining defect:
- sometimes a false tail artifact appears:
  - `To be continued...`
- this artifact is known and tolerated for now because this state is still the best practical checkpoint

Important current decision:
- stop chasing more heuristics for now
- treat current `faster_whisper` state as the best checkpoint to resume from next session

## Current Active Architecture

Current active path:
- audio capture via sounddevice / PulseAudio-PipeWire environment
- VAD + `faster-whisper` for realtime-ish STT
- Marian/OPUS for text translation
- Piper for TTS
- output to `translator_mic`

Important active backend in config:
- `stt.backend = "faster_whisper"`

Important files now:
- [core/audio_engine.py](/media/yaroslav/DATA/Мой%20переводчик/core/audio_engine.py)
- [core/main_window.py](/media/yaroslav/DATA/Мой%20переводчик/core/main_window.py)
- [core/stt_runtime.py](/media/yaroslav/DATA/Мой%20переводчик/core/stt_runtime.py)
- [core/sst/faster_whisper_realtime_stt_service.py](/media/yaroslav/DATA/Мой%20переводчик/core/sst/faster_whisper_realtime_stt_service.py)
- [core/translation/translation_service.py](/media/yaroslav/DATA/Мой%20переводчик/core/translation/translation_service.py)
- [core/tts/tts_service.py](/media/yaroslav/DATA/Мой%20переводчик/core/tts/tts_service.py)
- [core/audio_service.py](/media/yaroslav/DATA/Мой%20переводчик/core/audio_service.py)
- [app_config.json](/media/yaroslav/DATA/Мой%20переводчик/app_config.json)
- [logs/app.log](/media/yaroslav/DATA/Мой%20переводчик/logs/app.log)
- [logs/startup.log](/media/yaroslav/DATA/Мой%20переводчик/logs/startup.log)

## What Was Tried And Rejected

### Old NIM partial heuristics

We spent a long time tuning partial/final heuristics inside `core/audio_engine.py`.

Results:
- could sometimes improve latency
- often created broken tails, duplicates, contextual bleed, bad names, and unstable long-session behavior

Conclusion:
- not the right main path

### NVIDIA Riva

Riva was integrated and tested as a separate backend.

Results:
- some promising starts on latency
- but repeated problems with noisy partials, duplicated chunks, bad sentence segmentation, and unstable output

Conclusion:
- not chosen as current best baseline

### Boundary layer / confirm-pass variants

We also tried:
- separate sentence boundary layer
- confirm-pass logic
- boundary + confirm combinations

Results:
- complexity increased
- quality/latency tradeoff still bad
- too many moving parts for unreliable gain

Conclusion:
- not current direction

### Canary AST

Tried NVIDIA Canary AST path.

Problems:
- image availability / access issues
- offline-like behavior
- poor latency for this scenario
- not practical enough on this setup

Conclusion:
- abandoned

### Silero VAD + NIM hybrid

Tried a hybrid path with Silero VAD plus NIM-based transcription.

Problems:
- wrong API fit in earlier attempts
- later working version still did not beat current best path

Conclusion:
- abandoned

## What Finally Worked Better

The most useful changes that improved first-run and practical usage were:

1. Switch to `faster_whisper`
- this became the best practical STT backend in this repo so far

2. Stop requiring NIM runtime for `faster_whisper`
- `core/stt_runtime.py` now returns early for backend `faster_whisper`
- this avoids fake startup failures like:
  - `NIM did not become ready on http://localhost:9000 within ...`

3. Add background prewarm on app open
- app now starts warming runtime in the background after UI opens
- this reduces first real phrase latency

4. Cache heavy services between start/stop
- translation service
- TTS service
- realtime STT service

5. Keep low-latency direct pipeline for `faster_whisper`
- there is a dedicated low-latency path in `core/audio_engine.py`
- this is why startup can reach around `2-3 sec` in the good runs

## Current Known Behavior

### Good

- first audible output can begin around `2-3 sec`
- translation quality is now good enough to use live
- cold start is much better than before
- app is in a better state than all previous Riva / Canary / hybrid experiments

### Known bad

- sometimes `To be continued...` appears even though user did not say it
- this is a known artifact in the current best checkpoint
- some phrases can still produce imperfect early partial output
- `Move recording stream to source ... FAILED` still appears often in logs, but it does **not** seem to be the main blocker now

### Important nuance

There was a long sequence of tiny heuristic edits after the good checkpoint.
Some later edits made quality worse or pushed latency back to `5-6 sec`.

The state to continue from next session should be:
- keep the fast-start `faster_whisper` baseline
- do not blindly continue the latest heuristic churn
- if changing anything, move carefully and compare against the known good baseline

## What Logs Show

Useful files:
- runtime app log: [logs/app.log](/media/yaroslav/DATA/Мой%20переводчик/logs/app.log)
- startup log: [logs/startup.log](/media/yaroslav/DATA/Мой%20переводчик/logs/startup.log)

User often provides:
- Telegram `.ogg` files with the actual English output

This is very useful because:
- logs show STT/translation/TTS timings
- `.ogg` proves what really went out to Telegram

When resuming next session:
- inspect `logs/app.log`
- inspect latest Telegram `.ogg`
- compare actual audio start time and spoken text

## Current User Preference

User explicitly prefers:
- practical progress over theory
- local-only solutions when possible
- strong focus on `RU -> EN`
- willingness to use lots of GPU/CPU/RAM if needed

User also explicitly said:
- it is acceptable to stop here and fix on the current best checkpoint
- this current checkpoint is the best one so far even with `To be continued...`

## What Next Session Should Remember

1. Read this file first.
2. Assume current best backend is `faster_whisper`.
3. Do not restart from Riva or Canary unless user explicitly asks.
4. The main unresolved defect at this checkpoint is:
   - false `To be continued...`
5. The main success at this checkpoint is:
   - startup around `2-3 sec`
   - generally good translation

## Important Do And Don't

Do:
- preserve fast-start behavior
- preserve prewarm and service caching
- compare any new heuristic against the current `2-3 sec` startup baseline
- use Telegram `.ogg` plus logs as the source of truth

Do not:
- revert unrelated user changes
- stop Docker containers on app close
- assume Riva/Canary are still active directions
- reintroduce slow `5-6 sec` startup just to clean one artifact

## Practical Resume Point

If next session continues work:
- start from the current `faster_whisper` baseline
- treat `To be continued...` as the known regression/defect
- any further fix must preserve:
  - `~2-3 sec` startup
  - generally good RU -> EN translation quality

