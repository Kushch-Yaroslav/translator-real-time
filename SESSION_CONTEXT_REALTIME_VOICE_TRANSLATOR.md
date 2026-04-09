# AI Session Context: Голосовой перевод в реальном времени

## 0. Purpose Of This File

This file is for AI bootstrap, not for end-user docs.

Goal:
- give a new coding agent enough context to continue work fast
- describe current architecture, current routing, current backends, current known-good state, current sharp edges
- prevent re-litigating old abandoned directions

Read this file before making assumptions about the project.

## 1. Project Identity

Canonical current product name:
- `Голосовой перевод в реальном времени`

Old historical name:
- `Мой переводчик`

Core product:
- local Linux desktop app
- real-time bidirectional voice translation
- UI on `PySide6`
- audio routing through `PulseAudio` / `PipeWire`

Current two working branches:
- `RU => EN`
- `EN => RU`

User-level mental model:
- user speaks Russian -> other side hears English
- other side speaks English -> user hears Russian

## 2. Current Working Product Model

There are two separate runtime branches inside one app.

### Branch A: `RU => EN`

Use case:
- outgoing speech
- user speaks Russian into real mic
- app sends English speech into a virtual microphone for Telegram / Meet / Zoom

Current pipeline:
- input audio: real microphone
- STT: `faster-whisper`
- translation: `Helsinki-NLP/opus-mt-ru-en`
- TTS: `Piper`, voice `en_US-ryan-medium`
- output: virtual sink/source path ending in `TranslatorMicrophone`

UI semantics:
- `Продолжить`:
  remote side hears English synthesized speech
- `Пауза`:
  remote side hears original Russian passthrough

### Branch B: `EN => RU`

Use case:
- incoming speech
- English source audio is routed into a virtual sink
- app translates it and plays Russian into headphones

Current pipeline:
- input audio: `TranslatorListen.monitor`
- STT: `whisper.cpp` server
- VAD: `Silero VAD`
- translation: `Helsinki-NLP/opus-mt-en-ru`
- TTS: `Piper`, voice `ru_RU-dmitri-medium`
- output: real headphones / real sink

UI semantics:
- `Продолжить`:
  hear Russian translation
- `Пауза`:
  translation muted, original English still available
- original-audio controls:
  - `Заглушить`
  - `Приглушить`
  - `Слушать 100%`
  - slider controls original English loudness

## 3. Audio Routing Model

This project is fundamentally a PulseAudio routing application.

Important virtual devices:

- sink `translator_mic`
  - human-facing description: `TranslatorMic`
  - internal outgoing translated audio sink

- remapped source `translator_microphone`
  - human-facing description: `TranslatorMicrophone`
  - this is the device user should choose as mic in Telegram / Meet / Zoom
  - backed by `translator_mic.monitor`

- sink `translator_listen`
  - human-facing description: `TranslatorListen`
  - user routes English source apps here when they want local English->Russian translation

Operational meaning:

- `TranslatorMicrophone`
  = what remote communication apps should use as mic

- `TranslatorListen`
  = what source apps should use as playback target if user wants their audio translated locally

Current expected usage:

- Telegram / Meet / Zoom microphone:
  `TranslatorMicrophone`

- Chrome / YouTube / any incoming English source playback:
  `TranslatorListen`

Do not confuse:
- `TranslatorMic`
  is internal sink
- `TranslatorMicrophone`
  is the end-user visible source to select in comm apps

## 4. Current UI State

UI was intentionally simplified.

Removed from UI:
- manual direction switching
- profiles / presets
- low-level STT/TTS tuning
- segmentation controls
- manual backend choice

Current visible controls are product-level only:
- `Обновить устройства`
- `Запустить пайплайн`
- `Остановить пайплайн`
- `Говорить RU=>EN` pause/resume
- `Слушать EN=>RU` pause/resume
- original English loudness controls
- logs

There is also a separate backend status window.

Status window behavior:
- opens with app start
- can remain open if main window is closed
- shows per-backend status
- has per-backend restart/stop controls

## 5. Current Backend Management

### `EN => RU`

Managed as external server:
- `whisper.cpp`

Launcher:
- `scripts/start_whispercpp_server.sh`

Default endpoint:
- `http://127.0.0.1:8178`

Whisper model currently used:
- `medium.en`

whisper.cpp repo path:
- `tmp/whisper.cpp`

Model path:
- `/media/yaroslav/DATA/ai_models/whisper.cpp/models/ggml-medium.en.bin`

### `RU => EN`

Managed as embedded runtime:
- `faster-whisper`

Important nuance:
- this is not a separate server process
- backend status window should treat it as embedded runtime, not as docker or external daemon

## 6. Current Config Truth

Primary config file:
- [app_config.json](/media/yaroslav/DATA/Мой%20переводчик/app_config.json)

Current key facts from config:

- global STT backend:
  `whisper_cpp`

- primary branch:
  `EN -> RU`

- secondary branch:
  `RU -> EN`

- audio:
  - samplerate `48000`
  - channels `1`
  - blocksize `1024`

- `EN => RU` conservative quality-biased settings:
  - `commit_interval_sec = 0.65`
  - `final_debounce_sec = 0.9`
  - `partial_emit_enabled = false`
  - `partial_stability_sec = 0.8`
  - `partial_min_words = 7`

These are intentionally not optimized for absolute minimum latency anymore.
Reason:
- live English media quality was too unstable with aggressive partial emission

### Runtime override for `RU => EN`

`RU => EN` is built by `build_ru_to_en_runtime_config()` in:
- [core/backend_manager.py](/media/yaroslav/DATA/Мой%20переводчик/core/backend_manager.py)

It overrides branch settings to faster, low-latency values for outgoing speech:
- backend forced to `faster_whisper`
- lower debounce
- partial emission enabled
- smaller VAD windows

This asymmetry is intentional.

## 7. Core Files To Read First

If resuming work, read these first:

1. [core/main_window.py](/media/yaroslav/DATA/Мой%20переводчик/core/main_window.py)
2. [core/audio_engine.py](/media/yaroslav/DATA/Мой%20переводчик/core/audio_engine.py)
3. [core/audio_service.py](/media/yaroslav/DATA/Мой%20переводчик/core/audio_service.py)
4. [core/backend_manager.py](/media/yaroslav/DATA/Мой%20переводчик/core/backend_manager.py)
5. [core/backend_status_window.py](/media/yaroslav/DATA/Мой%20переводчик/core/backend_status_window.py)
6. [core/translation/translation_service.py](/media/yaroslav/DATA/Мой%20переводчик/core/translation/translation_service.py)
7. [core/sst/whispercpp_realtime_stt_service.py](/media/yaroslav/DATA/Мой%20переводчик/core/sst/whispercpp_realtime_stt_service.py)
8. [core/sst/faster_whisper_realtime_stt_service.py](/media/yaroslav/DATA/Мой%20переводчик/core/sst/faster_whisper_realtime_stt_service.py)
9. [main.py](/media/yaroslav/DATA/Мой%20переводчик/main.py)
10. [app_config.json](/media/yaroslav/DATA/Мой%20переводчик/app_config.json)

## 8. Logging And Truth Sources

Primary logs:
- [logs/app.log](/media/yaroslav/DATA/Мой%20переводчик/logs/app.log)
- [logs/startup.log](/media/yaroslav/DATA/Мой%20переводчик/logs/startup.log)

High-value log prefixes:
- `Audio routing |`
- `Audio routing snapshot |`
- `Original audio loopback`
- `Speak passthrough loopback`
- `[LISTEN] ...`
- `[SPEAK] ...`
- `TRANSLATED:`
- `TTS audio ready:`

User also often provides:
- Telegram `.ogg`

For this project, `.ogg` + `app.log` together are the source of truth.

## 9. Current Known Good State

As of latest stable session:

### Good

- `EN => RU`:
  - YouTube / Chrome can be routed to `TranslatorListen`
  - Russian translation is heard in headphones
  - original English loudness can be controlled from UI

- `RU => EN`:
  - user speaks Russian
  - English TTS goes into `TranslatorMicrophone`
  - Telegram receives English translated audio
  - user no longer hears their own outgoing English locally after routing fixes

- backend startup works
- separate status window works
- app start no longer freezes UI

### Approximate resource usage

- combined VRAM around `7-8 GB`

## 10. Current Known Fragile Areas

These are still sensitive:

1. PulseAudio routing
- this project breaks easily if routing assumptions drift
- duplicate leftover loopback modules can cause phantom local playback
- cleanup of translator-related loopback modules matters

2. English original loudness
- this is controlled through loopback sink-input volume
- low values were previously not quiet enough
- current implementation uses mute at `0%` and non-linear scaling for low values

3. `EN => RU` quality on long natural speech
- still imperfect
- acceptable for many cases, but not “solved”
- current config is a compromise favoring quality over minimum latency

4. `RU => EN` is optimized for outgoing short/interactive speech, not for arbitrary Russian media transcription

## 11. Important Historical Lessons

Do not restart abandoned experiments unless user explicitly asks.

Previously tried and effectively abandoned:
- NVIDIA Riva
- NVIDIA Canary
- NIM hybrid variants
- Kyutai / Moshi
- Nemotron streaming
- sherpa-onnx
- WhisperLiveKit
- SimulStreaming

Reason:
- worse quality
- worse latency
- unstable streaming behavior
- too many hallucinations / duplicated chunks / wrong names
- operational complexity not worth gain

Current active best practical stack is:
- `whisper.cpp` for `EN => RU`
- `faster-whisper` for `RU => EN`

This asymmetry is intentional and should not be treated as a bug.

## 12. Important Product Semantics

The user does not want low-level controls exposed in UI.

The desired UX is product-level only:
- choose target app playback to `TranslatorListen` when wanting local translation
- choose mic `TranslatorMicrophone` in communication apps
- control behavior using pause buttons and loudness buttons

Meaning of pauses:

### Pause `Говорить RU=>EN`

- translation muted
- remote side hears original Russian passthrough

### Resume `Говорить RU=>EN`

- remote side hears English synthesized translation

### Pause `Слушать EN=>RU`

- Russian translation muted
- original English remains available

### Resume `Слушать EN=>RU`

- Russian translation audible again

## 13. Startup Expectations

At app start:
- status window should open
- `EN => RU whisper.cpp` should go to ready
- `RU => EN faster-whisper` should go to ready as embedded runtime

At pipeline start:
- translator loopbacks should be cleaned first
- `listen_engine` starts
- `speak_engine` starts
- original English loopback starts from `translator_listen.monitor` to real headphones sink

If user later connects new hardware:
- `Обновить устройства` should refresh routes
- real mic auto-selection should prefer JBL/headset-like input over webcam-like inputs

## 14. Common User Test Patterns

### YouTube / English content test

User action:
- set Chrome / YouTube playback to `TranslatorListen`

Expected result:
- hears Russian in headphones

### Telegram / outgoing speech test

User action:
- set Telegram microphone to `TranslatorMicrophone`
- speak Russian

Expected result:
- Telegram receives English translation

### Dual mode live conversation

Expected:
- incoming English audio translated locally
- outgoing Russian speech translated outward
- two branches do not leak into each other

## 15. Practical Resume Guidance For Next AI

If resuming next session:

1. Read this file first.
2. Assume current architecture is valid and intentional.
3. Do not reintroduce old Riva/Canary/NIM experiments.
4. Treat PulseAudio routing as the most fragile subsystem.
5. Use `logs/app.log` before guessing.
6. Prefer minimal, targeted fixes over architecture churn.

If a routing issue appears again:
- first inspect `Audio routing | ...` lines
- inspect translator loopback cleanup lines
- verify which app is routed to `TranslatorListen`
- verify remote comm app microphone is `TranslatorMicrophone`

## 16. Canonical Docs

Human-facing current doc:
- [README.md](/media/yaroslav/DATA/Мой%20переводчик/README.md)

AI-facing current doc:
- [SESSION_CONTEXT_REALTIME_VOICE_TRANSLATOR.md](/media/yaroslav/DATA/Мой%20переводчик/SESSION_CONTEXT_REALTIME_VOICE_TRANSLATOR.md)
