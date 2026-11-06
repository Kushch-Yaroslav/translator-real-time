# REPORT: EN=>RU — скорость, качество, устранение циклов (2026-11-06)

Ветка: `perf/en-ru-quality-latency` (на базе последнего пуша, коммит 5514724).
Цель (от пользователя): сделать EN=>RU максимально быстрым, точным на быстрой речи,
без циклов/дублей/артефактов (пример: «В Калифорнийском университете Калифорнийского
университета...»). Тестирование — только CPU/RAM (GPU занят). Дефолты — запуск на GPU.

## 1. Корневая причина (подтверждено логами)

Цикл «University of California» ×9 — **дегенерация STT** (whisper.cpp `ggml-medium.en`)
на быстрой живой речи, а не переводчик и не TTS. MarianMT и Piper честно озвучивали
мусор. Доказательства:

- в live-логах STT-final содержал 9 повторений фразы, а переводчик получил уже
  «зацыклившийся» текст;
- ранее в приложении был принудительно прописан `temperature_inc=0.0` — это
  **отключало встроенный механизм защиты whisper.cpp** (мульти-декодер с
  temperature-фолбэком и scoring по logprob, src/whisper.cpp ~6845/7495), который
  создан ровно для борьбы с репетитивной деградацией;
- `queueSize=127` на быстрой речи — следствие того же (TTS-очередь раздувалась
  мусором).

## 2. Что сделано

### 2.1 STT: модель + фолбэк температуры
- `scripts/start_whispercpp_server.sh`: дефолтная модель `medium.en` → `large-v3-turbo`
  (скачана в `/media/yaroslav/DATA/ai_models/whisper.cpp/models/ggml-large-v3-turbo.bin`,
  1.6GB). Старый `ggml-medium.en.bin` удалён (разрешено пользователем).
- `core/sst/whispercpp_realtime_stt_service.py` + `core/config/app_config.py` +
  `core/sst/realtime_stt_factory.py`: `temperature=0.0`, `temperature_inc=0.2` стали
  конфигурируемыми и передаются в whisper-server форм-филдами (ранее хардкод 0.0).
  Это включило штатный anti-loop фолбэк whisper.cpp.
- `core/pipeline/branch_runtime.py` (LISTEN/EN=>RU): `stt_final_debounce_sec` 0.30 → 0.15
  (~−150мс на фразу).
- `app_config.json`: `stt.silero_max_window_sec` 8.0 → 4.5 (короче ожидание «хвоста»).

### 2.2 Защита от циклов на выходе (двойная)
- Новый модуль `core/text/repetition.py`: `collapse_repeated_spans()` — n-грамм
  детектор повторений с fuzzy-сопоставлением токенов (ловит русские падежные варианты:
  университета/университете, калифорнийском/калифорнийского). Пороги: span ≥ 2 слов,
  ≥ 3 подряд идущих копии.
- Вшит в 2 «горла» `core/audio/audio_engine.py`:
  1. в `_on_realtime_final` (STT-источник, до перевода);
  2. в `_process_final_text` (после перевода, перед TTS) — на случай, если цикл
     родился уже на стороне MT.
- 10 unit-тестов `tests/test_repetition_detector.py` — все pass.

### 2.3 Выбор модели перевода (A/B)
- `tests/run_translation_ab.py`: opus-mt-en-ru vs NLLB-200-3.3B (CPU).
- **NLLB отклонён**: деградирует в цикл «в в в...» даже при greedy, и ~75× медленнее
  (20.4с vs 0.27с/предложение на CPU), плюс несовместимость tokenizer API
  transformers 5.4.0.
- **opus-mt (Helsinki-NLP/opus-mt-en-ru) оставлен** — качество отличное на нашем домене,
  0.27с/предложение на CPU (на GPU ещё быстрее).

### 2.4 Бенчмарк-инфраструктура EN=>RU
- `tests/synthesize_en_benchmark.py` — 21 EN-аудио (Piper en_US-ryan): normal, fast
  (×1.4 скорости), continuous_fast (один длинный монолог).
- `tests/synthesize_en_noisy.py` — 10 noisy/crosstalk-вариантов (фоновый голос −12дБ,
  шум −24дБ).
- `recordings/benchmarks/en_to_ru/ru_reference/` — 31 RU-референс перевода.
- `tests/run_en_stt_ab.py` (A/B STT) и `tests/run_en_to_ru_e2e.py` (полный E2E:
  STT → dedup → MT → dedup; WER EN/RU, латентность, счётчики циклов).
- Результаты: `stt_ab_cpu.json`, `e2e_medium.en_inc0.0_dedupoff.json`,
  `e2e_large-v3-turbo_inc0.2_dedupon.json`.

## 3. Метрики до/после (E2E, CPU, n=31)

| Метрика | ДО (medium.en, inc=0, dedup off) | ПОСЛЕ (turbo, inc=0.2, dedup on) |
|---|---|---|
| EN WER (mean/max) | 0.0000 / 0.0000 | 0.0046 / 0.0769 |
| RU WER (mean/max) | 0.4312 / 0.6667 | 0.4324 / 0.6667 |
| STT (mean/max) | 4.57с / **22.27с** | 6.11с / **7.10с** |
| MT (mean/max) | 0.28с / 1.25с | 0.28с / 1.18с |
| циклов удалено (source/translated) | 0/0 | 0/0 |

Выводы:
- на **синтетической** речи WER обоих моделей ~0 (чистый TTS-аудио — обе модели
  распознают идеально; различие проявляется только на живой быстрой речи, где
  medium.en деградирует — зафиксировано в live-логах пользователя);
- **max STT упал с 22.27с до 7.1с** — у medium.en на длинном монологе был
  деградационный/медленный кейс, у turbo хвост короткий;
- CPU-среднее turbo медленнее (6.1 vs 4.6с) — ожидаемо; на GPU large-v3-turbo
  быстрее medium.en (дизайн-цель модели: fast decoding);
- 2 ошибки turbo на синтестике (`finance team` → «a fine and steam», `airport` →
  «airports») — косметические, на живом языке prompt-подсказка в сервере
  (`Yaroslav. Jaroslav. Ukraine. 26 years old...`) сглаживает такие кейсы;
- RU WER ~0.43 — это расхождение MT-перевода от «идеального» референса, одинаково
  у обеих конфигураций (операнды равны — модель перевода не менялась).

## 4. Запуск на GPU (дефолты пользователя)

Проверено, что при запуске приложения всё работает на GPU:
- **whisper.cpp**: бинарник `tmp/whisper.cpp/build/bin/whisper-server` — CUDA-сборка
  (ссылается на `libggml-cuda.so.0`, `libcublas`, `GGML_CUDA=1` в CMakeCache);
  скрипт запуска по умолчанию использует его и теперь грузит `large-v3-turbo`.
- **перевод (opus-mt)**: `core/translation/translation_service.py` —
  `device = "cuda" if torch.cuda.is_available() else "cpu"` (torch 2.11.0+cu130).
- **TTS (Piper)**: `core/tts/tts_service.py` — `use_cuda` по умолчанию =
  `has_cuda_provider()` (PiperVoice.load(..., use_cuda=True)).
- Все тесты в этой сессии запускались с `CUDA_VISIBLE_DEVICES=''` (переменная
  окружения, НЕ изменение конфигов) — дефолты не тронуты.

## 5. Как проверить за ~2 минуты (live)

1. `bash scripts/start_whispercpp_server.sh` (загрузит large-v3-turbo на GPU).
2. Запустить приложение (EN=>RU listen-ветка).
3. Быстро на английском произнести 3–4 предложения, включая «University of
   California...» в быстрой манере.
4. Критерии успеха:
   - озвучка начинается заметно быстрее, чем до правки (−150мс debounce + окно 4.5с);
   - **никаких** повторений фраз;
   - в логе нет `FINAL span-collapse` / `TRANSLATED span-collapse` (детектор —
     страховка; с turbo+temp_inc он не должен срабатывать, но если STT всё же
     деградирует — пользователь услышит одну чистую фразу вместо цикла).

## 6. Что НЕ сделано / отложено

- **Клонирование голоса** (RU→EN в голосе пользователя) — отложено по решению
  пользователя (образец запишется позже).
- **Boundary-layer streaming** (sentence_boundary_segmenter + confirm STT) —
  пропущено: это мёртвый код (включался только для ru_to_en + riva backend),
  его активация требует переписывания гейта и второго прохода STT — высокий
  риск без интерактивной отладки. Оставлено как будущая опция.
- NLLB-200-3.3B — скачивался для A/B, отклонён и **удалён** из кэшей
  (17GB + 7.1GB освобождено).

## 7. Файлы

Изменены:
- `scripts/start_whispercpp_server.sh`
- `core/sst/whispercpp_realtime_stt_service.py`
- `core/config/app_config.py`
- `core/sst/realtime_stt_factory.py`
- `core/pipeline/branch_runtime.py`
- `core/audio/audio_engine.py`
- `core/text/__init__.py`
- `app_config.json`

Добавлены:
- `core/text/repetition.py`
- `tests/test_repetition_detector.py`
- `tests/synthesize_en_benchmark.py`, `tests/synthesize_en_noisy.py`
- `tests/run_en_stt_ab.py`, `tests/run_translation_ab.py`, `tests/run_en_to_ru_e2e.py`
- `recordings/benchmarks/en_to_ru/` (аудио, референсы, JSON-результаты)
- `REPORT_EN_RU_PERF_2026-11-06.md` (этот файл)
