# RU->EN Offline Benchmark

Сюда кладётся один эталонный русский аудиофайл для повторяемого прогона
`RU => EN` пайплайна без микрофона.

Рекомендуемая структура:

- `source/current_benchmark.wav`
  - основной эталонный файл
  - его будет перезаписывать отдельное desktop-окно benchmark recorder
- `runs/`
  - сюда скрипт будет складывать JSON-отчёты и текстовые логи по каждому прогону

Основной сценарий:

1. Записать эталонный файл через benchmark recorder окно.
2. После каждого изменения запускать offline-runner.
3. Сравнивать:
   - задержку старта первой озвучки
   - межфразовые паузы
   - список очередей `LOWLAT sentence queued`
   - список `TRANSLATED`
   - дубли и странные хвосты

Запуск:

```bash
python3 tests/run_ru_to_en_offline_benchmark.py
```

По умолчанию runner использует фиксированный файл:

`recordings/benchmarks/ru_to_en/source/current_benchmark.wav`

Скрипт использует текущий runtime-профиль `speak` и старается идти через те же
компоненты, что и живой `AudioEngine`:

- realtime STT
- low-latency orchestration
- translation
- TTS

При этом микрофон и аудиоустройства не требуются.
