# Мой переводчик

Локальное desktop-приложение для перевода живой речи `EN -> RU`:

- берет звук с микрофона
- распознает английскую речь через локальный `NVIDIA NIM`
- переводит текст локально через `MarianMT`
- синтезирует русский голос локально через `Piper`
- выводит результат в виртуальный микрофон `TranslatorMic`, который можно выбрать в Telegram

Текущая стабильная ветка ориентирована именно на сценарий:

`английская речь -> русский голос -> Telegram / голосовые сообщения / звонки`

## Что нужно для работы

### 1. ОС и аудио

Приложение рассчитано на Linux с `PulseAudio` или `PipeWire` и установленной утилитой `pactl`.

Используется виртуальный sink `translator_mic`, который приложение создает автоматически.

### 2. Python

Рекомендуемая версия: `Python 3.12`

Проект запускается из виртуального окружения `.venv`.

### 3. Docker + NVIDIA runtime

Для STT используется локальный контейнер `NVIDIA NIM`:

- контейнер: `parakeet-1-1b-ctc-en-us`
- порты: `9000` и `50051`
- GPU runtime: `nvidia`

Приложение умеет автоматически:

- проверить, поднят ли NIM
- при необходимости запустить Docker-контейнер само
- взять `NGC_API_KEY` из окружения
- если переменная не задана, попытаться взять ключ из `.env`

### 4. Видеокарта

Минимально приложение может работать и без GPU для части задач, но нормальный рабочий режим предполагает NVIDIA GPU.

Рекомендуемо:

- `RTX 3090 24 GB` или аналогичный уровень
- установленный драйвер NVIDIA
- рабочий `docker --runtime=nvidia`

Что использует GPU:

- `NVIDIA NIM` для STT
- `Piper` через `onnxruntime-gpu` для TTS

### 5. Локальные модели

#### STT

Используется `NVIDIA NIM`:

- `parakeet-1-1b-ctc-en-us`

Кэш по умолчанию:

- `/media/yaroslav/DATA/nim-cache`

#### TTS

Используется `Piper`.

Текущий голос по умолчанию:

- `ru_RU-dmitri-medium`

Каталог голосов по умолчанию:

- `/media/yaroslav/DATA/ai_models/piper`

Если нужного голоса нет, `Piper` скачает его через Hugging Face при первом использовании.

## Что нужно положить в `.env`

Файл `.env` находится в корне проекта.

Минимально нужен один из вариантов:

```env
NGC_API_KEY=...
```

или

```env
NV_API_KEY=...
```

Сейчас проект поддерживает оба имени, но для новой машины лучше использовать именно `NGC_API_KEY`.

`.env` добавлен в `.gitignore` и не должен попадать в репозиторий.

## Запуск

Из корня проекта:

```bash
.venv/bin/python main.py
```

Что происходит при запуске:

1. Приложение проверяет, доступен ли локальный NIM на `localhost:9000`
2. Если нет, пытается поднять контейнер `parakeet-1-1b-ctc-en-us`
3. После готовности NIM запускается Qt UI

Если Docker у пользователя работает только через `sudo`, приложение попробует использовать `sudo` автоматически.

## Настройка в Telegram

После старта приложения:

1. В приложении выбери свой реальный микрофон как `Input microphone`
2. В приложении выбери `translator_mic` как `Output sink for app`
3. Нажми `Start pipeline`
4. В Telegram выбери устройство записи:

`Monitor of TranslatorMic`

Именно этот monitor получает синтезированный русский голос.

## Конфигурация

Основные параметры вынесены в файл:

- [app_config.json](./app_config.json)

Секции:

### `audio`

```json
"audio": {
  "samplerate": 48000,
  "channels": 1,
  "blocksize": 1024
}
```

### `stt`

```json
"stt": {
  "base_url": "http://localhost:9000",
  "ws_url": "ws://localhost:9000/v1/realtime?intent=transcription",
  "language": "en-US",
  "sample_rate_hz": 16000,
  "num_channels": 1,
  "timeout": 10.0,
  "commit_interval_sec": 0.5,
  "enable_automatic_punctuation": true,
  "final_debounce_sec": 0.6,
  "partial_emit_enabled": true,
  "partial_stability_sec": 0.45,
  "partial_min_words": 4,
  "noise_gate_threshold": 0.009,
  "noise_gate_hangover_sec": 0.35
}
```

Что здесь важно:

- `commit_interval_sec`: как часто коммитится аудиобуфер в realtime STT
- `final_debounce_sec`: задержка перед отправкой финального сегмента в перевод/TTS
- `partial_emit_enabled`: можно ли начинать перевод по стабильным `partial`
- `partial_stability_sec`: сколько ждать, чтобы partial считался стабильным
- `partial_min_words`: минимальная длина partial для раннего перевода
- `noise_gate_threshold`: подавление очень тихих шумов/дыхания
- `noise_gate_hangover_sec`: защита, чтобы не откусывать окончания слов

### `translation`

```json
"translation": {
  "direction": "en_to_ru",
  "enabled": true
}
```

Сейчас стабильный рабочий режим:

- `en_to_ru`

### `tts`

```json
"tts": {
  "voice_name": "ru_RU-dmitri-medium",
  "data_dir": "/media/yaroslav/DATA/ai_models/piper",
  "use_cuda": null,
  "max_queue_latency_sec": 0.75
}
```

Что важно:

- `use_cuda: null` означает автоопределение GPU backend
- `max_queue_latency_sec` ограничивает накопление очереди озвучки

## Полезные сценарии настройки

### Telegram / голосовые

Более быстрый отклик:

```json
"commit_interval_sec": 0.35,
"final_debounce_sec": 0.45,
"partial_stability_sec": 0.30,
"partial_min_words": 3
```

### Более стабильный, но менее быстрый режим

```json
"commit_interval_sec": 0.5,
"final_debounce_sec": 0.6,
"partial_stability_sec": 0.45,
"partial_min_words": 4
```

## Диагностика

Если приложение не стартует:

1. Проверь, что работает Docker
2. Проверь, что `NGC_API_KEY` или `NV_API_KEY` есть в `.env`
3. Проверь, что `localhost:9000` не занят другим процессом
4. Проверь, что у Docker есть доступ к NVIDIA GPU
5. Проверь, что каталог кэша NIM существует и доступен:

`/media/yaroslav/DATA/nim-cache`

Если TTS не идет на GPU:

1. Проверь, что установлен `onnxruntime-gpu`
2. Проверь, что `onnxruntime` видит `CUDAExecutionProvider`
3. В логах при старте должно быть:

`TTS voice loaded (backend=cuda)`

## Текущее состояние проекта

Стабильно отработано:

- `EN -> RU`
- локальный STT через `NVIDIA NIM`
- локальный TTS через `Piper`
- автозапуск NIM при старте приложения
- работа через виртуальный микрофон для Telegram

Следующие направления развития:

- `RU -> EN`
- двусторонний режим
- UI-настройки для конфигурации
- логирование в файл
- отдельная ветка для агрессивного снижения latency
