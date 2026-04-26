# PROJECT STATUS

## Iteration Rule

Каждая новая итерация строго ограничена:

1. Одно конкретное изменение или один небольшой связанный набор изменений.
2. Обязательный запуск benchmark после изменения.
3. Короткий отчёт с измеримыми результатами.
4. Остановка и ожидание пользовательского фидбека.

Запрещено:

- продолжать оптимизацию без нового запроса;
- смешивать несколько независимых экспериментов в одной итерации;
- одновременно пытаться улучшать все метрики без явного основного фокуса.

## Strategic Goals

1. Ускорить старт озвучки до 1 секунды, максимум 2 секунды.
2. Убрать дубликации, артефакты и ошибки озвучки. Озвучка должна быть качественной.
3. Убрать межфразовые паузы.

Перспективная цель, пока не приоритет:

- после устранения лишних межфразовых пауз вернуть контролируемые паузы для более естественной речи.

## Current Pipeline

Высокоуровнево текущий `RU => EN` путь такой:

1. Realtime STT получает partial/final.
2. `AudioEngine` маршрутизирует `RU => EN` low-latency поток.
3. Основной рабочий baseline сейчас снова живёт в `AudioEngine`.
4. Translation service переводит сегменты.
5. TTS synthesizes output.
6. Playback queue озвучивает результат.

Основные файлы:

- [core/audio/audio_engine.py](/media/yaroslav/DATA/Мой%20переводчик/core/audio/audio_engine.py)
- [core/audio/ru_to_en_stream_controller.py](/media/yaroslav/DATA/Мой%20переводчик/core/audio/ru_to_en_stream_controller.py)

## Benchmark

Benchmark уже создан.

Компоненты:

- окно записи benchmark-файла:
  [core/ui/benchmark_recorder_window.py](/media/yaroslav/DATA/Мой%20переводчик/core/ui/benchmark_recorder_window.py)
- headless runner:
  [tests/run_ru_to_en_offline_benchmark.py](/media/yaroslav/DATA/Мой%20переводчик/tests/run_ru_to_en_offline_benchmark.py)
- benchmark paths:
  [core/benchmark/paths.py](/media/yaroslav/DATA/Мой%20переводчик/core/benchmark/paths.py)

Запуск UI:

```bash
python main.py
```

Запуск benchmark-runner:

```bash
./.venv/bin/python tests/run_ru_to_en_offline_benchmark.py
```

## Benchmark Audio

Эталонный benchmark-файл один:

- [current_benchmark.wav](/media/yaroslav/DATA/Мой%20переводчик/recordings/benchmarks/ru_to_en/source/current_benchmark.wav)

Файл перезаписывается через benchmark recorder окно.

Каталоги:

- source:
  [recordings/benchmarks/ru_to_en/source](/media/yaroslav/DATA/Мой%20переводчик/recordings/benchmarks/ru_to_en/source)
- runs:
  [recordings/benchmarks/ru_to_en/runs](/media/yaroslav/DATA/Мой%20переводчик/recordings/benchmarks/ru_to_en/runs)

## Metrics

Сейчас используем такие метрики:

- старт первой озвучки;
- старт первого queued сегмента;
- старт первого translated сегмента;
- дубли queued сегментов;
- дубли translated сегментов;
- длинные TTS сегменты;
- большие gaps между translated сегментами;
- субъективная проверка качества смысла на ключевых фразах.

Целевые направления интерпретации:

- latency;
- duplicates;
- inter-phrase pauses;
- semantic quality.

## Last Measured Results

Полноценные benchmark-runner прогоны уже получены.

Текущий baseline после отката к старому быстрому состоянию:

- benchmark report:
  [offline_benchmark_2026-04-26_21-44-57.json](/media/yaroslav/DATA/Мой%20переводчик/recordings/benchmarks/ru_to_en/runs/offline_benchmark_2026-04-26_21-44-57.json)
- benchmark log:
  [offline_benchmark_2026-04-26_21-44-57.log](/media/yaroslav/DATA/Мой%20переводчик/recordings/benchmarks/ru_to_en/runs/offline_benchmark_2026-04-26_21-44-57.log)

Ключевые baseline-метрики:

- `queue_start_delay_sec = 0.793`
- `translate_start_delay_sec = 1.940`
- `tts_ready_start_delay_sec = 3.002`
- `duplicate_queued_segments = 0`
- `duplicate_translated_segments = 0`

Вывод по baseline:

- старт озвучки быстрый;
- формальных дублей по строкам нет;
- но остаются semantic duplicates и weak-to-refined pattern:
  сначала выходит слабая ранняя версия фразы, потом более полная версия той же мысли;
- межфразовые паузы остаются системной проблемой по всему длинному тексту.

## Rolled Back Experiments

Была отдельная серия экспериментов по устранению дублей и паттерна `1 2 3 1 4 5`.

Что пробовали:

- возврат к старому быстрому pipeline и точечные правки внутри `AudioEngine`;
- sequential cursor для already-emitted source fragments;
- `upgrade-last-phrase` и `replace-last-emitted-fragment`;
- локальный модуль reconstruction из набора коротких фрагментов:
  [core/text/phrase_fragment_reconstructor.py](/media/yaroslav/DATA/Мой%20переводчик/core/text/phrase_fragment_reconstructor.py);
- интеграцию reconstruction в `RU => EN` final-flush;
- серию узких `partial-defer` фильтров для weak partial fragments.

Почему эти изменения отменены:

- они не дали устойчивого общего решения;
- часть из них была слишком эвристической и местами приближалась к подстройке под конкретный benchmark-текст;
- главное: при подавлении слабых ранних фраз заметно росли межфразовые паузы;
- проблема смещалась из `повторяет старое` в `слишком долго ждёт более хороший кусок`.

Итог отменённой ветки:

- некоторые semantic duplicates действительно удавалось локально убрать;
- но цена за это была слишком высокой:
  хуже latency старта следующих сегментов и больше системных gaps по всему тексту;
- поэтому эти изменения не приняты как основное направление и baseline был возвращён к более быстрому состоянию.

## Current Known Problems

1. Быстрый baseline всё ещё даёт semantic duplicates:
   - weak phrase -> refined phrase;
   - локальные возвраты к уже начатой мысли.
2. Межфразовые паузы остаются системной проблемой на длинном benchmark-тексте.
3. Последняя серия эвристических fixes отменена, потому что уменьшала дубли ценой заметного роста gaps.
4. Нужна более общая стратегия восстановления фраз без hardcode и без роста пауз.

## Current Focus

Текущий ближайший фокус:

1. Сохранить быстрый baseline по latency.
2. Найти общую стратегию борьбы с semantic duplicates без hardcode.
3. Не допускать роста межфразовых пауз при следующих итерациях.
4. Продолжать только малыми benchmark-driven изменениями по одной гипотезе за раз.
