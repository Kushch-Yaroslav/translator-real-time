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

## Accepted Current State 2026-04-30

Текущий лучший live baseline для `RU => EN`: состояние после `speak_ru_to_en(23).log`.

Принято:

- latest live test на `large_simple_text` не показал major duplicate issue;
- weak draft `это проект для конвертации` теперь корректно пропускается и не озвучивается перед полной фразой;
- normalization / rewrite fixes приняты:
  - `пет-проект`;
  - `лендинги`;
  - `но ... лимит` -> `но я сталкивался с лимитом`;
  - `которым активно пользуюсь...` -> `я активно пользуюсь этим...`;
- `partial rewind suffix suppression` принят как полезная защита от standalone tail artifacts.

Оставшиеся проблемы сейчас относятся в основном к semantic translation quality, а не к duplicate pipeline:

- `так как мой английский слабее уровня комфортного общения` всё ещё может переводиться неестественно;
- `который помог мне невероятно быстро ускорить работу` может сокращаться до неудачного `The one who helped me incredibly fast.`

Не возвращать отклонённые направления без отдельного решения:

- `early-stable partial`;
- `LLM / hybrid`;
- `defer / hold / buffering`;
- broad filters;
- `adjacent translated refinement filter`.

Следующая работа должна быть сфокусирована на semantic quality нескольких известных фраз:

- только маленькие deterministic rewrite / normalization шаги;
- по одной фразе или одному узкому паттерну за итерацию;
- обязательно benchmark-driven и затем live-validated.

## Current State 2026-04-29

Текущий принятый baseline: `RU=>EN clean baseline v2`.

Состояние baseline:

- `early-stable partial` выключен и не должен возвращаться без отдельного решения.
- `phrase_seen stale_after_sec = 0.8` остаётся принятым latency/stability компромиссом.
- Playback-level logs добавлены и полезны для live анализа:
  - `PLAYBACK queued`
  - `PLAYBACK started`
  - `PLAYBACK finished`
  - `PLAYBACK skipped`
  - `PLAYBACK merged`
- `LLM / hybrid` ветка была протестирована отдельно, архивирована и не merge-илась в стабильный baseline.
- `defer / hold` эксперименты отклонены и удалены.
- `final-tail filter` отклонён и удалён.
- `adjacent translated refinement filter` отклонён и удалён.

Текущая принятая стратегия:

- не менять admission / segmentation / TTS;
- не возвращать hold / defer / merge;
- улучшать качество через deterministic `RU source normalization / rewrite` перед переводом;
- изменения должны быть узкими, exact-match, benchmark-driven и проверяться live.

Принятый deterministic rewrite перед translation:

```text
но упирался то в лимит -> но я сталкивался с лимитом
но опирался то в лимит -> но я сталкивался с лимитом
```

Причина принятия:

- live / offline показали более естественный результат:
  - `But I ran into a limit,`
- latency и duplicates не ухудшились заметно.

Текущая glossary normalization перед admission / translation:

```text
перевозчик -> переводчик
пэт проект -> пет-проект
педпроект -> пет-проект
подпроект -> пет-проект   # только если context содержит "еще один" / "ещё один"
лайтинги -> лендинги
лейтинги -> лендинги
лайндинги -> лендинги
ландинги -> лендинги
лэндинги -> лендинги
```

Последний `large_simple_text` benchmark:

```text
queue_start_delay_sec: 0.80
tts_ready_start_delay_sec: 3.25
duplicate_translated_count: 0.00
long_translation_gaps_count: 6.00
realtime_factor: 1.07
```

Оставшиеся проблемы:

- STT instability всё ещё создаёт варианты типа `лейдинги / лайтинги / лендинги`.
- Остаются causal/dependent fragments вокруг `так как мой английский...`.
- Некоторые dependent fragments всё ещё переводятся неестественно.
- Нужен live sanity test после reset состояния приложения.

Следующий рекомендуемый шаг:

1. Сделать live sanity test с текущей normalization / rewrite.
2. После live теста анализировать только `PLAYBACK started` lines.
3. Если regressions нет, зафиксировать этот baseline как следующий stable candidate.

## Benchmark System (RU→EN)

### 1. Назначение
Бенчмарк предназначен для автоматизированной проверки качества работы пайплайна перевода `RU => EN` в оффлайн-режиме.
Он позволяет измерять:
- **Качество распознавания и перевода:** сравнение финального текста с эталонным (`expected.txt`).
- **Стабильность:** наличие дубликатов в очередях и финальном тексте.
- **Задержки (Latency):** время до появления первой речи, первой очереди, первого перевода и готовности TTS.
- **Производительность:** `realtime_factor` (насколько быстрее/медленнее реального времени идет обработка).

Используется как обязательный приемочный тест после любых изменений в логике `AudioEngine`, `StreamController` или сервисах перевода/TTS.

### 2. Структура файлов
```text
recordings/benchmarks/ru_to_en/
  source/
    simple_text/
      simple_text.wav             # Аудио эталон
      simple_text.expected.txt    # Текст эталон
    natural_speech/
      natural_speech.wav
      natural_speech.expected.txt
    pauses_and_hesitation/
      pauses_and_hesitation.wav
      pauses_and_hesitation.expected.txt
    difficult_phrases/
      difficult_phrases.wav
      difficult_phrases.expected.txt
    noisy_or_unclear/
      noisy_or_unclear.wav
      noisy_or_unclear.expected.txt
    large_simple_text/
      large_simple_text.wav
      large_simple_text.expected.txt

  runs/
    run_YYYY-MM-DD_HH-MM-SS/      # Папка конкретного прогона
      summary.json                # Общий отчет по всем тестам в прогоне
      comparison.json             # Сравнение с предыдущим прогоном
      <category>/
        <file>.json               # Детальные метрики файла
        <file>.log                # Лог событий обработки
```
- **source**: неизменяемые эталонные данные.
- **runs**: результаты запусков. Система хранит только **3 последних прогона**, старые удаляются автоматически.

### 3. Категории
- **simple_text** — "Простой текст": короткие, понятные фразы без сложных пауз.
- **natural_speech** — "Естественная речь": обычный темп, нормальная разговорная речь.
- **pauses_and_hesitation** — "Паузы и запинки": речь с остановками, повторами, неидеальным темпом.
- **difficult_phrases** — "Сложные фразы": длинные предложения, термины, числа, вкрапления английских слов.
- **noisy_or_unclear** — "Шумная / нечеткая речь": плохая запись, фоновый шум, нечеткая дикция.
- **large_simple_text** — "Большой простой текст": длинный связный текст без пауз и запинок для проверки семантических дубликатов на длинной дистанции.

### 4. Как использовать
1. **Запись:** Открыть `Benchmark Recorder`, выбрать категорию, ввести `Expected Text`, нажать `Record`.
2. **Сохранение:** После остановки файл сохранится автоматически с именем категории в нужную папку `source`.
3. **Запуск:**
   - В UI: кнопки "Запустить текущий" (выбранная категория) или "Запустить все".
   - В терминале: `python tests/run_ru_to_en_offline_benchmark.py --all`.

### 5. Поведение системы
- **Изоляция:** `source` файлы никогда не меняются при прогонах бенчмарка.
- **Ротация:** При создании 4-го прогона, самый старый (1-й) удаляется целиком.
- **Сравнение:** `comparison.json` создается автоматически, если есть хотя бы один предыдущий прогон.

### 6. Параллельный запуск
- Прогон идет в **offline режиме**: аудио подается напрямую в pipeline (без микрофона и виртуальных устройств).
- Используется параллельная обработка файлов для ускорения тестов.
- **ВАЖНО (Ситуация с Latency):** Было замечено, что при `concurrency > 1` (например, 2 или более) метрики `delay_sec` значительно ухудшаются (на 0.5 - 1.5 сек) из-за конкуренции процессов за GPU и CPU. Это не является деградацией кода, а является артефактом самого процесса тестирования.
- **Concurrency**: по умолчанию 2. Для точных замеров задержек использовать 1.
- Логи и результаты каждого файла полностью изолированы.

### 7. Метрики
Основные технические метрики:
- `queue_start_delay_sec`: задержка до первой постановки в очередь.
- `translate_start_delay_sec`: задержка до получения первого перевода.
- `tts_ready_start_delay_sec`: задержка до готовности первого TTS-чанка.
- `total_pipeline_time_sec`: общее время обработки файла.
- `realtime_factor`: отношение времени обработки к длительности аудио (меньше = лучше, < 1.0 — быстрее реального времени).
- `duplicate_queued_count` / `duplicate_translated_count`: количество повторов.
- `long_translation_gaps_count`: количество аномальных пауз между фразами.
- `word_count_diff`: разница в количестве слов между финальным текстом и эталоном.

**Примечание по параллелизму:** При параллельном запуске нескольких тестов (`concurrency > 1`) метрики задержки (latency) могут ухудшаться из-за конкуренции процессов за GPU/CPU. Для получения эталонных (чистых) значений задержки рекомендуется использовать последовательный запуск (`--concurrency 1`).

### 8. Сравнение прогонов (Comparison)
Файл `comparison.json` содержит анализ изменений:
- Каждая метрика получает статус: `improved` (лучше), `worse` (хуже), `same` (без изменений) или `unknown`.
- Помогает мгновенно понять, как правка кода повлияла на производительность и качество.

## Benchmark Usage Rule (for AI)

ИИ (Junie/Codex) ОБЯЗАН:
1. **После любых изменений** в pipeline или логике обработки звука/текста запускать полный бенчмарк строго последовательно для точности метрик: `python tests/run_ru_to_en_offline_benchmark.py --all --concurrency 1`.
2. **Анализировать `summary.json` и `comparison.json`** последнего прогона.
3. **Не считать задачу выполненной**, если:
   - Выросла любая из метрик `delay_sec` (ухудшилась latency).
   - Увеличилось количество дубликатов (`duplicates`).
   - Появились новые `long_translation_gaps`.
   - Резко вырос `realtime_factor`.
4. В отчете пользователю всегда указывать, как изменились метрики по сравнению с предыдущим состоянием.

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

## Latest Iteration (Live-Validated Baseline)

### What Was Successfully Solved

В последней стабильной итерации были успешно устранены:

- exact duplicates;
- same-prefix retries, включая кейсы вида `AEMDays / AMDiys / I'm days`;
- suffix/contained duplicates, включая кейсы уровня `language barrier` и `landing logic`.

### Stabilized Pipeline State

Текущий baseline после live-проверки показывает:

- быстрый старт, примерно `~0.7–1.5 сек`;
- допустимые межфразовые паузы, примерно `~до 1–1.5 сек`;
- отсутствие дублей в live Telegram тесте;
- предсказуемое и стабильное поведение pipeline.

### Incomplete-Fragment Hold/Merge Experiment

Была отдельная попытка улучшить semantic quality через `incomplete-fragment hold/merge` перед translation:

- был добавлен короткий `hold` перед переводом;
- offline benchmark не показал явных регрессий по safety-метрикам;
- но в live Telegram тесте:
  - pipeline стал нестабильным;
  - появились out-of-order сегменты;
  - ухудшилось общее качество речи.

### Decision

Решение по итогам live-проверки:

- все изменения `incomplete-fragment hold/merge` были полностью откатаны;
- система зафиксирована на предыдущем стабильном baseline.

### Current System State

На текущий момент:

- дубли устранены;
- latency находится в норме;
- pipeline стабилен;
- основная оставшаяся проблема: `semantic quality` на коротких незавершённых сегментах.

Типовые примеры оставшейся проблемы:

- `но упирался то в лимит,` + `то в отсутствие.` -> плохой перевод;
- `который помог мне невероятно.` + `быстро ускорить работу.` -> потеря смысла.

### Key Conclusion

Ключевой вывод этой итерации:

- проблема находится не в duplicates;
- корень проблемы сейчас в `segmentation / fragmentation`;
- naive `hold/merge` ломает pipeline и не подходит как решение.

### Validation Rule

Нужно явно зафиксировать:

1. Live Telegram тест является финальным источником истины.
2. Offline benchmark используется как safety check, но не как единственный критерий качества.

### Rules For Future Changes

## LLM Experiment Archive

### Branch / Git State

- Стабильный baseline был сохранён без принятия LLM-экспериментов в основную рабочую ветку.
- Ветка `experiment/llm-translation` тестировалась отдельно как исследовательская.
- Экспериментальная ветка была сохранена в истории как архив исследования.
- После завершения тестов работа продолжена снова из исходного stable baseline branch.
- Qwen model files после завершения эксперимента были удалены из `/media/yaroslav/DATA/ai_models`.

### What Was Tested

Тестировался hybrid `RU=>EN` backend:

- `MarianMT` как default fast backend;
- optional local LLM только для dependent fragments.

Проверенные варианты:

- full transformers `Qwen2.5-1.5B-Instruct`;
- full transformers `Qwen2.5-7B-Instruct`;
- quantized `Qwen2.5-7B-Instruct` GGUF через `llama.cpp / llama-cpp-python`.

### Results

- `Qwen2.5-1.5B-Instruct` технически работал, но качество перевода оказалось слишком слабым.
- full transformers `Qwen2.5-7B-Instruct` давал лучшее качество, но вызывал проблемы по `VRAM / OOM` в live runtime.
- В экспериментальной ветке был исправлен lifecycle bug:
  - shared singleton;
  - preload;
  - reuse;
  - без per-fragment model reload.
- Quantized GGUF `7B` успешно загружался и работал без `fallback / OOM`.
- В live logs действительно появлялись:
  - `HYBRID translation route: llm`
  - `HYBRID llm translation result`

### Key Conclusion

Ключевой итог эксперимента:

- LLM backend технически заработал после перехода на quantized backend.
- Но live quality всё равно оставалась плохой.
- Главная оставшаяся проблема находится не в качестве translation model как таковой.
- Главная проблема находится выше по pipeline:
  - source fragmentation;
  - admission;
  - semantic duplicate / refinement emissions.
- Плохие или нестабильные русские source fragments попадают в `LOWLAT sentence queued` ещё до перевода.
- LLM не может надёжно исправить плохой source stream без `context / hold / merge`.

### Problematic Live Behaviors

Типовые проблемные live cases:

- draft partial emitted too early:
  - `Я его создал.`
  - затем позже `Я его создал для помощи...`
- bad STT / refinement fragment:
  - `так как мой английский язык не использовал`
- repeated / refined semantic fragments вокруг:
  - `но упирался то в лимит, то в отсутствие гибкости`
- final cleanup всё ещё мог выпускать лишний refined-content:
  - `мы активно пользуюсь до сих пор`

### Decision

По итогам эксперимента:

- не merge-ить LLM experiment в stable baseline;
- не продолжать LLM model work как основное направление прямо сейчас;
- сохранить hybrid experiment только как archived research;
- текущий принятый вектор:
  - улучшать source stream quality;
  - segmentation / admission;
  - semantic duplicate / refinement handling.

Дополнительные ограничения:

- избегать `hold/merge`, если они не введены под очень жёстким контролем;
- не возвращать `early-stable partial`.

### Next Recommended Work

Следующая сессия должна начинаться с analysis-only задачи:

1. Inspect stable baseline live logs.
2. Точно определить, почему semantic duplicate / refinement source fragments проходят в `LOWLAT sentence queued`.
3. Предложить одно маленькое benchmark-driven исправление.
4. Без model changes.

Любые будущие изменения:

- не должны ломать стабильность pipeline;
- не должны ухудшать latency;
- не должны возвращать дубли;
- должны проверяться через `live + benchmark`.

### Explicitly Forbidden

Запрещено:

- повторно вводить `hold/merge` без строгого контроля времени и порядка;
- добавлять эвристики, которые могут ломать порядок сегментов.

### Next Stage Goal

Цель следующего этапа:

- улучшить `semantic quality` без нарушения текущей стабильности;
- искать решения, которые не требуют удержания сегментов.

## Stopped Filter Experiments

Последние две узкие гипотезы были проверены и в текущем виде остановлены как неполезные.

### 1. Post-Translation Short-Fragment Filter

Что проверяли:

- узкий post-translation фильтр коротких low-confidence translated fragments перед TTS.

Результат:

- benchmark-проверка не показала реальной активации фильтра;
- `skipped fragments = none`.

Решение:

- подход не дал результата в текущем виде;
- дальнейшая работа по этому фильтру остановлена.

### 2. Source-Side Admission Rule

Что проверяли:

- узкое source-side admission rule для `RU => EN` fragments перед `LOWLAT sentence queued`.

Результат:

- benchmark-проверка не показала реальной блокировки target-fragments;
- `blocked fragments = none`.

Решение:

- подход не дал результата в текущем виде;
- дальнейшее усиление этого admission rule запрещено без отдельного разрешения.

### 3. Context-Aware Translation MVP

Что проверяли:

- узкий `RU => EN` translation MVP с коротким previous source context только для ограниченного набора context-sensitive fragments;
- сегментация, очередь, TTS и final cleanup не менялись;
- feature был добавлен под флаг `ru_to_en_context_aware_translation_enabled`.

Результат:

- MVP технически сработал и context реально применялся;
- но context leaked into output:
  модель начала повторять previous context в переводе current fragment;
- ухудшились benchmark safety-метрики latency;
- live Telegram test был пропущен, потому что offline safety-check уже показал неприемлемый результат.

Решение:

- гипотеза не принята;
- feature disabled в baseline (`ru_to_en_context_aware_translation_enabled = False`);
- дальше не развивать эту гипотезу без новой стратегии защиты от `context leakage`.

### 4. Partial Stability Filter MVP

Что проверяли:

- узкий `RU => EN` partial stability filter только на partial-path;
- только для зависимых starters (`как / чтобы / то / который / ...`);
- со строгим `exact-repeat` criterion для `stable accepted`;
- translation, TTS, final cleanup и duplicate filters не менялись.

Результат:

- MVP технически начал работать;
- `unstable skipped` оказалось слишком много;
- `stable accepted` оказалось слишком мало;
- ухудшились `coverage_ratio` и `long_translation_gaps_count`;
- live Telegram test был пропущен, потому что offline safety-check уже показал неприемлемый результат.

Решение:

- partial stability filter MVP не принят;
- baseline возвращён без этой partial-логики;
- `exact-repeat` partial stability дальше не развивать без новой стратегии, потому что критерий слишком строгий и режет полезные fragments.

### 5. Comma-Tail `то ...` Partial-Path MVP

Что проверяли:

- узкий `RU => EN` partial-path MVP только для `comma-produced tail`;
- только pattern `starts with "то"`;
- только короткие fragments (`<= 5` слов);
- final-path, translation, TTS, cleanup и duplicate filters не менялись.

Результат:

- benchmark-проверка не показала реальной активации правила;
- `skipped fragments = 0`;
- заметного влияния на pipeline не было;
- live Telegram test был пропущен как ненужный.

Решение:

- `comma-tail "то ..."` partial-path MVP не принят;
- baseline возвращён без этого exact trigger;
- дальше не развивать этот exact trigger без нового анализа, потому что он не совпал с реальным path появления problematic fragments.

### Current Decision

Нужно явно зафиксировать:

1. Оба фильтрационных подхода (`post-translation filter` и `source-side admission rule`) остановлены как неполезные в текущем виде.
2. `Context-aware translation MVP` тоже не принят и выключен флагом.
3. `Partial stability filter MVP` тоже не принят и откатан.
4. `Comma-tail "то ..."` partial-path MVP тоже не принят и откатан.
5. Текущий рабочий baseline остаётся предыдущим stable live-validated baseline.
6. Следующая работа должна начинаться не с новых фильтров или расширения context-MVP, а с нового анализа / новой гипотезы о root cause.
