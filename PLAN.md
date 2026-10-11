# PLAN — staged implementation

## Порядок работы

Day01 завершён 2026-10-09. D2-01 и D2-02 завершены 2026-10-11; следующий пункт — D2-03.
Если пользователь называет ID, выполняем только его; если день —
его зависимые пункты последовательно. Каждый завершённый ID имеет раздел notebook,
проверку и свидетельство ниже. `[ ]` — ещё не завершено, `[x]` — подтверждено.
Будущие дни — roadmap; детализация до реализации, без опережающих features.

## Day01 — данные и EDA

### D1-00 — аудит, контракты и окружение

- [x] Прочитать все исходники, сформировать AGENTS/SPEC/PLAN.
- Цель: согласовать три класса, границы day01 и учебный рабочий цикл.
- Выход: документы, pinning используемых day01 packages, notebook setup.
- Проверка: interpreter/packages доступны; notebook определяет root и импортирует src.
- Самопроверка: чем SPEC отличается от PLAN и AGENTS? Почему notebook вызывает modules?
- Свидетельство: Прочитаны все файлы задания и исходники; AGENTS/SPEC/PLAN созданы. D1-00 notebook выполнен на Python 3.14.6; версии в requirements-day01.txt и reports/day01/environment.json.

### D1-01 — источник и воспроизводимая загрузка

- [x] Подготовить Financial PhraseBank `sentences_75agree` и canonical CSV.
- Зависит от D1-00. Вход: pinned ZIP; выход: `(3453, 2)` с `text`, `sentiment`.
- Сохранить revision, checksum, license, class order. Download отдельно от EDA.
- Исправить CSV loader под минимальный контракт; сохранить raw data и порядок строк.
- Проверка: hash, archive parsing, CSV roundtrip; missing/invalid schema tests.
- Самопроверка: agreement subset — это train/test split? Почему encoding надо фиксировать?
- Свидетельство: Архив 681890 bytes загружен с pinned revision 8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964; SHA-256 подтверждён. CSV (3453, 2); roundtrip/hash/schema проверки проходят.

### D1-02 — структура и качество данных

- [x] Shape/head/info/dtypes, missing, blanks, duplicates и conflicting labels.
- Зависит от D1-01. Выход: quality summary и `reports/day01/quality.json`.
- Никакой очистки или скрытого удаления строк. Counts имеют явный denominator.
- Проверка: fixtures с NaN, пробелами, duplicates/conflicts; input не изменён.
- Самопроверка: почему отсутствие missing ещё не означает хорошие данные?
- Свидетельство: Реальный quality.json: missing=0, blanks=0, duplicate rows/texts=5, conflicting texts=0. Notebook проверяет неизменность df; edge cases покрыты offline tests.

### D1-03 — баланс классов

- [x] Counts, доли и график классов в фиксированном порядке.
- Зависит от D1-02. Выход: class CSV и sentiment_distribution.png.
- Missing labels учитываются отдельно; доля класса = count / общее число строк.
- Проверка: сумма counts + missing = N; отсутствующий класс имеет count=0.
- Самопроверка: почему accuracy может скрыть плохую работу на редком классе?
- Свидетельство: Реальные counts: negative=420 (12.16%), neutral=2146 (62.15%), positive=887 (25.69%). Сумма=3453; class CSV и PNG сохранены; fixed order/denominator tests проходят.

### D1-04 — длины и примеры

- [x] Символы/слова, describe, histogram и до трёх валидных примеров каждого класса.
- Зависит от D1-02. Выход: text_lengths.csv и text_length_distribution.png.
- Никаких regex cleaning или NLP tokenization; null не становится строкой `nan`.
- Проверка: известные длины fixture; пропуски сохранены; графики сохраняются без GUI.
- Самопроверка: чем слова по пробелам отличаются от tokens? Зачем исследовать хвост длин?
- Свидетельство: Median: 116 символов / 21 слов; max=81 слов. Сохранены text_lengths.csv, histogram PNG, по три примера каждого класса. Проверены missing lengths и сохранение PNG.

### D1-05 — выводы и clean-kernel checkpoint

- [x] Записать фактические наблюдения, ограничения и вопросы для day02.
- Зависит от D1-01…D1-04. Выход: observations.md и полностью выполненный notebook.
- Проверка: unit tests, compile, diff check, real-data run, notebook clean kernel.
- Сохранить версии runtime; повторный offline run не требует download или training.
- Самопроверка: что day01 позволяет заключить, а что требует обученной модели и test?
- Свидетельство: 12/12 unittest passed; compileall passed; git diff --check passed. Реальный notebook: 6/6 code cells executed, 0 errors, outputs сохранены; чистый offline kernel venv. Повторный запуск командой nbconvert из docs/day01.md успешен. Для kernel потребовался выход из sandbox из-за локальных sockets; Pyright gate добавлен отдельным последующим исправлением ниже; Ruff не запускался.

## Day02 — очистка текста и подготовка данных

Статусы пунктов указаны ниже. Один ID — один самостоятельный промпт Codex; выполнять
последовательно D2-01 → D2-06. Это детализация `classification_day02.txt` с учётом
SPEC §6, а не начало реализации. Day02 готовит тексты и метки; TF-IDF, split,
training и оценка моделей относятся к следующим дням.

Общий checkpoint: `notebooks/day02_preprocessing.ipynb`. После каждого пункта
добавить раздел «цель → теория → вызов кода → результат → интерпретация →
самопроверка», сохранить outputs и записать фактическое свидетельство в PLAN.
Проверки Python/code cells: `venv/bin/python scripts/check_types.py`, соответствующие
offline tests, compileall и `git diff --check`; реальный dataset и clean-kernel
execution проверять отдельно. Notebook должен работать из корня и `notebooks/`
на локальных данных offline. Ограничение kernel/sockets записывать явно;
пункт не отмечать завершённым без подтверждённого checkpoint.

### D2-01 — контракт минимальной очистки и notebook setup

- [x] Согласовать конкретную policy Day02 и создать начальный notebook checkpoint.
- Зависит от D1-05. Вход: SPEC §6, канонический CSV и шаблон `features/preprocess.py`.
- Зафиксировать в SPEC порядок преобразований: обработка HTML/entities,
  нормализация регистра и пробелов; явно определить, что сохраняется.
  Сохранить отрицания (`not`, `no`, `never`, `n't`), числа, проценты, валюты
  и финансовую пунктуацию; не вводить stopword removal, lemmatization или
  грубую замену всех uppercase words на тикеры без отдельной гипотезы.
- Определить контракт `preprocess_text(str) -> str`, поведение для пустой строки
  и неверного типа, ожидаемую идемпотентность. Пропуски обрабатываются на уровне
  таблицы, не через `str(value)`.
- Определить схему результата: исходные `text`, `sentiment`, стабильный
  `source_row_id` по позиции в canonical CSV, `text_clean`, будущий `label_id`.
  Зафиксировать причины исключения строк и отчётность; «сломанный пример»
  означает нарушение конкретного контракта, а не субъективно необычный текст.
- Выход: раздел Day02 в SPEC и notebook с root/src setup, загрузкой реальных
  данных и объяснением policy. Сам preprocessing пока не реализуется.
- Проверка: notebook загружает canonical CSV без изменения входа; root/import
  работают в обоих cwd; type gate и чистый kernel для созданных code cells.
- Самопроверка: как удаление `not` или `-5%` меняет смысл? Почему TF-IDF ещё нет?
- Промпт: «Выполни D2-01 из PLAN.md: зафиксируй контракт Day02 в SPEC и создай
  начальный day02 notebook. Не реализуй следующие пункты. Следуй общему checkpoint».
- Свидетельство (2026-10-11): SPEC §7 фиксирует `minimal-v1`, API, схему,
  исключения и границы этапов. Новый day02 notebook загрузил реальный CSV
  `(3453, 2)`; повторная загрузка совпала, SHA-256 до/после неизменен
  (`791b78a0be391241bd6ea600fdd0128caee7c0dbe89eed10dfd00de505fb3c2e`).
  Pyright sources + code cells всех notebooks: 0 errors/warnings; 12/12 offline
  unittest passed; compileall и diff check passed. Два чистых venv kernels
  через nbclient (cwd=root и cwd=notebooks): по 3/3 code cells, 0 errors;
  outputs сохранены. Sandbox запрещал sockets; разрешённый запуск вне sandbox
  подтвердил checkpoint. `preprocess.py` не изменён и не импортирован.

### D2-02 — повторно используемая очистка одного текста

- [x] Реализовать `preprocess_text` по policy D2-01 в `features/preprocess.py`.
- Зависит от D2-01. Вход: строка; выход: очищенная строка без обучаемого состояния.
- Устранить чтение NLTK resources при import и скрытые runtime/download зависимости
  этой функции; использовать минимальные необходимые средства. Проверить вызовы
  в исходных заготовках и не расширять реализацию training/inference.
- Проверка: offline tests на HTML/entities, mixed case, повторные пробелы,
  tabs/newlines, отрицания, числа/валюты/проценты, пустой текст и неверный тип;
  повторный вызов даёт тот же результат. Import не читает NLTK data и не скачивает их.
- Notebook: вручную сравнить raw → clean на нескольких реальных предложениях,
  объяснить каждое изменение; fixtures остаются в tests.
- Самопроверка: почему детерминированная очистка не требует `fit`?
- Промпт: «Выполни только D2-02 из PLAN.md: реализуй и проверь preprocess_text
  по SPEC. Обнови day02 notebook и свидетельство, выполни общий checkpoint».
- Свидетельство (2026-10-11): `preprocess_text` реализует `minimal-v1` без NLTK,
  с TypeError для non-str и fixed-point normalization. Девять новых offline tests
  проверяют financial signals, whitespace, HTML (включая quoted `>`), entities,
  unknown names, пустые строки, типы и идемпотентность; subprocess import/call
  проходит с запрещённым NLTK import и network. Полный suite: 21/21 passed;
  Pyright sources + все notebook code cells: 0 errors/warnings; compileall и
  diff check passed. Notebook в двух чистых venv kernels (cwd=root/notebooks):
  по 4/4 code cells, 0 errors; outputs сохранены. Реальные строки 393/0/1
  показывают raw → clean для negative/neutral/positive; `no plans`, число 6
  и пунктуация сохранены, регистр нормализован. DataFrame/CSV неизменны.
  Kernel запускался вне sandbox из-за ограничения sockets. Вызовы в train/predict
  проверены: они используют прежние helpers, которые остаются заготовками;
  training/inference и табличный preprocessing не изменены.

### D2-03 — обработка таблицы и отчёт об исключениях

- [ ] Реализовать preprocessing DataFrame с явной валидацией и отчётом.
- Зависит от D2-02. Вход: canonical DataFrame `(N, columns)`; выход: новая таблица
  с `source_row_id`, `text_clean` и отчёт о принятых/исключённых строках.
- Не менять исходный DataFrame, тексты, метки и относительный порядок строк.
  Не использовать `astype(str)` для маскировки missing. Missing text/label,
  whitespace-only и пустой результат очистки исключать с явной причиной;
  ошибочную схему, неизвестный непустой класс и нестроковый непустой текст
  диагностировать как ошибки контракта, не скрывать массовым удалением.
- Отчёт связывает причины со стабильными source IDs, фиксирует число строк
  до/после и однозначный учёт пересекающихся причин. Дубликаты пока не удалять.
- Проверка: fixtures на missing, blanks, текст, пустеющий после cleanup,
  неверную схему/типы/класс; неизменность входа, порядок и source IDs;
  `N_input = N_retained + N_excluded` без двойного счёта.
- Notebook: применить обработку к реальному dataset, показать shape, counts
  исключений и примеры изменений. Нулевые исключения тоже записать как результат.
- Самопроверка: почему отсутствие текста отличается от строки `"nan"`?
- Промпт: «Выполни только D2-03 из PLAN.md: добавь preprocessing таблицы и
  отчёт об исключениях. Обнови notebook и свидетельство, выполни общий checkpoint».
- Свидетельство: пока отсутствует.

### D2-04 — числовые метки и duplicate/conflict policy

- [ ] Добавить `label_id` и диагностику групп дубликатов до и после очистки.
- Зависит от D2-03. Mapping неизменен: `negative=0`, `neutral=1`, `positive=2`;
  строковый `sentiment` сохраняется. Unknown labels не получают fallback ID.
- В SPEC уточнить policy: повторы с одной меткой сохраняются и учитываются;
  conflicting labels диагностируются явно, без majority vote и молчаливого
  выбора первой строки. Финальная подготовка блокируется при неразрешённых conflicts.
- Проверять как одинаковые raw texts, так и новые совпадения `text_clean` после
  нормализации. Зафиксировать требование Day03: строки одной группы совпадений
  raw/clean не должны пересекать split; сами splits в Day02 не создавать.
- Выход: таблица с `label_id`, summary дубликатов/conflicts и source IDs групп.
- Проверка: fixed mapping независимо от порядка/наличия классов; одинаковые
  raw texts, совпадения после очистки, conflicts; воспроизводимый порядок групп.
- Notebook: показать mapping, counts и фактические совпадения до/после cleanup;
  объяснить отличие class ID от числовой величины и риск leakage через дубликаты.
- Самопроверка: означает ли `positive=2`, что он вдвое больше `neutral=1`?
- Промпт: «Выполни только D2-04 из PLAN.md: numeric labels и duplicate/conflict
  policy с диагностикой. Обнови SPEC, notebook и свидетельство; общий checkpoint».
- Свидетельство: пока отсутствует.

### D2-05 — сохранение processed dataset и provenance

- [ ] Добавить явную повторяемую команду подготовки и сохранение результатов.
- Зависит от D2-04. Вход: локальный canonical CSV и зафиксированная policy;
  выход: `data/processed/financial_phrasebank_75agree.csv` и `reports/day02/`.
- Команда вызывает общий preprocessing, не дублирует его и не скачивает данные.
  Сохранять UTF-8 CSV с согласованной схемой, отчёт об исключениях/дубликатах,
  mapping/class order, версию policy, источник/revision/checksum входа и фактически
  используемые версии библиотек. Raw и interim остаются неизменными.
- Проверка: CSV roundtrip (тексты, метки, IDs, порядок), согласованность отчётов,
  повторный offline запуск даёт ту же таблицу и counts; checksums входа не меняются.
  Артефакты остаются локальными согласно `.gitignore`, команда — в Git.
- Notebook: вызвать общую функцию подготовки и показать пути, схему и summary
  реальных сохранённых результатов; дать точную команду воспроизведения.
- Самопроверка: почему одного processed CSV недостаточно для воспроизводимости?
- Промпт: «Выполни только D2-05 из PLAN.md: явная offline подготовка, сохранение
  processed dataset и provenance. Обнови notebook и свидетельство; общий checkpoint».
- Свидетельство: пока отсутствует.

### D2-06 — ручная проверка и итоговый clean-kernel checkpoint

- [ ] Подтвердить критерии Day02 и записать фактические выводы/ограничения.
- Зависит от D2-01…D2-05. Не добавлять новые transformations или ML features.
- Notebook: собрать таблицу raw → clean → sentiment → label_id на нескольких
  реальных примерах каждого класса; объяснить сохранение финансового смысла,
  фактические изменения количества строк и ограничения минимальной policy.
- Выход: законченный `day02_preprocessing.ipynb` с outputs и
  `reports/day02/observations.md`, отдельный preprocessing step для будущих дней.
- Проверка: полный unittest suite, обязательный type gate, compileall,
  `git diff --check`, реальная offline подготовка и clean-kernel notebook
  сверху вниз из корня и `notebooks/`. Зафиксировать результаты отдельно.
- Приёмка: стабильная очистка, явные исключения, согласованные class IDs,
  reusable preprocessing, сохранённые артефакты и подтверждённый notebook;
  качество классификации и рост macro F1 на этом этапе не измеряются.
- Самопроверка: какие преобразования не обучаются, а какие в Day03 требуют train?
  Почему сохранение дубликатов требует группового split?
- Промпт: «Выполни только D2-06 из PLAN.md: ручная проверка, выводы и итоговая
  приёмка Day02. Сохрани outputs и свидетельства; не начинай Day03».
- Свидетельство: пока отсутствует.

## Roadmap day03–day07

| День | Подзадачи (все pending) | Результат / notebook |
| --- | --- | --- |
| 03 | D3-01 stable group-aware splits; D3-02 Dummy + TF-IDF/LogReg Pipeline; D3-03 validation metrics; D3-04 config/results/provenance | baseline reference, day03_baseline.ipynb |
| 04 | D4-01 hypotheses/protocol; D4-02 n-grams; D4-03 LinearSVC; D4-04 hyperparameter CV; D4-05 comparison/selection | ≥3 controlled experiments, day04_experiments.ipynb |
| 05 | D5-01 aligned predictions; D5-02 confusion matrix; D5-03 failure categories; D5-04 frozen final test/report | error analysis, day05_error_analysis.ipynb |
| 06 | D6-01 artifact bundle; D6-02 load/predict API; D6-03 roundtrip consistency; D6-04 new-text demos | inference without fit, day06_inference.ipynb |
| 07 | D7-01 end-to-end review; D7-02 runnable README; D7-03 summary/limitations; D7-04 clean-environment acceptance | reproducible mini-product, day07_review.ipynb |

## Шаблон запроса для следующего чата

> Выполни D2-03 из PLAN.md. Объясни цель, ML/NLP понятия и tradeoffs. Реализуй только
> этот шаг, обнови notebook и свидетельство в PLAN. Запусти нужные проверки,
> назови ограничения и предложи commit message; не делай commit.

При работе целым днём замените ID на day02. Крупные решения фиксируем в SPEC;
вопросы пользователя можно обсуждать, не меняя scope реализации.

## Дополнение day01 — Pylance/Pyright quality gate

- [x] Добавить pinned Pyright и pandas-stubs, standard config для venv/src/tests/scripts.
- [x] Исправить ROOT: Path через find_project_root; scalar row selection через iloc;
  word count без неподдерживаемого в stubs list-string accessor.
- [x] Добавить scripts/check_types.py с проверкой code cells всех notebooks и настройки VS Code.
- Проверки: Pyright sources и notebook export — 0 errors, 0 warnings, 0 informations; 12/12 unit tests; compileall и diff check прошли. Исправленный notebook повторно выполнен в чистом kernel, outputs обновлены.
