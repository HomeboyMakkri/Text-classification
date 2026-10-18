# PLAN — staged implementation

## Порядок работы

Day01 завершён 2026-10-09. Day02 (D2-01…D2-06) завершён 2026-10-11.
Day03: D3-01…D3-07 завершены 2026-10-14.
Day04: D4-01…D4-08 завершены 2026-10-18.
Day05: D5-01…D5-06 завершены 2026-10-18; приёмка подтверждена.
Day06: D6-01/D6-02 выполнены 2026-10-18; D6-03…D6-08 pending по SPEC §11.
Следующий исполнимый пункт — D6-03.
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

- [x] Реализовать preprocessing DataFrame с явной валидацией и отчётом.
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
- Свидетельство (2026-10-11): `preprocess_news` возвращает новую таблицу из
  четырёх колонок и typed `PreprocessingReport`; source IDs — позиции до
  фильтрации. Ошибки schema/types/classes явные; missing/blank/empty-clean
  исключаются с source IDs и всеми причинами, без двойного счёта строк.
  Полностью исключённый dataset вызывает `EmptyProcessedDatasetError` с report.
  `clean_news_data` делегирует тому же API и предупреждает об исключениях.
  Пять новых table tests; полный offline suite 26/26 passed. Итоговый Pyright
  sources + notebook code cells: 0 errors/warnings; compileall/diff check passed.
  Реальный CSV: `(3453, 2)` → `(3453, 4)`, retained=3453, excluded=0,
  все reason counts=0, changed retained texts=3447; class counts неизменны.
  Исходный DataFrame и SHA-256 CSV неизменны. Два clean venv kernels через
  nbclient (cwd=root/notebooks): по 5/5 code cells, 0 errors; outputs сохранены.
  Для kernels потребовался разрешённый выход из sandbox из-за sockets.
  Numeric labels, duplicate diagnostics и запись processed artifacts не добавлены.

### D2-04 — числовые метки и duplicate/conflict policy

- [x] Добавить `label_id` и диагностику групп дубликатов до и после очистки.
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
- Свидетельство (2026-10-11): `features/labels_duplicates.py` добавляет fixed
  int64 mapping и typed raw/clean groups с source IDs, метками и conflict/new
  flags. `prepare_labeled_news` сохраняет repeats и блокирует conflicts через
  exception с `.report`; SPEC фиксирует policy и group-aware boundary Day03.
  Новые pytest cases: 20 passed; полный suite 46 passed, 46 subtests passed;
  старые unittest tests не изменены. Pyright sources/notebook exports:
  0 errors/warnings; compileall и diff check passed. Реальные данные:
  `(3453, 4)` → `(3453, 5)`, raw/clean по 5 duplicate groups и 5 repeats
  сверх первого, new clean groups=0, raw/clean conflicts=0; все строки сохранены.
  Два чистых venv kernels (cwd=root/notebooks): по 6/6 code cells, 0 errors;
  outputs сохранены, исходные DataFrame/CSV неизменны. Kernels потребовали
  разрешённый запуск вне sandbox из-за sockets. Splits и запись artifacts
  не добавлены; дубликаты требуют группового split в Day03.

### D2-05 — сохранение processed dataset и provenance

- [x] Добавить явную повторяемую команду подготовки и сохранение результатов.
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
- Свидетельство (2026-10-11): offline `data.prepare_day02` сверяет локальный архив,
  canonical CSV и class/source contract до записи, вызывает общий preprocessing
  и блокирующую conflicts финализацию. Сохранены processed CSV `(3453, 5)`
  (929290 bytes) и preprocessing/duplicates/provenance JSON с source/runtime/policy,
  mapping и checksums; excluded=0, raw/clean repeats=5/5, conflicts=0.
  CSV SHA-256: `39dda898a281dc3294be081917b0681524e73edb60f00913183c67714da0c448`.
  Семь новых pytest cases; полный suite 53 passed, 46 subtests passed.
  Pyright sources + notebook exports: 0 errors/warnings; compileall/diff check
  passed. Реальная CLI выполнена offline; CSV roundtrip и одинаковые bytes всех
  outputs при повторном запуске подтверждены. Raw/interim checksums неизменны.
  Два clean venv kernels (cwd=root/notebooks): по 7/7 code cells, 0 errors,
  outputs сохранены; артефакты совпали между cwd. Kernels вне sandbox из-за sockets.
  `.gitignore`: `/data/` исключает корневые артефакты, не новые src/data modules;
  все четыре артефакта ignored. Команды и ограничения описаны в docs/day02.md.

### D2-06 — ручная проверка и итоговый clean-kernel checkpoint

- [x] Подтвердить критерии Day02 и записать фактические выводы/ограничения.
- Зависит от D2-01…D2-05. Не добавлять новые transformations или ML features.
- Notebook: собрать таблицу raw → clean → sentiment → label_id на нескольких
  реальных примерах каждого класса; объяснить сохранение финансового смысла,
  фактические изменения количества строк и ограничения минимальной policy.
- Выход: законченный `day02_preprocessing.ipynb` с outputs и
  `reports/day02/observations.md`, отдельный preprocessing step для будущих дней.
- Проверка: полный suite через pytest, обязательный type gate, compileall,
  `git diff --check`, реальная offline подготовка и clean-kernel notebook
  сверху вниз из корня и `notebooks/`. Зафиксировать результаты отдельно.
- Приёмка: стабильная очистка, явные исключения, согласованные class IDs,
  reusable preprocessing, сохранённые артефакты и подтверждённый notebook;
  качество классификации и рост macro F1 на этом этапе не измеряются.
- Самопроверка: какие преобразования не обучаются, а какие в Day03 требуют train?
  Почему сохранение дубликатов требует группового split?
- Промпт: «Выполни только D2-06 из PLAN.md: ручная проверка, выводы и итоговая
  приёмка Day02. Сохрани outputs и свидетельства; не начинай Day03».
- Свидетельство (2026-10-11): notebook дополнен полной raw/clean таблицей для
  трёх реальных примеров каждого класса (IDs 393/396/397, 0/48/49, 1/2/3),
  ручной интерпретацией сохранённых negation/чисел/валют/процентов/пунктуации
  и самопроверкой. `reports/day02/observations.md` пересоздаётся notebook из
  фактических reports; ограниченность ручной выборки и отсутствие ML metrics явны.
  Реальная offline CLI прошла; итог `(3453, 5)`, excluded=0, changed=3447,
  class counts 420/2146/887 неизменны, raw/clean repeats=5/5, new/conflicts=0.
  Notebook проверил идемпотентность и согласованность cleanup всех 3453 retained
  texts, fixed mapping и неизменность источника. Offline pytest: 53 passed,
  46 subtests passed; Pyright sources + notebook exports: 0 errors/warnings;
  compileall и diff check passed. Два чистых venv kernels (cwd=root/notebooks):
  по 9/9 code cells, 0 errors; outputs сохранены. Все пять локальных артефактов
  совпали по bytes между cwd. Kernels запускались с разрешённым выходом из
  sandbox из-за sockets. Day02 критерии подтверждены.

## Day03 — baseline классификации

Статусы и свидетельства указаны ниже. Один ID — один самостоятельный промпт
агенту; выполнять последовательно D3-01 → D3-07. Это реализация
`classification_day03.txt` по SPEC §8; D3-01…D3-07 завершены.
Вход дня — проверенный processed dataset Day02 `(N, 5)`; выход — обученный
TF-IDF/LogisticRegression Pipeline, ориентир DummyClassifier и сохранённая
baseline reference на validation. Held-out test выделяется, но не оценивается
до D5-04: это согласованное в SPEC §2 отличие от формулировки задания.

Общий checkpoint: `notebooks/day03_baseline.ipynb`. Каждый пункт добавляет
свой раздел «цель → теория → вызов кода → результат → интерпретация →
самопроверка» с outputs на реальных локальных данных. Логика — в
`src/finnews_sentiment/`, новые tests — pytest. После изменений Python/code cells
обязательны `venv/bin/python scripts/check_types.py`, соответствующие offline
tests, compileall и `git diff --check`; проверки реального dataset и execution
notebook в чистых kernels из корня и `notebooks/` записываются отдельно.
Без подтверждённого notebook checkpoint и нулевых type errors ID не закрывать.
В свидетельствах указывать фактические результаты, а недоступные проверки — явно.

### D3-01 — ML runtime, проверка входа и notebook setup

- [x] Подготовить минимальное окружение Day03 и проверенный вход обучения.
- Зависит от D2-06. Вход: processed CSV, `reports/day02/provenance.json`,
  SPEC §8 и стартовый `configs/train_baseline.json`.
- Проверить доступность scikit-learn в текущем `venv/`; добавить только нужные
  Day03 зависимости с проверенными pins в `requirements-day03.txt`, сохранив
  runtime Day01. Не устанавливать весь будущий stack из `requirements.txt`.
- Исправить baseline config по SPEC: реальный input path, `text_clean`,
  `label_id`, seed, доли splits, параметры TF-IDF/LogReg и Dummy strategy.
  Старые `clean_news.csv`, `text`, `sentiment`, `model_path` не определяют
  новый training workflow; JSON читается явно, без config framework.
- Реализовать typed loader/validation processed dataset: пять колонок §7,
  IDs/метки/порядок, непустые тексты, fixed mapping, policy и checksums
  provenance. Ошибки/пропуски/conflicts не исправлять через dropna/astype(str).
  При отсутствии артефактов дать точную отдельную команду подготовки Day02.
- Выход: loader, согласованный config, минимальный runtime и notebook с
  root/src setup, загрузкой реального `(N, 5)` и объяснением потока данных.
  Split, vectorizer fit и training на этом пункте не выполнять.
- Проверка: missing files, неверные schema/types/mapping/IDs, пустые тексты,
  checksum/policy mismatch, conflicts; roundtrip буквальных `NA`/`nan` и
  numeric-only texts; импорт без network/training, неизменность входа.
- Самопроверка: почему числовой label — категория? Почему text_clean уже готов,
  а TF-IDF ещё не обучен? Зачем проверять provenance вместе с CSV?
- Промпт: «Выполни только D3-01 из PLAN.md по SPEC §8: ML runtime, config,
  проверенный loader и начальный day03 notebook. Не начинай split/training.
  Выполни общий checkpoint и запиши свидетельство».
- Свидетельство (2026-10-14): существующий venv уже содержал scikit-learn 1.9.1,
  SciPy 1.18.1, joblib 1.6.0 и threadpoolctl 3.7.0; imports и pip check прошли,
  переустановка/обновление не требовались. Версии закреплены в
  requirements-day03.txt с сохранением Day01 pins. Config исправлен на реальные
  paths, text_clean/label_id, seed/fractions и явные estimator settings; L2
  в закреплённом API задан через l1_ratio=0.0 без deprecated penalty.
  Typed BaselineConfig и read-only loader/validator проверяют provenance/hash,
  CSV schema/records/IDs/mapping/classes, raw→clean consistency и conflicts.
  73 новых offline pytest cases passed; полный suite: 126 passed,
  46 subtests passed. Pyright sources + exports всех трёх notebooks:
  0 errors/warnings; compileall и diff check passed. Реальный processed CSV
  загрузился `(3453, 5)`, counts negative/neutral/positive=420/2146/887,
  raw/clean repeats=5/5, conflicts=False; checksum совпал с provenance
  (`39dda898a281dc3294be081917b0681524e73edb60f00913183c67714da0c448`).
  Повторная загрузка сохранила значения/dtypes/порядок; checksums шести
  доступных inputs, включая raw/interim, неизменны. Новый day03 notebook
  исполнен в двух чистых kernels `text-classification` с venv interpreter
  (cwd=root/notebooks): по 4/4 code cells, 0 errors, outputs сохранены.
  Loader не перечитывает raw/interim archive содержимое; связь с источником
  подтверждается проверенным Day02 provenance. Split, estimators, fit,
  predictions и metrics не реализованы; source/model templates не изменены.

### D3-02 — воспроизводимый stratified group-aware split

- [x] Разделить группы текстов на train, validation и held-out test.
- Зависит от D3-01. Вход: проверенная таблица; выход: manifest
  `source_row_id, group_id, split` и summary в памяти, таблицы трёх splits.
- Реализовать split в `data/split_data.py`: группы по точному `text_clean`,
  стабильный group ID = минимальный source ID, группы в порядке этого ID.
  Проверить отсутствие raw/clean conflicts и согласованность raw → clean.
- Stratify уникальные группы по их единственной метке: сначала test=20%,
  затем validation=20% от оставшихся групп, seed=42 на обоих этапах.
  Сохранить все исходные строки и их порядок внутри каждого split.
- Показать фактические counts/доли строк и групп, counts/доли классов во всех
  splits. Пропорции ≈64/16/20 относятся к группам, не к гарантированным row counts.
  Недостаточное число групп/классов — информативная ошибка без fallback.
- Проверка: покрытие всех source IDs ровно один раз, попарная непересекаемость
  IDs/raw/clean texts, три класса в каждом split, одинаковый manifest при
  повторе, сохранение same-label duplicates, ошибки conflicts/малых классов.
- Notebook: таблица размеров/баланса, проверки disjointness, объяснение
  train/validation/test и причины group split. Метрики моделей ещё не считать.
- Самопроверка: почему обычный stratified row split допускает leakage?
  Чем размер split по группам отличается от размера по строкам?
- Промпт: «Выполни только D3-02 из PLAN.md: stable stratified group-aware
  train/validation/test split по SPEC §8. Обнови day03 notebook, выполни
  общий checkpoint; не обучай модели и не оценивай test».
- Свидетельство (2026-10-14): `data/split_data.py` реализует
  `group-stratified-v1`, seed=42 на обоих этапах и сортировку outer-train
  перед validation; группы по точному text_clean, ID=min source_row_id.
  `NewsSplits` возвращает manifest, три таблицы и typed row/group summary;
  reuse validator блокирует raw→clean inconsistency и raw/clean conflicts.
  Все строки, включая same-label repeats, и их порядок/index сохраняются.
  21 новый pytest case проверяет coverage/disjointness, stable IDs, баланс,
  повторяемость/index independence, сохранение repeats, seed42 reference,
  conflicts, малые классы/rounding и неверные fractions/seed без fallback.
  Полный offline suite: 147 passed, 46 subtests passed; Pyright sources и
  exports всех notebooks: 0 errors/warnings; compileall/diff check passed.
  На реальных данных: rows=3453, groups=3448; train/validation/test rows
  2210/553/690, groups 2206/552/690; class row counts negative/neutral/positive:
  train=269/1374/567, validation=67/344/142, test=84/428/178.
  Все три класса присутствуют, IDs/raw/clean texts попарно disjoint,
  manifest повторяется, все 5 repeats сохранены; checksums inputs неизменны.
  Предыдущие условия перепроверены: исходный suite 126+46 прошёл, pip check
  без конфликтов; Day02 archive/canonical/processed hashes и policy совпали,
  loader/config D3-01 валидны; Day02 clean venv kernel — 9/9 cells без ошибок.
  Day03 clean venv kernels (cwd=root/notebooks): по 7/7 code cells, 0 errors;
  outputs новых D3-02 ячеек сохранены, пользовательские outputs D3-01 оставлены.
  Runtime Python 3.14.7, Day03 pins сохранены; полный curriculum stack ранее
  установлен по явному запросу пользователя, PyTorch — CPU.
  Первую Jupyter попытку ограничил read-only ~/.jupyter; после переключения
  пользователем на Full access kernelspec и clean kernels успешно проверены.
  Split/summary пока только в памяти; estimators, fit и test metrics отсутствуют.

### D3-03 — конструкция TF-IDF/LogReg Pipeline и Dummy baseline

- [x] Создать необученные estimators с явно зафиксированными параметрами.
- Зависит от D3-02. Вход: согласованный config; выход: новый unfitted
  `Pipeline([('tfidf', ...), ('clf', ...)])` и DummyClassifier `most_frequent`.
- Доработать `models/train_model.py` в пределах создания estimators:
  TF-IDF max_features=5000, ngram_range=(1, 2), без stopword removal;
  LogisticRegression по SPEC §8. Не использовать add_basic_features,
  повторный preprocessing, TF-IDF fit всей таблицы или dense conversion.
- Каждый вызов builder возвращает независимый estimator. Dummy принимает
  простой constant feature array; общий интерфейс обучения появится в D3-04.
- Проверить, что новые modules не скрыты правилом `.gitignore` `models/`;
  при необходимости сузить его до `/models/` до проверки и добавления кода.
  Старый training CLI пока не считать поддерживаемым; новый появится в D3-06.
- Проверка: параметры/шаги Pipeline, отсутствие fitted attributes до fit,
  независимость экземпляров и отсутствие training при import/build.
- Notebook: показать config/estimators, объяснить unigram/bigram, vocabulary,
  IDF и будущую sparse форму `(N_train, V)`, где `V <= 5000`; fit пока нет.
  Разобрать ограничение стандартного tokenizer для валют/процентов/`n't`.
- Самопроверка: почему Pipeline защищает границу fit/transform? Что теряется
  при bag-of-n-grams? Почему сохранённая preprocessing пунктуация может не
  попасть в vocabulary стандартного TF-IDF tokenizer?
- Промпт: «Выполни только D3-03 из PLAN.md: builders TF-IDF/LogReg Pipeline
  и DummyClassifier по SPEC §8. Обнови notebook и проверки; пока без fit».
- Свидетельство (2026-10-14): `models/train_model.py` строит Pipeline
  `tfidf → clf` и DummyClassifier по typed config; оба builder создают новый
  unfitted объект и не читают данные/не вызывают fit. В `.gitignore` правило
  `models/` сужено до `/models/`, чтобы пакет `src/finnews_sentiment/models/`
  отслеживался, а локальные корневые model artifacts оставались игнорируемыми.
  Pytest проверяет все заданные параметры, типы/шаги, отсутствие fitted
  attributes и независимость экземпляров. Новый раздел notebook объясняет
  sparse `(N_train, V)`, unigram/bigram, tokenizer limits и Dummy input;
  clean kernel успешно исполнил notebook сверху вниз из корня и `notebooks/`,
  outputs сохранены. Полный offline suite: 148 passed, 46 subtests passed;
  Pyright sources и notebook exports: 0 errors/warnings; compileall и
  `git diff --check` прошли. После получения Pylance diagnostics установлен
  `scikit-learn-stubs==0.0.3` в `venv/` и закреплён в dev requirements;
  Pyright теперь видит sklearn Literal contracts, reported type mismatches
  устранены, повторная общая проверка остаётся чистой. Fit/predict не выполнялись.

### D3-04 — обучение на train и aligned validation predictions

- [x] Обучить оба estimators только на train и предсказать validation.
- Зависит от D3-03. Вход: таблицы D3-02 и estimators; выход: fitted Pipeline,
  fitted Dummy, validation predictions `(N_valid,)` с source IDs.
- Training API принимает готовые splits и не делит данные повторно.
  `fit(train.text_clean, train.label_id)` для Pipeline; Dummy fit по train
  labels. Validation проходит только predict/transform, held-out test —
  только структурные проверки manifest, без transform/predict/score.
- Отчёт: фактическая sparse форма `(N_train, V)`, число признаков,
  classes_ и статус сходимости. ConvergenceWarning делает checkpoint
  неподтверждённым до устранения/документированного исправления config.
  Не подбирать параметры по validation и не обещать качество заранее.
- Проверка: validation-only token отсутствует в vocabulary; vocabulary/IDF
  не меняются после predict; predictions имеют валидные IDs классов,
  длину/порядок source IDs; повторное обучение воспроизводимо в том же runtime;
  неверные/пересекающиеся splits диагностируются до fit.
- Notebook: реальные fit/predict, shapes и alignment; без списка ошибок,
  подбора гиперпараметров, test metrics или сохранения inference bundle.
- Самопроверка: что обучается в TF-IDF и LogReg? Чем transform отличается
  от fit_transform? Почему vocab/IDF нельзя дообучить на validation?
- Промпт: «Выполни только D3-04 из PLAN.md: обучение двух baseline estimators
  на train и aligned validation predictions. Обнови notebook/свидетельство,
  выполни общий checkpoint; не трогай test и не начинай tuning».
- Свидетельство (2026-10-14): `models.train_model.fit_baselines` принимает
  готовый `NewsSplits`, валидирует manifest, схему, classes, group IDs и
  попарную disjointness до создания estimators; повторного split нет. Pipeline
  обучен на `train.text_clean/train.label_id`, Dummy — на train labels с
  constant `(N_train, 1)` features. Test проверен структурно, в fit/predict/
  transform не передавался. Возвращается aligned validation DataFrame с
  `source_row_id` и двумя prediction columns; feature shape `(2210, 5000)`,
  `classes_=[0,1,2]`, convergence `converged_no_warning`. Реальный checkpoint:
  N_train=2210, N_valid=553; 5 706 validation-only terms отсутствуют в train
  vocabulary, а повторный predict оставил vocabulary/IDF неизменными. Повторный
  fit дал те же predictions. Unit tests проверяют train-only vocabulary,
  sparse matrix, alignment, воспроизводимость, неизменность TF-IDF state и
  malformed manifest до создания estimators. Notebook выполнен clean kernel
  сверху вниз из корня и `notebooks/`, outputs сохранены, метрики не считались.
  Полный suite: 150 passed, 46 subtests passed; Pyright source/notebook exports:
  0 errors/warnings; compileall, `git diff --check`, `pip check` прошли.

### D3-05 — validation metrics и сравнение с Dummy

- [x] Оценить baseline и Dummy на одних и тех же validation rows.
- Зависит от D3-04. Вход: aligned truth/predictions; выход: typed metrics
  и таблица сравнения с фиксированным class order.
- Реализовать reusable evaluation в `models/evaluate.py`: macro F1,
  accuracy и precision/recall/F1/support для negative/neutral/positive.
  Использовать labels=[0, 1, 2] и zero_division=0 явно; denominator=N_valid.
- Проверка: известные true/pred arrays и независимый ручной расчёт metrics,
  неизвестные IDs, пустые/разной длины массивы, отсутствие predicted класса;
  support sums=N_valid, одинаковые validation IDs обеих моделей.
- Notebook: фактические метрики без test, сравнение macro F1/accuracy,
  влияние neutral majority и ограничения одного split. Плохое качество тоже
  валидный baseline; не менять параметры по результатам этого пункта.
  Confusion matrix и разбор ошибочных предложений остаются Day05.
- Самопроверка: почему macro F1 может быть низким при высокой accuracy?
  Что означает нулевой recall класса? Какие выводы одного split ненадёжны?
- Промпт: «Выполни только D3-05 из PLAN.md: validation metrics baseline
  и Dummy, fixed class order и учебная интерпретация. Обнови notebook,
  выполни общий checkpoint; без tuning и test evaluation».
- Свидетельство (2026-10-14): `models/evaluate.py` предоставляет typed
  `ClassificationMetrics`/`ClassMetrics` и `evaluate_baselines`; macro F1,
  accuracy и per-class precision/recall/F1/support считаются с явными
  labels `[0,1,2]`, `average=None` для per-class и `zero_division=0`.
  Валидация блокирует пустые/не одномерные массивы, bool/float/unknown IDs и
  разные длины; support всех трёх классов суммируется в `N_valid`, в том числе
  когда класс ни разу не предсказан. Unit tests сверяют известный пример с
  ручным расчётом и проверяют comparison table. Notebook executed clean kernel
  из корня и `notebooks/`; оба estimator использовали те же 553 validation IDs,
  test не оценивался. Реальные validation results: baseline accuracy=0.813743,
  macro F1=0.720965; Dummy accuracy=0.622061, macro F1=0.255667. У baseline
  recall negative/neutral/positive = 0.462687/0.976744/0.584507; у Dummy
  majority neutral recall=1.0, recall двух других классов=0. Метрики сохранены
  только как notebook outputs; tuning и confusion matrix не выполнялись.
  Полный suite: 162 passed, 46 subtests passed; Pyright source/notebook exports:
  0 errors/warnings; compileall, `git diff --check`, `pip check` прошли.

### D3-06 — сохранение baseline reference и offline команда

- [x] Сохранить splits, predictions, metrics, config и provenance для Day04.
- Зависит от D3-05. Вход: проверенные результаты D3-01…D3-05;
  выход: артефакты SPEC §8 и явная offline команда воспроизведения.
- Добавить единый runner, вызывающий общий loader/split/build/train/evaluate,
  без дублирования логики в CLI/notebook. Явная команда из корня:
  `PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.train_model --config configs/train_baseline.json`.
- Сохранить manifest/summary, aligned validation CSV двух моделей, metrics,
  effective config и provenance: source/policy/hash, seed/split algorithm,
  параметры, class order/mapping и фактические runtime versions.
  Метрики обязаны называться validation; test predictions/metrics отсутствуют.
- Проверка: CSV/JSON roundtrip, повторный offline запуск даёт тот же split,
  predictions и metrics в том же runtime; metrics пересчитываются из CSV,
  все source IDs согласованы; raw/interim/processed checksums неизменны.
- Notebook: вызвать runner, показать реальные пути и сверить сохранённые
  результаты с D3-02…D3-05. Артефакты локальные, код/config/инструкция — в Git;
  новые modules должны присутствовать в diff/status и проверке типов.
  Joblib inference bundle/load/predict API остаются Day06.
- Самопроверка: почему одного macro F1 недостаточно для сравнения экспериментов?
  Почему одинаковые seed при изменённом dataset не гарантируют тот же split?
- Промпт: «Выполни только D3-06 из PLAN.md: baseline reference artifacts,
  provenance и offline runner/CLI по SPEC §8. Обнови notebook и свидетельство,
  выполни общий checkpoint; без inference bundle и будущих experiments».
- Свидетельство (2026-10-14): добавлен `models/run_baseline.py` и поддерживаемый
  CLI `PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.train_model
  --config configs/train_baseline.json`. Runner использует D3-01…D3-05 reusable
  APIs; повторного preprocessing/split logic в notebook нет. Сохранены локально:
  `data/processed/day03/splits.csv` (manifest), `reports/day03/split_summary.json`,
  `validation_predictions.csv`, `metrics.json`, effective `config.json` и
  `provenance.json`. Provenance содержит source/revision/subset, Day02 policy и
  provenance checksum, checksums исходных input bytes и config/processed CSV/
  manifest/summary/predictions/metrics, seed/algorithm, V=5000, classes и runtime
  versions; timestamp отсутствует. CSV/JSON roundtrip проверяется до записи;
  metrics повторно пересчитываются из round-tripped validation CSV; входные
  bytes сверяются повторно перед outputs. Runner tests покрывают все outputs,
  schemas/ID order, provenance checksums, deterministic repeat и missing input.
  Реальная CLI команда и notebook runner прошли; все шесть artifacts побайтно
  совпали при повторном запуске в текущем runtime. Notebook clean kernel прошёл
  из корня и `notebooks/`, outputs сохранены; test metrics/predictions и model
  bundle отсутствуют. Полный suite: 165 passed, 46 subtests passed; Pyright
  sources/notebook exports: 0 errors/warnings; compileall, `git diff --check`,
  `pip check` прошли. Reference artifacts остаются локальными/ignored.

### D3-07 — выводы и итоговая приёмка Day03

- [x] Подтвердить baseline как воспроизводимую точку отсчёта для Day04.
- Зависит от D3-01…D3-06. Новые модели/features/подбор не добавлять.
- Выход: полный `day03_baseline.ipynb` с outputs,
  `reports/day03/observations.md` и инструкция `docs/day03.md`.
- Выводы: фактические размеры/баланс splits, параметры и vocabulary size,
  baseline/Dummy validation metrics, ограничения TF-IDF, repeats и одного
  split; held-out test ещё не использован, рост качества не гарантируется.
- Проверка: полный pytest suite, type gate, compileall, diff check;
  отдельно реальная offline CLI/согласованность артефактов и notebook сверху
  вниз в чистых kernels из корня и `notebooks/`, без download.
- Приёмка: рабочий Pipeline обучается и предсказывает, group leakage нет,
  validation metrics и baseline reference сохранены и воспроизводимы;
  все пункты имеют проверенный notebook checkpoint и фактическое свидетельство.
- Самопроверка: на каких данных Day04 вправе выбирать модель? Почему оценка
  test сейчас меняет смысл будущей финальной оценки? Что baseline не доказывает
  о работе на новых финансовых новостях?
- Промпт: «Выполни только D3-07 из PLAN.md: выводы, документация и финальный
  checkpoint Day03. Запиши реальные проверки; не начинай Day04 и не оценивай test».
- Свидетельство (2026-10-14): `notebooks/day03_baseline.ipynb` завершён разделом
  D3-07; clean-kernel execution из корня и `notebooks/` сохранён. Итоговая
  notebook-проверка подтвердила 3453 строки/3448 групп, disjoint train/validation/
  test по source ID/raw/clean text, manifest и predictions round-trip, metric
  checksum, effective config и provenance. Splits: train 2210 строк/2206 групп,
  validation 553/552, test 690/690; строковый баланс отрицательный/нейтральный/
  положительный: train 269/1374/567, validation 67/344/142, test 84/428/178.
  Train TF-IDF sparse shape — (2210, 5000); baseline validation accuracy/macro F1
  0.813743/0.720965, Dummy — 0.622061/0.255667. Test только структурно включён
  в manifest; test predictions, metrics, fit и tuning отсутствуют. Подготовлены
  `reports/day03/observations.md` и `docs/day03.md` с интерпретацией, границами
  применения и командами воспроизведения.
  Проверки: offline CLI repeat дал побайтно идентичные шесть outputs;
  `venv/bin/python -m pytest tests -v` — 165 passed, 46 subtests passed;
  `venv/bin/python scripts/check_types.py` — 0 errors/warnings;
  `venv/bin/python -m compileall -q src tests`, `venv/bin/python -m pip check`,
  `git diff --check` прошли. Notebook исполнялся чистым kernel сверху вниз из
  корня и `notebooks/`, без download. Полный протокол и ограничения — в
  `reports/day03/observations.md`.

## Day04 — контролируемые эксперименты и выбор модели

Детализация `classification_day04.txt` по SPEC §2, §6 и §8. Один ID — один
самостоятельный промпт Codex; выполнять последовательно D4-01 → D4-08.
Завершены D4-01…D4-07: протокол зафиксирован, reference проверена, общие CV
folds построены, все кандидаты сравнены и CV-победитель подтверждён на validation.
Вход — проверенные Day02 artifacts и сохранённая Day03 baseline reference,
включая неизменный manifest train/validation/test. Выход — сравнение трёх
направлений улучшения, выбранная конфигурация и воспроизводимый протокол.
Прирост не гарантируется: baseline может остаться лучшим вариантом.

Протокол: одинаковые group-aware CV folds только внутри Day03 train; каждый
fold заново fit всего Pipeline, включая TF-IDF. CV выбирает конфигурацию;
после фиксации выбора она fit на полном train и оценивается на прежнем
validation для сравнения с Day03. Validation не добавляется в обучение.
Held-out test не участвует в fit/transform/predict/score; его оценка — D5-04.
Ручной разбор ошибок и confusion matrix — Day05, inference bundle — Day06.

Общий checkpoint: `notebooks/day04_experiments.ipynb`. После каждого ID добавить
раздел «цель → теория → вызов кода → результат → интерпретация → самопроверка»,
сохранить реальные outputs и свидетельство в PLAN. Логика — в
`src/finnews_sentiment/`, новые tests — pytest; notebook — тонкий caller.
После изменений Python/code cells обязательны type gate
`venv/bin/python scripts/check_types.py`, соответствующие offline tests,
compileall и `git diff --check`. Реальные данные и clean-kernel execution
из корня и `notebooks/` проверять отдельно, offline, без download.
Пункт закрывать только с подтверждённым checkpoint и без type errors;
недоступные проверки записывать явно. Документные правки не требуют tests.

### D4-01 — гипотезы, протокол сравнения и notebook setup

- [x] Зафиксировать до запуска экспериментов контракт Day04 в новом разделе SPEC
  и создать начальный notebook с загрузкой реальных входных артефактов.
- Зависит от D3-07. Проверить наличие Day03 reference, config и provenance;
  при отсутствии дать команду воспроизведения Day03, не подменять fixtures.
- Три направления: word unigrams вместо baseline word (1, 2);
  LinearSVC вместо LogisticRegression с прежним TF-IDF; подбор LogReg C
  при прежнем baseline TF-IDF. Это независимые проверки, без смешивания
  изменений и без автоматического добавления комбинаций/новых моделей.
- Зафиксировать CV: StratifiedGroupKFold(n_splits=3, shuffle=True,
  random_state=42) внутри train, groups из Day03 manifest. Для всех кандидатов
  одни и те же folds; три класса и непересечение групп обязательны.
- Критерий выбора: максимальное среднее fold macro F1, labels=[0, 1, 2],
  zero_division=0; дополнительно fold scores, std и accuracy. При точном
  равенстве предпочесть baseline, далее заранее заданный порядок candidate IDs.
  Baseline участвует в выборе наравне с кандидатами. Std не считать
  доверительным интервалом или доказательством статистической значимости.
- Выход: SPEC-контракт с ограниченным бюджетом экспериментов, выбором по CV,
  правилами validation confirmation, ошибок/сходимости и схемой артефактов;
  notebook setup без training. Параметры LinearSVC определить по закреплённому
  sklearn API, сохранив C=1, class_weight=None, seed=42 и явный max_iter.
- Проверка: документы согласованы с Day03/SPEC; notebook читает локальные inputs
  и показывает baseline reference; исходные файлы/outputs Day03 не изменены.
- Самопроверка: что именно проверяет каждая гипотеза? Почему CV и validation
  имеют разные роли? Почему лучший вариант может совпасть с baseline?
- Промпт: «Выполни только D4-01 из PLAN.md: контракт Day04 в SPEC и начальный
  notebook. Не запускай эксперименты; выполни общий checkpoint».
- Свидетельство (2026-10-16): `SPEC.md` §9 фиксирует три независимых направления,
  общие StratifiedGroupKFold folds на train, метрики/selection/tie rules,
  validation confirmation, test boundary, LinearSVC 1.9.1 параметры, обработку
  ошибок/convergence warnings и схему Day04 artifacts. Создан
  `notebooks/day04_experiments.ipynb`: чистый kernel загрузил реальные 3453×5
  Day02 данные, сверил manifest (3453×3), пять Day03 checksums, схему и порядок
  553 validation predictions, показал сохранённые baseline/Dummy metrics; fit и
  эксперименты не выполнялись. Notebook успешно исполнен сверху вниз из корня
  и `notebooks/`; split sizes 2210/553/690, sklearn 1.9.1. Day02/Day03 outputs
  не изменены. Type gate: 0 errors/warnings после точечной typing-совместимости
  `zero_division=0` с установленными sklearn stubs; compileall и diff check
  прошли; `venv/bin/python -m pytest tests/test_day03_evaluate.py -v` — 12 passed.
  Эксперименты не запускались.

### D4-02 — загрузка фиксированных splits и общие CV folds

- [x] Реализовать проверенную загрузку Day03 manifest и построение train-only folds.
- Зависит от D4-01. Переиспользовать Day03 loader/validation; splits восстановить
  по сохранённым source IDs, не вызывать новый train_test_split.
- Проверить checksums/provenance, coverage/order/labels/group IDs и disjointness;
  отсутствующий/несогласованный manifest — ошибка с инструкцией восстановления.
  Не переписывать reference и не менять seed в поисках удобного разбиения.
- Создать фиксированный fold manifest по train source IDs/group IDs;
  каждый train row ровно один раз в CV validation, duplicate group целиком
  в одном fold. Ни validation, ни test source IDs в folds не входят.
- Выход: reusable input/fold helpers, fold manifest и summary в памяти.
  При невозможной stratification — информативная ошибка без fallback.
- Проверка: повреждённый manifest/hash, неверные IDs/groups, неполное покрытие,
  group leakage, отсутствие класса, воспроизводимость и неизменность inputs.
- Notebook: реальные размеры/баланс folds; объяснить вложенность CV в train.
  Fit и метрик моделей на этом шаге ещё нет.
- Самопроверка: почему одинаковый seed без сохранённого manifest недостаточен?
- Промпт: «Выполни только D4-02 из PLAN.md: загрузка сохранённых splits и
  одинаковые train-only group-aware CV folds. Обнови notebook и проверки».
- Свидетельство (2026-10-16): добавлены `data/experiment_splits.py` с
  `load_day04_inputs` (Day02 loader + Day03 config/provenance/checksum/schema/
  ID/order/label/group/disjointness checks) и `build_train_cv_folds`
  (`StratifiedGroupKFold`, seed 42; fold manifest и positional indices только
  для train). Повторный запуск даёт тот же manifest; отсутствующий/повреждённый
  reference и несовпадения IDs/groups отвергаются; тестовые/validation IDs в CV
  не попадают. Реальные размеры Day03: train 2210 rows/2206 groups,
  validation 553/552, test 690/690. Train CV folds: validation 737/736/737 rows,
  735/735/736 groups; все три класса представлены в обеих частях каждого fold.
  Notebook D4-02 выполнен из корня и `notebooks/`. Проверки: 8 pytest passed,
  type gate 0 errors/warnings, compileall и `git diff --check` прошли. Модели и
  метрики качества не запускались; Day02/Day03 files не изменены.

### D4-03 — общий CV runner и baseline на тех же folds

- [x] Реализовать небольшой runner одного кандидата и измерить baseline CV.
- Зависит от D4-02. Вход: unfitted Pipeline, train и готовые folds;
  выход: fold scores, mean/std macro F1, accuracy и контекст запуска.
- Переиспользовать baseline builder/evaluation; для каждого fold новый clone
  Pipeline. TF-IDF fit только на fold train, matrices остаются sparse.
  Не передавать runner таблицы outer validation/test и не вычислять их метрики.
- Фиксировать параметры, fold ID, размеры/число features, fit/score time и
  сходимость. Timing — диагностическое измерение, не детерминированный output.
  ConvergenceWarning/fit failure — явная неуспешная проверка, не скрытый NaN.
- Проверка: fold-validation-only term отсутствует в vocabulary; нет общего
  fitted state между folds; labels/order/scoring верны; повтор scores;
  malformed folds отклоняются до fit. Импорт не запускает training.
- Notebook: реальные baseline fold scores, среднее и разброс; объяснить,
  почему Day03 validation macro F1 нельзя напрямую сравнить с CV mean.
- Самопроверка: почему нельзя один раз fit TF-IDF перед cross-validation?
- Промпт: «Выполни только D4-03 из PLAN.md: reusable CV runner и baseline
  на фиксированных folds. Обнови notebook; без новых моделей и tuning».
- Свидетельство (2026-10-16): добавлен `models/cv_runner.py` с полной
  preflight-валидацией fold assignments, отдельным `clone` на fold, fit TF-IDF
  только на fold train, sparse feature checks, контекстными ошибками, явным
  отказом при ConvergenceWarning и fold-level метриками/таймингами/параметрами.
  Pytest покрывает leakage vocabulary, независимость clones, повторяемость
  scores, malformed folds до clone, prefit Pipeline и convergence failure;
  26 целевых тестов прошли. Реальный baseline на folds 0/1/2 получил macro F1
  0.634554/0.698353/0.659923 (mean 0.664277, std 0.026227), accuracy
  mean 0.792315 (std 0.011444), 5000 features на каждом fold, без convergence
  warnings. Сохранённый D4-03 checkpoint в `notebooks/day04_experiments.ipynb`
  выполнен чистым kernel из корня и `notebooks/`; отдельно отмечено, что эти
  CV scores не являются прямым сравнением с одиночным Day03 validation score.
  `scripts/check_types.py`: 0 errors/warnings; `pip check` clean.

### D4-04 — эксперимент с word n-grams

- [x] Проверить word unigram TF-IDF + прежнюю LogisticRegression.
- Зависит от D4-03. Единственное изменение относительно baseline:
  ngram_range=(1, 1); max_features=5000 и остальные параметры сохранить.
- Выполнить CV общим runner на тех же folds; показать paired fold scores,
  mean/std, delta macro F1 к CV baseline и фактическое число features.
  Сравнивать относительно CV baseline, а не Day03 validation score.
- Проверка: конфигурация отличается только ngram_range; train-only fit,
  source alignment и сходимость подтверждены. Новые tests нужны только
  для новых содержательных контрактов, не для повторения runner tests.
- Notebook: гипотеза о контексте bigrams против размера/разреженности словаря,
  реальные результаты и вывод «помогло/не помогло на этих folds».
- Самопроверка: почему добавление bigrams не гарантирует улучшение?
- Промпт: «Выполни только D4-04 из PLAN.md: unigram experiment против
  baseline на прежних CV folds. Обнови notebook и свидетельство».
- Свидетельство (2026-10-16): unigram pipeline создан из baseline config с
  единственным изменением `ngram_range=(1, 1)` и оценён общим runner на folds
  D4-03. Paired macro F1 deltas по folds 0/1/2: +0.019131/-0.000779/-0.001302;
  unigram mean/std 0.669960/0.019630 против baseline 0.664277/0.026227,
  средний delta +0.005683. Accuracy mean 0.789599 против 0.792315. Число
  features: 4959/5000/5000 против baseline 5000 на каждом fold; convergence
  warnings отсутствуют. Вывод: небольшой выигрыш среднего macro F1 на этих
  folds, но не на каждом fold и со снижением accuracy — сигнал, не гарантия.
  Notebook clean-kernel исполнен из корня и `notebooks/`; type gate 0 errors.

### D4-05 — эксперимент с LinearSVC

- [x] Сравнить LinearSVC с LogisticRegression при неизменном baseline TF-IDF.
- Зависит от D4-04. Создать новый unfitted Pipeline с параметрами D4-01;
  не менять n-grams, vocabulary limit, class_weight или preprocessing.
- Запустить общий CV runner, сравнить paired fold scores и mean/std с baseline.
  При несходимости выяснить причину, зафиксировать техническую коррекцию
  iteration settings; не скрывать warning и не выбирать её по качеству test.
- Проверка: независимые unfitted builders, параметры Pipeline, classes=[0,1,2],
  sparse input, aligned predictions и отсутствие fit при import/build.
- Notebook: linear margin classifier и отличие от LogReg; LinearSVC score
  не probability, predict_proba не требуется. Реальные scores и ограничения.
- Самопроверка: что меняется при прежних features и другом classifier?
- Промпт: «Выполни только D4-05 из PLAN.md: LinearSVC experiment на прежних
  TF-IDF и CV folds. Обнови notebook и проверки, без tuning».
- Свидетельство (2026-10-18): добавлен `build_linear_svc_pipeline` с явными
  SPEC параметрами; builder возвращает независимые unfitted Pipeline на baseline
  TF-IDF. Тесты подтверждают параметры, три класса `[0, 1, 2]` и aligned output.
  Общий CV runner использовал те же folds и 5 000 features. Macro F1 folds
  0/1/2: 0.762359/0.800847/0.742939, paired delta к baseline
  +0.127805/+0.102494/+0.083016; mean/std 0.768715/0.024064. Accuracy
  mean/std 0.836660/0.014334. ConvergenceWarning не было. Notebook clean-kernel
  исполнен из корня и `notebooks/`; 8 targeted pytest tests passed, type gate
  0 errors/warnings.

### D4-06 — ограниченный подбор regularization C

- [x] Проверить LogReg C ∈ {0.1, 1.0, 10.0} с baseline TF-IDF.
- Зависит от D4-05. Использовать GridSearchCV над полным Pipeline, прежние
  folds и scoring macro F1 с фиксированными labels/zero_division;
  error_score='raise', refit=False. Остальные параметры неизменны.
- C=1 — контроль: fold scores должны совпасть с D4-03 в том же runtime;
  это не новый независимый эксперимент. Не расширять grid по результатам,
  не объединять автоматически n-gram/SVC эксперименты с подбором C.
- Выход: полная компактная таблица параметров, fold scores, mean/std и
  выбранное C по правилу D4-01. Выбор между всеми направлениями — в D4-07.
- Проверка: grid использует ровно сохранённые train folds, TF-IDF внутри
  каждого fit; нет validation/test fit/score, failures не маскируются;
  повторяемость scores и контроль C=1. Не делать framework для tuning.
- Notebook: смысл C, сильная/слабая regularization, цена перебора
  и риск подгонки под CV; улучшение может отсутствовать.
- Самопроверка: почему лучший CV score после перебора — не финальная test оценка?
- Промпт: «Выполни только D4-06 из PLAN.md: ограниченный GridSearchCV для
  LogReg C на прежних folds. Обнови notebook; не расширяй search space».
- Свидетельство (2026-10-18): добавлен компактный `search_logreg_c` с
  `GridSearchCV` над полным Pipeline, ровно тремя C, сохранёнными folds,
  scoring macro F1 с labels `[0, 1, 2]` и `zero_division=0`, дополнительным
  accuracy scorer, `refit=False` и `error_score="raise"`. ConvergenceWarning
  повышается до явной ошибки. Средний macro F1 для C=0.1/1.0/10.0:
  0.381356/0.664277/0.759263 (std 0.023907/0.026227/0.023086); accuracy
  mean 0.672852/0.792315/0.832588. C=1 scores совпали с D4-03 для обеих
  метрик; warnings отсутствовали. C=10 — лучший в grid, общий выбор остаётся
  D4-07. Два pytest теста прошли (фиксированный grid/control и повторяемость),
  type gate без errors/warnings, notebook выполнен из корня и `notebooks/`.

### D4-07 — сравнение, фиксация выбора и validation confirmation

- [x] Свести результаты, выбрать конфигурацию по CV и сохранить артефакты Day04.
- Зависит от D4-06. Таблица включает baseline, unigram, LinearSVC и все C;
  одинаковые folds, mean/std macro F1, accuracy, delta к CV baseline.
  Явно отделить три направления экспериментов от числа конфигураций/fit.
- Выбрать вариант по заранее зафиксированному правилу, сохранив baseline
  при его победе. Fit выбранного Pipeline на полном Day03 train, затем
  один validation predict; сравнить с сохранёнными Day03 baseline predictions
  на той же последовательности source IDs. Не переизбирать по validation.
- Показать validation macro F1/accuracy и per-class metrics; absolute delta
  и относительную разницу macro F1 в процентах при baseline > 0.
  Снижение validation score записать честно; не запускать новый tuning.
- Выход: простой offline runner/CLI и `configs/experiments_day04.json`;
  `reports/day04/` с effective config, CV fold manifest/summary/results,
  selection.json (candidate ID, полные параметры, правило и rationale),
  aligned validation_predictions.csv, metrics.json и provenance.json.
  Provenance связывает Day02/Day03 hashes, folds, configs, seed, версии и
  class order. Сохранить все результаты, включая отрицательные.
- Проверка: CSV/JSON roundtrip, metrics из сохранённых predictions,
  воспроизводимость folds/scores/выбора/predictions, корректные deltas/ties,
  baseline остаётся допустимым победителем. Inputs и Day03 reference неизменны;
  timings не участвуют в побайтном сравнении. Artifacts локальные/ignored;
  код/config/docs отслеживаются. Нет model bundle и test predictions/metrics.
- Notebook: итоговые CV и validation таблицы отдельно, объяснение выбора
  и наблюдаемых trade-offs; причинные утверждения и значимость не обещать.
- Самопроверка: что делать, если CV победитель хуже baseline на validation?
  Почему эта validation оценка ещё не заменяет независимый test?
- Промпт: «Выполни только D4-07 из PLAN.md: сравнение/выбор по CV,
  validation confirmation и воспроизводимые Day04 artifacts/CLI».
- Свидетельство (2026-10-18): реализован offline CLI/module
  `run_experiments_day04` и фиксированный `configs/experiments_day04.json`.
  Сравнены шесть конфигураций на общих train folds; лучший по mean macro F1 —
  `linear_svc` (0.768715, std 0.024064); tie отсутствует. Он fit-ился один раз
  на полном train и оценён на прежних 553 validation строках: macro F1 0.805961
  против Day03 baseline 0.720965 (delta +0.084995, +11.79% relative), accuracy
  0.848101 против 0.813743; per-class metrics сохранены. Test в fit/score не
  передавался. CSV/JSON roundtrips, predictions alignment и input immutability
  проверены до сохранения reports/day04 artifacts; provenance содержит hashes,
  versions, seed и class order. Повторные CLI и clean-kernel notebook прогоны
  дали побайтно одинаковые SHA-256 всех config/artifact файлов. Notebook из
  корня и `notebooks/`; CLI: `PYTHONPATH=src venv/bin/python -m
  finnews_sentiment.models.run_experiments_day04 --root . --config
  configs/experiments_day04.json`. 22 targeted pytest tests passed, type gate 0
  errors/warnings, compileall и `git diff --check` прошли.

### D4-08 — выводы и итоговая приёмка Day04

- [x] Завершить учебный checkpoint и подтвердить воспроизводимость дня.
- Зависит от D4-07. Выход: полный `day04_experiments.ipynb`,
  `docs/day04.md` с командами и `reports/day04/observations.md` с выводами.
  Согласовать ignore policy: итоговые observations доступны для review,
  generated experiment artifacts локальные, как в Day03.
- Описать по фактическим результатам: что проверяли, что помогло/не помогло,
  выбранный вариант и основания выбора, CV variability, validation delta,
  дисбаланс и per-class trade-offs, ограниченный бюджет поиска и риск CV overfit.
  Отсутствие улучшения — допустимый исследовательский результат.
- Проверка: полный pytest suite, type gate, compileall, diff check;
  отдельно реальный offline CLI repeat и notebook из корня/`notebooks/`
  в чистых kernels. Не перезаписывать работающий пользовательский notebook.
- Приёмка: проверены три направления, сравнение с baseline на общих folds
  сохранено, выбор/config воспроизводимы, validation confirmation выполнен,
  held-out test не использован; каждый ID имеет проверенный checkpoint.
- Передача Day05: selection/config, прежний split manifest и aligned validation
  predictions; не начинать анализ ошибок, итоговый test или inference.
- Самопроверка: что даёт основание выбрать модель, а что остаётся неизвестным?
- Промпт: «Выполни только D4-08 из PLAN.md: выводы, документация и итоговая
  приёмка Day04. Не начинай Day05; запиши фактические проверки».
- Свидетельство (2026-10-18): добавлены `docs/day04.md`, tracked
  `reports/day04/observations.md` и D4-08 notebook section с проверкой hashes,
  per-class trade-offs, выводами и самопроверкой. Ignore policy сохраняет
  observations для review, generated reports остаются локальными.
  Полный suite: 190 pytest tests и 46 subtests passed; compileall src/tests/scripts
  и `git diff --check` прошли. Исправленный type gate: 35 Python файлов и
  4 notebook exports, 0 errors/warnings. Обнаружено, что прежний gate пропускал
  notebook exports в скрытой `.cache` (`filesAnalyzed=0`): прежнее свидетельство
  D4-07 о type gate не подтверждало типы code cells. Теперь exports в нескрытой
  временной папке, coverage проверяется по JSON summary, есть 3 регрессионных
  pytest cases (валидная/ошибочная ячейка, пропущенный анализ). Исправлены
  `.loc[idxmax()]` с Hashable в D4-06 и доступ к вложенным object metrics D4-07.
  Два реальных offline CLI запуска и два clean-kernel notebook исполнения
  (из root/`notebooks/`, 9/9 code cells каждое) дали одинаковые hashes всех
  9 generated artifacts и сохранили 11 input hashes. Полные executed copies и
  acceptance hashes: `.cache/day04-checkpoints/`; в исходном notebook сохранены
  прежние outputs и добавлен output D4-08. Пользовательские kernels не прерывались.
  Победитель остаётся LinearSVC: CV macro F1 0.768715, validation 0.805961
  против baseline 0.720965; test не передавался estimator. Day05 не начат.
- Коррекция повторных Pylance diagnostics (2026-10-18): прежний guard в D4-07
  не задавал устойчивый тип для D4-08. Склеенный export проходил, но отдельные
  cell modules воспроизвели ровно две сообщённые ошибки object.__getitem__.
  `Day04Run.metrics` теперь имеет вложенный TypedDict contract; guards из
  notebook удалены. Gate проверяет sequential exports и cell-boundary modules:
  35 Python файлов + 4 exports/37 cells (41 файл), 0 errors/warnings. Добавлены
  два regression cases на потерю narrowing и корректный TypedDict; в AGENTS
  закреплена обязательная репродукция конкретных Pylance diagnostics.
  Полный suite: 192 tests и 46 subtests passed; compileall/diff check прошли.
  Clean-kernel из root и `notebooks/`: по 9/9 code cells без ошибок, все 9
  artifact hashes и 11 input hashes прежние. Copies/evidence сохранены в
  `.cache/pylance-cell-fix/`, прежние notebook outputs/kernels не изменялись.
  CLI gate не является прямым запуском Pylance в редакторе; полнота его
  совпадения с Pylance не заявляется. Day05 не начат.

## Day05 — error analysis и финальная оценка

Детализация `classification_day05.txt` по SPEC §10. D5-01…D5-06 завершены.
Notebook содержит validation analysis, ручной разбор, проверенный final cache,
воспроизводимый offline report и итоговый checkpoint с handoff Day06.
Один ID — один промпт Codex. Порядок: D4-08 → D5-01 → D5-02 → D5-03 →
D5-04 → D5-05 → D5-06. До реализации завершить приёмку Day04; наличие
D4-07 artifacts само по себе не закрывает D4-08.

Цель: объяснить ошибки уже выбранной по CV модели на validation, затем получить
финальную held-out оценку замороженного решения. Day05 не меняет preprocessing,
модель, параметры, splits или правило выбора. Гипотезы улучшений записываются
для будущего эксперимента; найденные ошибки не запускают новый tuning.
D5-04 сохраняет прежний ID, на который уже ссылаются Day03/Day04.
Inference bundle, serialization и API новых текстов остаются Day06.

### Рекомендуемые модели Codex и reasoning effort

Рекомендации на 2026-10-18 — инженерный выбор для этих задач, не измеренный
benchmark проекта. Модель Codex пишет код; выбранный финансовый classifier
из Day04 — отдельная ML-модель, которую эти настройки не меняют.

| Пункт | Рекомендация | Причина |
| --- | --- | --- |
| D5-01 | GPT-6.1 Sol, high | Проверка provenance, схем и alignment нескольких artifacts |
| D5-02 | GPT-6 Luna, high | Локальная задача с однозначными инвариантами матрицы |
| D5-03 | GPT-6.1 Sol, high | Разбор реальных ошибок и осторожные исследовательские выводы |
| D5-04 | GPT-6.1 Sol, high | Freeze, fit boundary, replay и безопасный повтор финальной оценки |
| D5-05 | GPT-6.1 Sol, medium | Сборка уже проверенных функций в offline report runner |
| D5-06 | GPT-6.1 Sol, high | Проверка согласованности всего дня и clean-kernel воспроизведения |

Если нужна одна настройка на весь день: GPT-6.1 Sol + high. Для D5-02 Sol +
medium тоже разумен. Astra + high имеет смысл при сложном неразрешённом дефекте
контрактов; Extra High (`xhigh`)/Max не нужны по умолчанию. Сначала уточнять
входы, критерии и диагностировать провал проверки, затем повышать effort.
Доступность зависит от аккаунта/клиента; при отсутствии 6.1 Sol использовать
доступный GPT-6 Sol с тем же стартовым effort и теми же gates.

Основание: [official OpenAI documentation: Models](https://learn.chatgpt.com/docs/models)
и [Using GPT-6](https://developers.openai.com/api/docs/guides/latest-model).
Они описывают роли моделей и effort; распределение по ID — наша рекомендация.

### Общий checkpoint каждого пункта

- Reusable логика — `src/finnews_sentiment/`, новые содержательные tests — pytest.
  Переиспользовать Day03 metrics и Day04 loaders/builders, без побочного refactor.
- В `notebooks/day05_error_analysis.ipynb` после каждого ID отдельный раздел:
  цель → теория → вызов reusable кода → реальные outputs → интерпретация →
  самопроверка. До D5-04 в notebook нет test predictions/metrics.
- После Python/code cells: `venv/bin/python scripts/check_types.py`, целевые pytest,
  `venv/bin/python -m compileall -q src tests`, `git diff --check`.
  Отдельно реальная offline проверка и clean-kernel notebook из корня и
  `notebooks/`; сохранять outputs, не прерывать работающий kernel.
- Закрывать ID только с фактическими свидетельствами; недоступную проверку
  записать явно. Финальный полный suite — D5-06. Commit только по запросу.

### D5-01 — проверенная таблица validation predictions и notebook setup

- [x] Загрузить Day04 reference и собрать aligned таблицу для анализа без fit.
- Зависит от D4-08. Вход: Day02 processed data/provenance, Day03 manifest,
  Day04 selection/config/provenance, validation predictions/metrics.
- Добавить reusable loader/table builder (например, `models/error_analysis.py`),
  используя точные существующие имена колонок и hashes по SPEC §10.
  Join по `source_row_id`, а не pandas index/позиции; проверить затем исходный
  validation порядок, уникальность и полноту IDs, true labels и class order.
- Выход: все validation строки с raw/clean text, true/pred labels, group ID и
  `is_error`; errors — отдельный фильтр. Создать начальный day05 notebook:
  N_valid, число/доля ошибок, реальные примеры, источник выбранного candidate.
  Сохранённые Day04 predictions являются reference: не перезапускать CV/fit.
- Проверка: переставленные/пропущенные/лишние/повторные IDs, неверные labels,
  checksum mismatch и неизменность входов; ошибки схем диагностируются явно.
- Самопроверка: почему одинаковой длины двух таблиц недостаточно для alignment?
- Промпт: «Выполни только D5-01 из PLAN.md по SPEC §10; обнови notebook и PLAN,
  выполни общий checkpoint. Не обучай модель и не оценивай test».
- Свидетельство (2026-10-18): `models/error_analysis.py` переиспользует
  `load_day04_inputs`, проверяет Day04 input/output SHA-256, provenance,
  effective config, candidate/parameters, CV selection и восстановление metrics.
  `build_validation_table` выполняет one-to-one joins по source IDs и отклоняет
  перестановки, неполное покрытие, повторы, неизвестные/несогласованные labels,
  пропуски, пустые тексты и пересечение groups. `errors` — отдельный фильтр.
- Реальные данные: `linear_svc` из сохранённого `selection.json`, таблица
  `(553, 9)`, 84 ошибки (15.189873%). Никакого fit/CV/predict/test evaluation;
  в offline test вызовы Pipeline.fit/predict запрещены monkeypatch.
- Checkpoint: 16 новых offline tests; общий pytest — 208 passed,
  46 subtests passed; compileall и git diff --check успешны. Type gate:
  37 project files и 44 notebook modules (5 sequential exports + 39 cell-boundary
  modules), 0 errors/warnings. CLI Pyright не является полным запуском Pylance
  внутри VS Code.
- `notebooks/day05_error_analysis.ipynb` выполнен с чистого kernel из корня и
  `notebooks/`, реальные outputs сохранены. SHA-256 25 входных файлов
  configs/processed/Day02–Day04 до/после execution совпали. D5-02 не начат.

### D5-02 — confusion matrix и направления ошибок

- [x] Построить counts и row-normalized confusion matrix validation.
- Зависит от D5-01. Вход — aligned таблица, выход — две матрицы `(3, 3)`,
  CSV/PNG и таблица шести направленных off-diagonal пар с counts.
- Строки = true, столбцы = predicted; порядок negative/neutral/positive.
  Показать count и долю по истинному классу, подписать axes и denominators.
  Для нулевого support нормированная строка = 0; не скрывать отсутствие класса.
  Переиспользовать `models/evaluate.py` для per-class precision/recall/F1/support.
- Проверка: известная асимметричная fixture обнаруживает transpose; сумма counts
  = N, диагональ = correct, off-diagonal = errors, row sums = support;
  отсутствующий predicted/true класс и сохранение графиков без GUI.
- Notebook: оба графика, per-class metrics и фактическая интерпретация наиболее
  частых направлений; не сравнивать сырые counts классов как error rates.
- Самопроверка: чем true negative → predicted neutral отличается от обратной пары?
- Промпт: «Выполни только D5-02 из PLAN.md: validation confusion matrix,
  проверки и notebook checkpoint; не переходи к ручному анализу или test».
- Свидетельство (2026-10-18): `models/confusion.py` строит counts и normalized
  `(3, 3)` в фиксированном class order, шесть directed pairs с count,
  true_support и fraction_of_true; per-class metrics через общий
  `evaluate_predictions`. Для support=0 строка нулевая и помечена absent.
  Headless Agg PNG подписывают true/predicted axes, support и нормировку.
- Реальные validation counts: `[[47, 15, 5], [3, 326, 15], [4, 42, 96]]`;
  N=553, correct=469, errors=84. Positive → neutral: 42/142=29.58%,
  neutral → positive: 15/344=4.36%. Recall negative/neutral/positive:
  70.15%/94.77%/67.61%; macro F1=0.805961. Интерпретация использует true
  support, не сравнивает сырые counts как error rates.
- Сохранены только Day05 artifacts: `validation_confusion_{counts,normalized}`
  CSV/PNG и `confusion_pairs.csv` в ignored `reports/day05/`; оба графика,
  per-class metrics, pairs, интерпретация и самопроверка — в notebook.
- Checkpoint: 10 новых offline tests (asymmetry/transpose, totals, support,
  absent true/predicted classes, zero errors, invalid labels, headless export
  и CSV roundtrip). Общий pytest: 218 passed, 46 subtests passed;
  compileall и git diff --check успешны. Type gate: 39 project files и
  46 notebook modules (5 sequential exports + 41 cell-boundary modules),
  0 errors/warnings; CLI Pyright не заменяет Pylance внутри VS Code.
- Notebook выполнен с чистого kernel из корня и `notebooks/`, outputs
  сохранены; 25 входных configs/processed/Day02–Day04 files неизменны по
  SHA-256. PNG проверен визуально. Fit/predict/CV/test evaluation не запускались;
  ручной разбор D5-03 не начат.

### D5-03 — реальные примеры и ручные failure categories

- [x] Разобрать реальные validation ошибки и записать проверяемые наблюдения.
- Зависит от D5-02. Для каждой из шести непустых true→pred пар выбрать до трёх
  примеров по возрастанию `source_row_id`, без cherry-picking; пустые пары
  отметить явно. Показывать raw/clean text, labels, IDs и group ID.
- Reusable функция выбирает examples; объяснения и категории записываются вручную
  в reviewable таблицу по SPEC §10. Не приписывать модели доказанную причину.
- Возможные категории: mixed sentiment, financial context, negation/comparison,
  insufficient context/ambiguous label, preprocessing/representation limitation,
  other/unclear. Это стартовый словарь, наличие категорий определяется примерами.
- Сохранить `docs/day05_failure_cases.csv`: одна primary category и rationale на
  каждый выбранный ID; observation отдельно от hypothesis. Не менять labels.
  Для duplicate groups отметить повторы и число уникальных groups.
- Выводы: 2–4 наблюдения с IDs и приоритетные будущие гипотезы с предлагаемой
  train/CV проверкой. Counts категорий описывают только просмотренную выборку.
  Не требовать категорий/ошибок, которых в реальных predictions нет.
- Проверка: детерминированная выборка, нулевые ошибки/пустые пары, валидность
  annotation IDs, отсутствие повторов/пропущенных annotations и test IDs.
- Notebook: examples, ручные annotations, ограничения и вопросы самопроверки.
- Самопроверка: доказывает ли один пример, что модель не понимает отрицания?
- Промпт: «Выполни только D5-03 из PLAN.md: разбери реальные validation ошибки,
  сохрани annotations/выводы, обнови notebook; не меняй модель и не открывай test».
- Свидетельство (2026-10-18): `models/failure_cases.py` выбирает первые три
  source IDs на пару (class order, внутри пары ascending source_row_id), явно
  отмечает пустые пары, не удаляет duplicate groups. Возвращает group counts
  по всей validation, всем errors и выбранным examples. Ручные annotations
  проверяются по точному набору IDs/schema/categories, join one-to-one по ID;
  observation/rationale непусты, hypothesis допускает пустую строку.
- Реальные данные: все шесть пар непусты, 18 examples / 18 groups из 84 errors;
  выбранные groups не повторяются в validation. Всего validation: 553 rows /
  552 groups, errors: 84 rows / 84 groups. ID 2854/2855 похожи, но это разные
  exact-clean-text groups. `docs/day05_failure_cases.csv`: вручную разобраны все
  18 строк; категории financial context 12/18, negation/comparison 3/18,
  insufficient context/ambiguous label 2/18, other/unclear 1/18. Denominator
  относится только к просмотренным examples; остальные категории не навязаны.
- `docs/day05.md` и notebook содержат четыре наблюдения с IDs/направлениями,
  отделённые hypotheses и две приоритетные будущие train/CV проверки; причины
  ошибок не объявлены доказанными, labels/candidate/config не менялись.
  Сохранены ignored `validation_analysis.csv`, `validation_errors.csv` и
  annotated `validation_examples.csv`; full errors отделены от examples.
- Checkpoint: 21 новый offline test — deterministic order/cap, zero errors,
  пустые пары, duplicate groups, blank hypothesis, annotation ID/schema/category
  contracts и исключение train/test IDs. Общий pytest: 239 passed,
  46 subtests passed; compileall и git diff --check успешны. Type gate:
  41 project files, 49 notebook modules (5 sequential exports + 44 cell-boundary
  modules), 0 errors/warnings. CLI Pyright не заменяет Pylance внутри VS Code.
- Notebook выполнен с чистого kernel из корня и `notebooks/`, все реальные
  examples/annotations и outputs сохранены; SHA-256 25 исходных inputs и файла
  ручных annotations до/после совпали. Fit/predict/CV/test evaluation не
  запускались, test examples не просматривались; D5-04 не начат.

### D5-04 — freeze и однократная финальная held-out оценка

- [x] Зафиксировать протокол и оценить единственную выбранную конфигурацию на test.
- Зависит от D5-03. До доступа к test predictions/metrics записать
  `reports/day05/frozen_protocol.json`: candidate/полные параметры из Day04,
  preprocessing, hashes inputs/manifest/selection, train IDs, seed, versions,
  class order, metric policy и fit policy `day03_train_only`.
- Day04 не сохраняет fitted model. Восстановить exact выбранный Pipeline через
  прежний builder; fit только на Day03 train, TF-IDF внутри Pipeline. Не
  добавлять validation к train: сохраняем модель, поведение которой разбирали.
- Сначала проверить точное совпадение replay validation predictions с reference
  Day04; при mismatch/ConvergenceWarning остановиться до test. Эта проверка
  воспроизводимости не выбирает заново candidate и не запускает CV.
- Затем один раз predict на прежнем test, сохранить aligned predictions `(N_test,)`,
  macro F1, accuracy, per-class metrics и test confusion matrix. Test не влияет
  на параметры/порог/seed/предобработку. Рост относительно validation не требуется.
  На test не оценивать заново кандидатов, baseline или Dummy.
- Повторный запуск использует проверенный cache по frozen protocol и hashes;
  не делает новый fit/predict. Повреждённый/неполный cache или иной protocol —
  явная ошибка, без автоматического overwrite/re-evaluation. Свидетельство
  фиксирует, первый это запуск или reuse, и пределы многофайловой записи.
- Проверка: spies подтверждают train-only fit, validation replay до test,
  остановку при mismatch, test predict только на первом успешном запуске,
  cache reuse без fit/predict, отказ при checksum/protocol mismatch.
  Fixtures в tests; реальный test запускается отдельно после offline gates.
- Notebook: freeze → replay → final metrics с указанием split и cache status;
  test результаты не превращаются в новый список изменений текущей модели.
- Самопроверка: почему tuning после чтения test разрушает независимость оценки?
- Промпт: «Выполни только D5-04 из PLAN.md по SPEC §10: freeze, train-only
  replay, финальная test оценка и проверяемый cache; без tuning и Day06 bundle».
- Свидетельство (2026-10-18): отдельный `models/final_evaluation.py` реализует
  явный CLI `PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.final_evaluation`.
  Первый запуск exclusive-create сохраняет frozen protocol до fit/test,
  включает параметры selection и полные get_params TF-IDF/classifier, policy,
  train/validation/test IDs, seed/classes/metrics, input hashes и runtime versions.
  Неверные inputs/config/версии Day03/Day04 останавливают запуск до freeze.
- Первый реальный run — `created`: candidate `linear_svc`, train-only fit на
  2210 строках, sparse TF-IDF `(2210, 5000)`, exact replay всех 553 validation
  predictions/IDs/truth/order, без ConvergenceWarning, затем один test predict.
  Test N=690, accuracy=0.8565217391, macro F1=0.7970501582;
  counts `[[48, 21, 15], [1, 414, 13], [2, 47, 129]]`.
  Negative/neutral/positive precision: 0.941176/0.858921/0.821656;
  recall: 0.571429/0.967290/0.724719; F1: 0.711111/0.909890/0.770149;
  support: 84/428/178. Ни tuning, ни baseline/Dummy/candidate comparison на test.
- Сохранены frozen protocol, aligned test predictions/metrics и counts/normalized
  CSV/PNG с подписями Test. Completion marker `final_evaluation.json` записан
  последним после проверки roundtrips/bytes; содержит protocol hash
  `4a9a778085b8f082a20374bf46e1a2053a72aa9a6ee872e3e252dbb8a2570a37`,
  runtime/replay metadata и hashes шести outputs, без собственного checksum.
  Это не общая транзакция: crash может оставить partial artifacts; exactly-once
  prediction при аварии до сохранения cache не гарантируется.
- Реальные function/CLI повторы — `reused`; `run_final_evaluation` и read-only
  `load_final_evaluation` проверены с запрещёнными Pipeline.fit/predict.
  Все восемь final files остались byte-identical. Неполный/повреждённый cache
  или иной protocol вызывают ошибку с диагностикой, без overwrite/delete/refit.
- 19 новых offline tests проверяют порядок freeze/fit/replay/test, train-only
  fit, replay mismatch/ConvergenceWarning до test, повтор без fit/predict,
  input/config/version mismatch, повреждение marker/protocol/checksums,
  ID/order/truth/predictions, metrics/matrices и partial cache refusal.
  Полный pytest: 258 passed, 46 subtests passed; после уточнения full parameter
  snapshot focused 19 tests повторно прошли. Compileall и git diff --check OK.
  Type gate: 43 project files и 51 notebook modules (5 sequential exports +
  46 cell-boundary modules), 0 errors/warnings; CLI Pyright не заменяет Pylance.
- Notebook выполнен с чистого kernel из корня и `notebooks/` через cache-only API,
  показывает split/cache status, freeze/replay policy, shapes, per-class metrics
  и обе test matrices; outputs сохранены, PNG просмотрен. Hashes 26 inputs
  (включая manual annotations) и восьми final artifacts неизменны. `docs/day05.md`
  фиксирует результаты/команды/ограничения. D5-05 и Day06 не начаты.

### D5-05 — offline отчёт и сохранение результатов анализа

- [x] Собрать report runner поверх D5-01…D5-04 без новой оценки test.
- Зависит от D5-04. CLI/module `models/run_error_analysis_day05.py` читает
  reference, ручные annotations и проверенный final cache; сохраняет таблицы,
  графики, `analysis_summary.json` и `provenance.json` по SPEC §10.
  Отсутствующий test cache требует отдельно выполнить D5-04, не запускает его
  молча. Не делать download, CV или fit при сборке отчёта.
- Проверять согласованность counts/metrics/annotations и CSV/JSON roundtrip;
  outputs принадлежат Day05, Day02…Day04 не изменяются. CLI повторно работает
  offline; сравниваемое содержимое не зависит от timestamp.
- `docs/day05.md`: входы/выходы, отдельная команда финальной оценки и команда
  report-only, первый запуск/cache reuse, правила ошибок, граница Day06.
  `reports/day05/observations.md`: фактические validation conclusions и отдельный
  final test результат/ограничения. Ключевые выводы также доступны в tracked docs.
- Проверка: end-to-end report на fixtures без fit/predict, cache отсутствует/
  повреждён, input immutability, roundtrip и repeat; отдельный real offline run.
- Notebook: вызывать те же функции; report-only повтор не переоценивает test.
- Самопроверка: какие artifacts нужны, чтобы связать метрику с конкретным запуском?
- Промпт: «Выполни только D5-05 из PLAN.md: offline report runner, artifacts,
  docs и notebook checkpoint; используй final cache без нового fit/test predict».
- Свидетельство (2026-10-18): `models/run_error_analysis_day05.py` предоставляет
  reusable `run_error_analysis_day05(root)` и report-only CLI
  `PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.run_error_analysis_day05`.
  Читает только сохранённые Day02–Day04 reference, manual annotations и
  `load_final_evaluation` cache; отсутствующий/повреждённый cache не запускает D5-04.
  Import не выполняет I/O/training. Возвращаемые summary/provenance — TypedDict,
  чтобы сохранять контракт между notebook cells.
- Сохраняет 11 Day05 report outputs: analysis/errors/annotated examples,
  validation counts/normalized CSV/PNG, directed pairs, summary, observations,
  provenance. Final protocol/marker и test CSV/PNG/metrics — только inputs.
  CSV/JSON roundtrip, counts/errors/pair denominators, annotation contracts и
  input immutability проверяются до записи; записанные bytes сверяются повторно.
- `analysis_summary.json` фиксирует class order, policies, validation/test metrics,
  denominators и row/group counts отдельно; `observations.md` — фактические
  validation направления и ручные observations/hypotheses с IDs, отдельно cached
  final test и ограничения. `provenance.json` содержит 29 input checksums,
  10 output checksums, runtime versions и hashes frozen protocol/final marker;
  собственный checksum исключён, циклических ссылок нет. Timestamp отсутствует.
  Последовательная запись отчёта не является общей multi-file транзакцией.
- Реальный offline report: candidate linear_svc, validation N=553/errors=84,
  reviewed 18 rows/18 groups, final cache reused (test N=690, macro F1=0.797050).
  Две сборки API с запрещёнными Pipeline.fit/predict, CV, network и final-run
  дали byte-identical все 11 outputs; 34 source/final files неизменны.
  Отдельный реальный report-only CLI успешно использовал тот же cache.
- 15 новых offline tests: end-to-end zero-error/reviewed-error fixtures,
  exact report/API/CLI repeat, пустая hypothesis и CSV quoting/newlines,
  provenance/roundtrip/input immutability, cache/annotation/input corruption,
  изменение входа во время сборки и import без I/O. Общий pytest:
  273 passed, 46 subtests passed; compileall и git diff --check OK.
  Type gate: 45 project files, 53 notebook modules (5 sequential exports +
  48 cell-boundary modules), 0 errors/warnings; CLI Pyright не заменяет Pylance.
- Notebook: отдельный D5-05 section вызывает тот же runner, показывает summary,
  artifacts и provenance; clean kernels из корня и `notebooks/` подтверждены,
  outputs сохранены. SHA-256 всех 45 input/final/report files совпали до/после.
  `docs/day05.md` дополнен командами first-final/report-only, inputs/outputs,
  repeat/error policies и Day06 boundary. D5-06 и Day06 не начаты.

### D5-06 — итоговая приёмка и передача Day06

- [x] Проверить весь Day05 и зафиксировать фактическое выполнение критериев дня.
- Зависит от D5-05. Полный pytest suite, type gate, compileall, diff check;
  отдельно реальные artifact checks, повтор offline report CLI и полный notebook
  из корня/`notebooks/` в чистых kernels с reuse одного final cache.
- Проверить notebook: все переменные определены выше, отдельные sections по ID,
  outputs/графики сохранены, validation/test не смешаны, примеры реальные,
  counts сходятся, annotations и будущие гипотезы имеют ссылки на evidence.
- Объяснить macro F1, дисбаланс, row/group weighting, пределы малого holdout,
  conditional nature test после model selection и отсутствие доказанной live quality.
  Не называть разброс CV доверительным интервалом; не обещать улучшение.
- Передача Day06: frozen selected config, preprocessing policy, class mapping,
  checksums и fit policy `day03_train_only`; Day06 bundle должен воспроизвести
  это решение, без refit на validation/test по умолчанию.
- Самопроверка: чем диагностика validation, итоговая оценка test и inference
  на новых текстах отличаются по назначению и доступным действиям?
- Промпт: «Выполни только D5-06 из PLAN.md: приёмка Day05, полный checkpoint,
  документы и фактические свидетельства; не начинай Day06, commit не делай».
- Свидетельство (2026-10-18): приёмка SPEC §10 выполнена; production Python
  не менялся. `docs/day05.md` и отдельный D5-06 notebook section объясняют macro F1,
  дисбаланс, row/group weighting, ограниченность одного split/малого holdout,
  conditional nature test после train/CV selection и validation diagnosis;
  std CV не объявлен CI, live financial quality/прирост не обещаны.
- Offline suite: `venv/bin/python -m pytest tests -v` — 273 passed,
  46 subtests passed. Type gate — 45 project files и 54 notebook modules
  (5 sequential exports + 49 cell-boundary modules), 0 errors/warnings.
  CLI Pyright не является полным запуском Pylance внутри VS Code.
  `compileall -q src tests` и `git diff --check` успешны.
- Отдельный real audit: exact IDs/order/true labels, raw/clean text, annotation
  coverage, class order, counts/trace/support/normalized denominators и hash chain
  согласованы. Train: 2210 rows/2206 groups; validation: 553/552; test: 690/690.
  Errors: 84 rows/84 groups; reviewed: 18 rows/18 groups. Все splits disjoint
  по source IDs и groups; examples — только validation errors.
- Frozen test остался прежним: accuracy=0.8565217391, macro F1=0.7970501582;
  protocol SHA-256
  `4a9a778085b8f082a20374bf46e1a2053a72aa9a6ee872e3e252dbb8a2570a37`.
  Cache 8 files и report 11 outputs согласованы; provenance 29 input hashes /
  10 output hashes, без self-checksum. Все четыре PNG успешно декодированы.
- Report-only CLI выполнен дважды с запрещёнными real fit/predict/CV/network,
  оба раза cache reused. Все 48 configs/data (raw/interim/processed)/Day02–Day05/
  annotation files SHA-256 неизменны; report outputs byte-identical.
- Полный notebook исполнен из корня и `notebooks/` в отдельных чистых kernels:
  шесть sections D5-01…D5-06, 12 code cells с outputs, четыре PNG сохранены,
  ошибок execution нет. Финальный runtime checkpoint сверяет IDs/classes,
  annotations/counts/denominators, protocol/contracts и input/output hashes.
  Реальные artifact checks и clean-kernel execution отделены от offline tests.
- Handoff Day06: frozen selection/полные параметры, minimal-v1 `preprocess_text`,
  class mapping/order, Day03 manifest/train IDs, runtime versions/checksums и
  fit policy day03_train_only. Clean text → TF-IDF → LinearSVC; train `(2210, 5000)`,
  predictions `(N,)`. Fitted Pipeline пока не сохранён как bundle. Day06 должен
  воспроизвести прежнее решение без refit на validation/test и без удаления
  retained train duplicates; реализация bundle/API и детализация Day06 не начаты.
- Ограничения multi-file writes и partial cache зафиксированы: completion marker
  не транзакция, exactly-once при crash до cache не гарантирован; автоматическая
  переоценка test не допускается. Day05 принят, commit не выполнялся.

## Day06 — inference pipeline и сохранение модели

Планирование выполнено 2026-10-18 по `classification_day06.txt`, SPEC §11
и принятому handoff D5-06 / `docs/day05.md`; на этапе планирования менялись
только PLAN/SPEC. D6-01 реализовал loader/contracts/tests/notebook;
D6-02 — явный train-only build и fresh validation replay. Fitted result
существует в памяти, model files ещё не созданы.
Предварительные четыре пункта roadmap заменены восемью небольшими задачами.
Один ID — один промпт Codex. Порядок: D5-06 → D6-01 → D6-02 → D6-03 →
D6-04 → D6-05 → D6-06 → D6-07 → D6-08. Фактические статусы — ниже.

Цель: сохранить замороженный TF-IDF/classifier Pipeline и предсказывать новые
raw texts с прежней `preprocess_text` (`minimal-v1`), без переобучения при inference.
В Day05 fitted Pipeline не сохранён: D6-02 выполняет явный train-only fit
и validation replay; D6-03 сохраняет результат. Load/predict/CSV и дальнейший
повтор notebook с готовым bundle не делают fit. Не вызывать final evaluation
ради модели, не менять frozen config и не делать новый test predict/metrics.
HTTP API, UI, calibration, новые ML experiments и финальный README — вне дня.

### Общий checkpoint каждого пункта

- Reusable логика в `src/finnews_sentiment/`; новые содержательные tests — pytest.
  Переиспользовать текущие loaders/builders и read-only final-cache validation;
  не создавать второй несовместимый config или framework.
- После каждого ID обновлять `notebooks/day06_inference.ipynb` отдельным section:
  цель → теория → вызов reusable кода → реальные outputs → интерпретация →
  самопроверка. Создать notebook при D6-01, не при планировании.
- После Python/code cells выполнить `venv/bin/python scripts/check_types.py`,
  целевые pytest, `venv/bin/python -m compileall -q src tests`, `git diff --check`.
  Фиксировать число project/notebook modules и режимы export; CLI Pyright
  не является полным запуском Pylance внутри VS Code. Полный suite — D6-08.
- Отдельно проверить реальные локальные artifacts и notebook из корня/`notebooks/`
  в чистых kernels, сохранить outputs; не прерывать чужой работающий kernel.
  До D6-03 D6-02 section явно делает train-only fit; при D6-03 заменить этот
  вызов показом проверенных bundle metadata и вынести build в явную команду.
- День не меняет configs/data/reports Day02–Day05; проверять hashes.
  Готовый bundle при повторе использовать без fit. Не закрывать пункт без
  фактического результата/checkpoint; недоступные проверки назвать прямо.
  Каждый запрос заканчивается diff stat с untracked files и commit message;
  commit только по отдельному запросу.

### D6-01 — frozen inputs, типизированный контракт и notebook setup

- [x] Подготовить проверенный read-only вход для сборки inference artifact.
- Зависит от D5-06. Вход: Day02 data/provenance, Day03 config/manifest,
  Day04 selection/config/reference, Day05 frozen protocol/final cache.
- В `models/inference_artifact.py` зафиксировать небольшие typed структуры
  reference/bundle metadata и публичные signatures будущих build/save/load.
  Переиспользовать loaders и `load_final_evaluation`, без вызова
  `run_final_evaluation`, fit, predict, CV или записи предыдущих artifacts.
- Проверить hashes, schema, versions, selected candidate, полные effective
  параметры, `minimal-v1`, mapping/order, seed, `day03_train_only`, IDs/order
  и split disjointness по SPEC §11. Не зашивать baseline или fallback.
- Выход: typed reference с train/validation и frozen metadata; notebook setup
  явно подключает src из корня/`notebooks/`, показывает candidate, источники,
  shapes/counts и поток raw → preprocessing → TF-IDF → classifier.
- Проверки: missing/partial/corrupt references, иной config/runtime/policy,
  переставленные/пропущенные IDs; input immutability. В tests запретить
  fit/predict/network/final-run. Реальная проверка выполняется offline.
- Самопроверка: почему saved predictions и config не заменяют fitted estimator?
- Промпт: «Выполни только D6-01 из PLAN.md по SPEC §11: frozen input loader,
  typed contracts и notebook setup. Выполни общий checkpoint, обнови PLAN.
  Не обучай/не сохраняй модель, не вызывай predict/test evaluation. Commit не делай».
- Свидетельство (2026-10-18): `models/inference_artifact.py` предоставляет
  `load_inference_reference`, переиспользуя `load_final_evaluation`,
  `load_day04_inputs` и `load_validation_analysis`. Frozen protocol и marker
  фиксируются до чтения; input/final hashes сверяются после всех readers.
  Отсутствующий/partial/corrupt reference не пересоздаётся. Проверяются
  selection/effective parameters/runtime/policy/classes/IDs/order/groups.
- Typed `InferenceReference` отдаёт только train `(2210, 5)`, validation
  `(553, 5)` и validation prediction reference `(553, 3)`, split summary counts
  и `FrozenInferenceContract`. Candidate `linear_svc`, seed=42,
  fit policy `day03_train_only`; train 2206 groups, validation 552 groups.
  28 hashes связывают frozen inputs и восемь final-cache files.
- Полные параметры нормализованы к saved JSON representation: ngram tuple →
  list, dtype → name. `day05_n_features=5000` и cached replay `exact_match`
  явно отделены от ещё не выполненного fresh Day06 build/replay.
  Future build/save/load signatures зафиксированы в module docstring и SPEC;
  executable placeholders, model bundle и prediction API не добавлялись.
- Новый `tests/test_day06_reference.py`: 31 passed. Проверены read-only repeat,
  detached snapshots, missing/partial/checksum/protocol/runtime/policy errors,
  validation ID/order/truth semantics даже при обновлённых hashes,
  изменение input между readers и import без artifact I/O.
  Model fit/predict/transform/score, CV, final-run и network запрещены при
  проверке loader. Обучение было только в synthetic fixture setup тестов.
- Целевой offline regression: D6 reference + Day04 splits + Day05 analysis/final
  evaluation — 74 passed. Полный pytest suite на этом шаге не запускался.
  Type gate: 47 project files и 58 notebook modules (6 sequential exports +
  52 cell-boundary modules), 0 errors / 0 warnings; CLI Pyright не является
  полным запуском Pylance внутри VS Code. Compileall и diff check успешны.
- Отдельная real проверка: два повторных loader вызова с запрещёнными model/
  CV/network/final-run calls дали одинаковые metadata/tables. Все 48 configs/
  data (включая raw/interim)/Day02–Day05/annotation files SHA-256 неизменны
  после real audit и notebook execution; `models/day06`/`reports/day06` не созданы.
- `notebooks/day06_inference.ipynb`: D6-01 цель/теория/вызовы/реальные outputs/
  интерпретация/самопроверка; три code cells. Notebook выполнен сверху вниз
  из корня и `notebooks/` в отдельных чистых kernels, outputs сохранены.
  D6-02…D6-08 не начаты; commit не выполнялся.

### D6-02 — train-only восстановление Pipeline и validation replay

- [x] Воспроизвести замороженное решение отдельной явной build-функцией.
- Зависит от D6-01. Вход: проверенный reference; выход: fitted Pipeline
  в памяти и typed build result с train/features counts и replay evidence.
- Использовать прежний builder выбранного candidate; до fit сверить полные
  параметры с protocol и raw→clean parity на train/validation. Fit только на
  прежнем clean train в manifest order, включая retained duplicates.
- Exact validation replay сверяет все source IDs, true labels, порядок и
  pred IDs с Day04 reference. `ConvergenceWarning`/mismatch — ошибка.
  Никаких test transform/predict/score и refit на validation/test.
- Проверки: train-only vocabulary/IDF, порядок/duplicates, правильный builder,
  неверный clean text/config, replay mismatch и convergence failure.
  Структура build не передаёт test строки в model calls; fixtures содержат
  уникальные train/validation/test tokens для проверки boundary.
- Notebook: явный fit, shapes `(N_train, V)` / `(N_valid,)`, actual replay status;
  текущие ориентиры 2210 train rows, V=5000, 553 validation rows брать из данных.
  После D6-03 этот section будет читать metadata готового artifact без fit.
- Самопроверка: что доказывает validation replay и почему он не доказывает
  побитовую идентичность утраченных весов Day05?
- Промпт: «Выполни только D6-02 из PLAN.md по SPEC §11: явный train-only build
  и exact validation replay. Обнови notebook/PLAN, выполни общий checkpoint.
  Не сохраняй bundle, не меняй модель и не предсказывай test. Commit не делай».
- Свидетельство (2026-10-18): `build_inference_bundle(reference)` в
  `models/inference_artifact.py` заново проверяет artifacts и mutable caller
  snapshots до model calls, raw→clean parity на всём train/validation,
  selected/full effective parameters прежнего candidate builder.
  Input hashes/preprocessing implementation/runtime сверяются до fit
  и после replay; изменение vocabulary/IDF/classes при predict запрещено.
- На реальных данных выбранный `linear_svc` обучен только на прежнем clean
  train: 2210 rows / 2206 groups, четыре retained duplicate rows сверх первого,
  прежний manifest order. Instrumented fit/fit_transform подтвердили sparse
  TF-IDF `(2210, 5000)`; ровно один validation predict `(553,)` дал exact match
  всех 553 Day04 predictions. IDs/order/truth проверены до fit; classes `[0,1,2]`,
  ConvergenceWarning не было. Test transform/predict/score не выполнялись.
- Возврат — typed `InferenceBundle` в памяти; `BundleMetadata` содержит fresh
  replay, n_train=2210, n_validation=553, n_features=5000 и SHA-256 validation
  reference CSV. Сохраняются frozen train IDs/parameters/fit policy; новых
  selection, metrics, serialization или inference API нет.
- Новый `tests/test_day06_build.py`: 26 passed. Проверены точные train-only
  fit/TF-IDF calls и single validation replay; уникальные validation/test
  tokens fixtures не попали в vocabulary. Проверены mutation metadata/tables,
  raw-clean mismatch на обоих splits, builder mismatch до fit, replay IDs/shape,
  unknown labels, wrong fitted class order, ConvergenceWarning при fit/replay,
  vocabulary mutation, изменение frozen input во время fit и repeat state.
  Fixtures synthetic; real-data audit отделён от offline tests.
- Целевой regression: D6 reference/build и Day05 final evaluation — 76 passed.
  Полный pytest suite на этом пункте не запускался. Type gate: 48 project files
  и 60 notebook modules (6 sequential + 54 cell-boundary), 0 errors/warnings.
  CLI Pyright не является полным запуском Pylance внутри VS Code.
  Compileall и diff check успешны.
- Реальный audit запрещал test/CV/final-run/network/serialization calls;
  разрешены fit только exact train и transform/predict только exact validation.
  Все 48 configs/data (включая raw/interim)/Day02–Day05/annotation files SHA-256
  неизменны после audit и обоих notebook runs; `models/day06`/`reports/day06`
  не созданы. Fit при каждом D6-02 clean-kernel run явный по плану; после
  D6-03 он будет вынесен в отдельную build-команду.
- Notebook содержит отдельный D6-02 section с двумя новыми code cells;
  все пять cells выполнены из корня и `notebooks/` в отдельных чистых kernels,
  outputs сохранены. Fresh replay отделён от Day05 cached metadata и не
  объявлен доказательством идентичности утраченных весов или новой независимой
  quality evaluation. D6-03…D6-08 не начаты, commit не выполнялся.

### D6-03 — joblib bundle, manifest, load и reuse без fit

- [ ] Сохранить и загрузить целый fitted Pipeline с inference metadata.
- Зависит от D6-02. Вход: успешный build result; выход:
  `models/day06/sentiment_bundle.joblib` и соседний `manifest.json` по SPEC §11.
- В `models/inference_artifact.py` реализовать save/load и явную build-команду;
  её точный invocation записать в `docs/day06.md`. Bundle включает vectorizer
  и classifier, schema/config/classes/preprocessing/fit policy, provenance и
  versions; manifest содержит bundle checksum без self-checksum.
- Preflight до deserialize: schema/hash/policy/implementation/runtime;
  после load: fitted state/types/classes/dimensions/effective params/metadata.
  Roundtrip validation replay обязателен до завершения save. Только trusted
  local joblib; checksum не является доказательством доверия к pickle.
- Manifest писать последним. Готовая согласованная пара используется без fit;
  partial/corrupt/mismatch output блокирует rebuild/overwrite. Для нового build
  можно явно задать новый каталог. Не обещать атомарность пары файлов.
- Проверки: unfitted/wrong estimator, schema/hash/version/policy mismatch,
  classes/dimensions/config, повреждённый bundle, partial output, metadata
  disagreement; повтор build с готовым bundle не вызывает fit.
- Notebook: вызвать read-only load, показать metadata/shapes и replay status.
  Убрать прямой fit из предыдущего D6-02 section; build остаётся отдельной явной
  командой. Существующий valid bundle не перезаписывается при исполнении cells.
- Самопроверка: почему сохранение classifier отдельно от vectorizer опасно?
- Промпт: «Выполни только D6-03 из PLAN.md по SPEC §11: bundle, manifest,
  проверенный load и reuse без fit. Вынеси build в явную команду, обнови
  notebook/PLAN/docs и выполни общий checkpoint. Не оценивай test, commit не делай».
- Свидетельство: pending.

### D6-04 — batch prediction для raw unlabeled texts

- [ ] Реализовать typed `predict_texts` в `models/predict.py`.
- Зависит от D6-03. Вход: loaded bundle и `Sequence[str]`; выход:
  `list[SentimentPrediction]` с text/text_clean/pred_label_id/pred_label.
  Один текст передаётся как `[text]`; ровно N результатов в исходном порядке.
- Заменить старую заготовку вызовом общей `preprocess_text` и fitted Pipeline;
  не применять табличные функции, `.astype(str)` или обработку true labels.
  Старый CSV adapter временно направить на новый API либо оставить явную
  диагностируемую границу до D6-06; не оставлять второй расходящийся predictor.
- Валидировать весь batch до predict: reject bare str/bytes, empty batch,
  None/NaN/non-string, blank и empty-after-clean text с индексом. Duplicates
  сохранять. Unknown-vocabulary nonempty text допускается без refit.
- Проверки: N/order/mapping/duplicates, ошибки ввода, negation/numbers/currency/
  HTML parity; fit/fit_transform/partial_fit запрещены. Snapshot vocabulary,
  IDF, weights/classes до/после; нет probability/score у LinearSVC API.
- Notebook: raw→clean→prediction, shapes, один текст и batch, понятный invalid
  input example. Это демонстрация API, не измерение accuracy.
- Самопроверка: что произойдёт, если все слова новой фразы отсутствуют в vocabulary?
- Промпт: «Выполни только D6-04 из PLAN.md по SPEC §11: typed raw-text batch
  inference и input validation. Обнови notebook/PLAN и выполни общий checkpoint.
  Используй готовый bundle без fit/test prediction. Commit не делай».
- Свидетельство: pending.

### D6-05 — сквозная consistency и standalone inference

- [ ] Доказать согласованность preprocessing, serialization и public API.
- Зависит от D6-04. Использовать готовый bundle и прежний validation reference,
  без нового fit и без test prediction. Offline fixtures отдельно от real replay.
- Проверить все train/validation raw→clean значения; все validation predictions
  из loaded Pipeline и raw-text public API равны reference по IDs/order.
  Сравнение до/после save опирается на D6-03 roundtrip и проверку состояния.
- Запустить отдельный Python process: доступны src/runtime, bundle/manifest
  и новые тексты, недоступны training dataset/configs/reports. Запретить
  fit/CV/network/final-run и убедиться, что import не выполняет I/O/training.
- Выход: reusable consistency check и `reports/day06/consistency.json` с
  hashes/runtime/counts/status каждого фактически выполненного режима.
  Report не содержит собственного checksum, не является новой ML evaluation.
- Проверки: raw/clean mismatch, altered reference/artifact, state mutation,
  standalone без data/reports и повтор с теми же predictions. Не требовать
  одинаковых joblib bytes после независимого rebuild; не запускать rebuild
  ради сравнения сохранённого artifact.
- Notebook: вывод результатов consistency и объяснение serialization parity;
  real fit/test calls запрещены, bundle и предыдущие inputs неизменны.
- Самопроверка: почему успешный load в текущем kernel не доказывает standalone работу?
- Промпт: «Выполни только D6-05 из PLAN.md по SPEC §11: consistency report,
  real validation parity и standalone process без data/reports/fit. Обнови
  notebook/PLAN, выполни общий checkpoint. Test не предсказывай, commit не делай».
- Свидетельство: pending.

### D6-06 — CSV inference CLI и форматы файлов

- [ ] Довести `predict_for_file` и CLI в `models/predict.py` до SPEC §11.
- Зависит от D6-05. CLI принимает `--model`, `--input`, `--output` и вызывает
  те же load/predict API. Пути — `Path`, без скрытого поиска training artifacts.
- Вход UTF-8: обязательна только `text`, дополнительные columns сохраняются.
  Выход: прежние columns + `text_clean`, `pred_label_id`, `pred_label`, без index;
  N/order/duplicates/raw text сохранены. Не требовать labels/provider/headline.
- Ошибки: missing/duplicate columns, empty CSV/table, missing/blank text,
  collision output columns, input=output или уже существующий output.
  Не превращать литерал `NA` в пропуск. Весь ввод проверить до готового output;
  source CSV и bundle не изменять.
- Проверки: API/CSV equivalence, UTF-8/quotes/newlines/NA/duplicates и failure
  modes на tmp_path, CLI subprocess без fit/network, CSV roundtrip.
- Notebook: временный CSV с новыми unlabeled texts, вызов reusable adapter,
  чтение реального output и сверка с API. `docs/day06.md`: точная CLI команда,
  таблица input/output columns и примеры ошибок; final README не менять.
- Самопроверка: почему inference CSV не обязан иметь колонку sentiment?
- Промпт: «Выполни только D6-06 из PLAN.md по SPEC §11: CSV adapter/CLI
  поверх готового load/predict API и документация формата. Обнови notebook/PLAN,
  выполни общий checkpoint без fit/test evaluation. Commit не делай».
- Свидетельство: pending.

### D6-07 — новые examples и инструкция использования

- [ ] Подготовить небольшую воспроизводимую демонстрацию на новых текстах.
- Зависит от D6-06. Создать tracked `docs/day06_examples.csv` с вручную
  составленными финансовыми фразами, которых нет в canonical dataset;
  не выбирать примеры из train/validation/test. Достаточно 6–8 разных ситуаций.
- Включить рост, убыток, нейтральное событие, отрицание/сравнение,
  числа/валюты/HTML и неизвестную лексику. Не добавлять ground-truth labels,
  не требовать желаемых predictions/покрытия всех predicted классов.
- Прогнать API/CLI без fit; сохранить фактический `reports/day06/demo_predictions.csv`.
  Duplicates/invalid inputs показать отдельно как контракт, без quality claims.
  Вручную составленные smoke examples не заменяют real replay D6-05.
- Завершить `docs/day06.md`: явная build-команда отдельно от load/predict,
  зависимости/локальные artifacts, Python пример, CSV пример, preprocessing,
  ошибки/reuse, trusted joblib/version constraints, ограничения OOV и LinearSVC.
- Notebook: actual raw/clean/predicted table, интерпретация ограничений;
  отсутствие labels означает отсутствие accuracy/F1. Не править модель или
  подбирать фразы ради удачного ответа. Проверить docs commands на реальном bundle.
- Самопроверка: почему несколько правдоподобных ответов не доказывают качество
  на новом потоке финансовых новостей?
- Промпт: «Выполни только D6-07 из PLAN.md по SPEC §11: новые smoke examples,
  сохранённые predictions и docs/day06.md. Обнови notebook/PLAN, выполни общий
  checkpoint. Без fit, tuning, test prediction и final README. Commit не делай».
- Свидетельство: pending.

### D6-08 — итоговая приёмка и handoff Day07

- [ ] Подтвердить все критерии `classification_day06.txt` и SPEC §11.
- Зависит от D6-07. Полный `venv/bin/python -m pytest tests -v`, type gate,
  compileall/diff check; отдельно real bundle load/reuse, validation consistency,
  standalone process, API/CSV demos и clean-kernel notebook из двух cwd.
- Весь готовый notebook выполняется с запрещёнными fit/CV/download/final-run,
  без test transform/predict/score. Есть восемь sections, все переменные
  определены выше использования, реальные outputs и самопроверка сохранены.
- Проверить hashes Day02–Day05 inputs, frozen final outputs и bundle/manifest
  до/после; inference не меняет vocabulary/IDF/weights/classes. Не повторять
  real build лишь ради приёмки: использовать evidence D6-02/D6-03 и valid bundle.
- В PLAN/docs записать фактические команды, counts, runtime, проверки и
  ограничения. Не объявлять unrun checks успешными или replay доказательством
  идентичности утраченных весов Day05; не обещать рост метрик/live quality.
- Handoff Day07: build/load/CSV команды, artifact paths/versions/hashes,
  frozen config/fit policy, готовые examples и docs; Day07 завершит README,
  общую итоговую таблицу/ограничения и воспроизведение всего workflow.
- Самопроверка: чем artifact build, inference и итоговая ML evaluation
  отличаются по данным, разрешённым операциям и результату?
- Промпт: «Выполни только D6-08 из PLAN.md: приёмка Day06 по SPEC §11,
  полный checkpoint и handoff Day07. Не выполняй новый real fit/test prediction,
  не начинай Day07. Обнови notebook/PLAN/docs, commit не делай».
- Свидетельство: pending.

## Roadmap day07

| День | Подзадачи (все pending) | Результат / notebook |
| --- | --- | --- |
| 07 | D7-01 end-to-end review; D7-02 runnable README; D7-03 summary/limitations; D7-04 clean-environment acceptance | reproducible mini-product, day07_review.ipynb |

## Шаблон запроса для следующего чата

Day06: D6-01/D6-02 завершены; следующий исполнимый пункт — D6-03:

> Выполни только D6-03 из PLAN.md по SPEC §11 и handoff в docs/day05.md.
> Реализуй joblib bundle, manifest, проверенный load и reuse без fit.
> Вынеси build в явную команду; notebook должен читать готовый bundle без fit.
> Выполни общий checkpoint и запиши фактическое свидетельство в PLAN.
> Не меняй модель и не выполняй test transform/predict/score.
> Покажи diff stat с untracked files, предложи commit message, не делай commit.

При запросе одного ID выполнять только его. При запросе целого будущего дня
сначала детализировать roadmap до реализации; не добавлять опережающие features.
