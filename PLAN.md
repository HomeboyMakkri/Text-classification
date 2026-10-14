# PLAN — staged implementation

## Порядок работы

Day01 завершён 2026-10-09. Day02 (D2-01…D2-06) завершён 2026-10-11.
Day03: D3-01…D3-06 завершены 2026-10-14; следующий пункт — D3-07.
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

## Roadmap day04–day07

| День | Подзадачи (все pending) | Результат / notebook |
| --- | --- | --- |
| 04 | D4-01 hypotheses/protocol; D4-02 n-grams; D4-03 LinearSVC; D4-04 hyperparameter CV; D4-05 comparison/selection | ≥3 controlled experiments, day04_experiments.ipynb |
| 05 | D5-01 aligned predictions; D5-02 confusion matrix; D5-03 failure categories; D5-04 frozen final test/report | error analysis, day05_error_analysis.ipynb |
| 06 | D6-01 artifact bundle; D6-02 load/predict API; D6-03 roundtrip consistency; D6-04 new-text demos | inference without fit, day06_inference.ipynb |
| 07 | D7-01 end-to-end review; D7-02 runnable README; D7-03 summary/limitations; D7-04 clean-environment acceptance | reproducible mini-product, day07_review.ipynb |

## Шаблон запроса для следующего чата

> Выполни только D3-03 из PLAN.md по SPEC §8 и AGENTS.md.
> Создай unfitted TF-IDF/LogReg Pipeline и DummyClassifier builders.
> Выполни общий checkpoint, обнови notebook/PLAN; не начинай D3-04 и не выполняй fit.
> Покажи diff stat, предложи commit message, не делай commit.

При работе целым днём замените ID на day03. Крупные решения фиксируем в SPEC;
вопросы пользователя можно обсуждать, не меняя scope реализации.

## Дополнение day01 — Pylance/Pyright quality gate

- [x] Добавить pinned Pyright и pandas-stubs, standard config для venv/src/tests/scripts.
- [x] Исправить ROOT: Path через find_project_root; scalar row selection через iloc;
  word count без неподдерживаемого в stubs list-string accessor.
- [x] Добавить scripts/check_types.py с проверкой code cells всех notebooks и настройки VS Code.
- Проверки: Pyright sources и notebook export — 0 errors, 0 warnings, 0 informations; 12/12 unit tests; compileall и diff check прошли. Исправленный notebook повторно выполнен в чистом kernel, outputs обновлены.

## Дополнение — pytest для новых тестов

- [x] Установить pytest в существующий `venv/` и закрепить `pytest==9.1.1`
  в `requirements-dev.txt` (2026-10-11).
- Все новые тесты писать с pytest; существующие unittest tests сохранять.
  Правило и основная команда запуска закреплены в AGENTS.md.
- Общая проверка suite: `venv/bin/python -m pytest tests -v`;
  исторические unittest свидетельства выше остаются фактическими результатами
  своих этапов. Для следующих этапов использовать pytest, включая D2-06.
- Свидетельство: pytest 9.1.1 — 26 passed, 46 subtests passed; существующие
  test files не изменены. `git diff --check` passed.
