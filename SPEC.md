# SPEC — Financial news sentiment classification

## 1. Назначение и scope

Учебный single-task mini-product для классификации англоязычных финансовых
предложений. Цель: понять ML workflow и научиться вести агентскую разработку
через явные контракты, небольшие изменения, проверки и воспроизводимые notebooks.

Путь: data → EDA → preprocessing → baseline → experiments → error analysis → inference.
Day01 и Day02 завершены; **day03** детализирован в §8, реализация идёт по PLAN.
Transformer fine-tuning, API, UI, deployment,
торговые стратегии и SOTA не входят в обязательный scope.

## 2. Источники требований и решения

Прочитаны README, overall, все семь day-файлов, requirements, baseline config,
все стартовые модули `src/finnews_sentiment/` и placeholder notebooks.
`venv/` и Git metadata — служебное окружение, не исходники задания.

Решения, устраняющие расхождения исходных примеров:

| Источник/проблема | Контракт проекта |
| --- | --- |
| Loader требует `headline` | Минимальная схема `text`, `sentiment`; headline необязателен |
| EDA показывает subject/provider | Day01 показывает классы и длины; дополнительные поля необязательны |
| README fit TF-IDF до split | Сначала split; fit transformations только на train |
| Day03 называет оценку test, Day04 выбирает улучшения | Day03 даёт baseline на validation; held-out test не участвует в выборе |
| README требует +5%, day04 требует попытки | Минимум три обоснованных эксперимента; прирост — исследовательская цель, не гарантированный gate |
| NLTK читается при import; тикеры заменяются после lower | Это известные дефекты шаблона day02; исправлять в day02, не использовать в day01 |
| Стартовый train/inference/config | Учебные заготовки, пока не приняты и не проверены как рабочий workflow |

Улучшение показывается и как абсолютная разница macro F1, и как относительное
`(candidate - baseline) / baseline * 100%`, если baseline > 0.

## 3. Данные и provenance

Пользователь выбрал подготовку открытого датасета. Используем
[takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank),
подмножество `sentences_75agree`: согласие разметчиков не менее 75%.
Это один вариант данных; четыре agreement subsets перекрываются, объединять их нельзя.

- Ожидается 3453 предложения, но фактический размер проверяется загрузчиком.
- Источник: архив `FinancialPhraseBank-v1.0.zip`, member `Sentences_75Agree.txt`.
- Archive revision и SHA-256 фиксируются в `configs/dataset.json`.
- Исходный текст читается как ISO-8859-1, разделитель метки — последний `@`.
- CSV адаптер сохраняет UTF-8, порядок и содержание предложений; это conversion,
  не NLP preprocessing. Метки сохраняются строками.
- Лицензия данных: CC-BY-NC-SA-3.0, авторство Malo et al. (2014),
  *Good debt or bad debt: Detecting semantic orientations in economic texts*.
  Архив и производный CSV не включаются в Git; инструкция воспроизводит загрузку.
- `data/raw/` — локальный immutable archive; `data/interim/financial_phrasebank_75agree.csv`
  — каноническое представление, ещё не очищенное для ML.

## 4. Контракт day01

### Вход и загрузка

`load_raw_news(Path) -> DataFrame` читает CSV UTF-8 и требует `text`, `sentiment`.
Отсутствующий файл, пустой dataset, отсутствующие колонки, неизвестные непустые
метки и нестроковые непустые тексты — явные ошибки. Пропуски и whitespace-only
тексты разрешены для диагностики, не удаляются загрузчиком.
Порядок классов всегда `negative`, `neutral`, `positive`.

### Анализ

EDA не изменяет входную таблицу. Отчёт включает:

- shape, первые строки, dtypes, missing по каждой колонке;
- blank texts отдельно от missing; количество дубликатов сверх первого;
- повторяющиеся тексты и тексты с противоречащими метками;
- counts и доли каждого класса, включая нулевой count; missing labels отдельно;
- длины исходного текста в символах и whitespace-separated словах;
  пропуски остаются пропусками, word count не называется tokenizer tokens;
- describe длин, примеры из каждого класса, графики классов и длин;
- наблюдения на реальных данных с ограничениями интерпретации.

Day01 ничего не обучает, не создаёт numeric labels, не чистит HTML/стоп-слова,
не удаляет строки. EDA всей таблицы предназначен для структурной проверки;
выбор preprocessing/model позже основывается на train, не на итоговом test.

### Выход и приёмка

- `notebooks/day01_eda.ipynb`: разделы D1-00…D1-05 с кодом и сохранёнными outputs.
- `reports/day01/`: `quality.json`, `class_distribution.csv`, `text_lengths.csv`,
  `sentiment_distribution.png`, `text_length_distribution.png`, `observations.md`,
  `environment.json` с фактически использованными версиями.
- Notebook исполняется сверху вниз в чистом kernel на подготовленных данных offline.
- Unit tests проверяют реальные failure modes на небольших fixtures.
- Обязательный type gate: `venv/bin/python scripts/check_types.py` проверяет Python
  sources и экспортированные code cells notebooks через Pyright + pandas-stubs;
  type errors исправляются до завершения шага, без отключения диагностики.
- Прогон реального источника отдельно подтверждает размер, schema и checksum.
- Статусы PLAN ссылаются на реальные результаты; инфраструктурные ограничения явны.

## 5. Архитектура

```text
configs/dataset.json                    # pinned source/labels
src/finnews_sentiment/
  data/load_data.py                     # canonical CSV contract
  data/prepare_phrasebank.py            # explicit source conversion/download
  visualization/eda.py                  # pure summaries and figures
  features/preprocess.py                # day02 template
  models/train_model.py, predict.py     # day03/day06 templates
notebooks/day01_eda.ipynb                # executable teaching narrative
tests/test_day01.py                     # offline contract checks
reports/day01/                          # reproducible local outputs
requirements-day01.txt                  # verified day01 versions
requirements.txt                       # original broad curriculum dependencies
```

Нет необходимости создавать общую config framework. Dataclasses и small functions
допустимы, если делают контракт понятнее. Notebook не дублирует бизнес-логику.

## 6. Договорённости для следующих дней

- Day02: минимальная воспроизводимая очистка; сохранять отрицания, числа и финансовые
  символы до проверки гипотез; зафиксировать label mapping и duplicate policy.
- Day03: разделить уникальные text groups, seed=42; outer test ≈20%, validation ≈20%
  от outer-train; держать source row IDs. Баланс классов и размеры проверить.
  Если групп/классов недостаточно — диагностировать, не молча отключать stratification.
- Baseline: TF-IDF (max_features=5000, ngram_range=(1, 2)) + Logistic Regression.
  DummyClassifier даёт ориентир. Sparse representation сохраняется.
- Метрика: macro F1, дополнительно accuracy и precision/recall/F1/support по классам.
- Day04: минимум три эксперимента на одинаковых validation folds; подбор через Pipeline.
- Day05: анализ ошибок выбранной модели на validation. Финальный test после заморозки
  модели/параметров; анализ test допускается для итогового отчёта без дальнейшей настройки.
- Day06: один bundle включает preprocessing contract, vectorizer, classifier, class order,
  provenance, versions и config. Inference не делает fit и совпадает с training transform.
  Вероятности выдаются только если модель поддерживает их; score не называть probability.
- Day07: runnable README, итоговая таблица, demo, ограничения и полный повторный запуск.

Точные API последующих дней уточняются перед реализацией. Тональность предложения
не является прогнозом доходности; историческая выборка не доказывает работу на live news.

## 7. Контракт day02 — минимальный preprocessing (policy `minimal-v1`)

### Границы D2-01 и поток данных

D2-01 фиксирует контракт и загружает локальный canonical CSV в notebook.
Реализация очистки начинается в D2-02, обработка таблицы — в D2-03,
метки/дубликаты — в D2-04, сохранение артефактов — в D2-05.
До этих шагов `features/preprocess.py` остаётся непринятой заготовкой;
Day02 notebook её не импортирует, в том числе из-за NLTK import side effects.

Будущий поток: canonical DataFrame `(N, 2)` → валидация → очистка каждого
валидного текста → явное исключение непригодных строк → метки и диагностика
дубликатов → processed DataFrame `(M, 5)`, где `M <= N`.
Это детерминированное преобразование без `fit`, vocabulary или learned state.
TF-IDF, дополнительные числовые features, split и обучение сюда не входят.
Policy фиксируется до оценки модели; её эффективность пока не измеряется.

### Очистка одного текста

Публичный контракт: `preprocess_text(text: str) -> str`.
Неверный тип, включая `None`, вызывает информативный `TypeError`; пустая строка
и строка, пустеющая после очистки, возвращают `""`. Функция не решает, удалять ли
строку dataset. Она не читает файлы/NLTK resources, не скачивает данные,
не обучает transformations и не имеет import side effects.

Порядок операций `minimal-v1`:

1. Декодировать стандартные HTML entities (`&amp;`, `&nbsp;`, числовые entities).
   Вложенные encoding layers раскрывать до стабильного результата, чтобы
   повторный вызов не менял текст. Неизвестные entities сохранять.
2. Заменить синтаксически корректные HTML tags пробелом, чтобы соседние слова
   не склеивались. Содержимое элементов сохранять; это обработка текстовых
   предложений, а не извлечение статьи из произвольной web page. HTML comments
   заменить пробелом. Обычные сравнения `profit < 5` и `growth > 3` сохранять;
   незавершённый markup не удалять молча.
3. Применить `str.lower()`; не заменять uppercase abbreviations/tickers.
4. Свести последовательности Unicode whitespace, включая tabs/newlines/NBSP,
   к одному ASCII пробелу и убрать пробелы по краям.

Повторять полный проход до стабильного результата: lowercase может превратить
неизвестное написание entity (например, `&AmP;`) в распознаваемое `&amp;`.
Это часть policy, необходимая для идемпотентности, а не обучение на данных.

Сохраняются слова и отрицания (`not`, `no`, `never`, `n't`), числа, знаки чисел,
десятичные разделители, `%`, валюты (`$`, `€`, `£`), пунктуация и апострофы.
Содержание тикера сохраняется, регистр нормализуется. Не выполняются stopword
removal, lemmatization/stemming, удаление цифр, punctuation stripping,
исправление опечаток, Unicode compatibility normalization или tokenizer fit.
Ожидаемая идемпотентность: `preprocess_text(preprocess_text(s)) == preprocess_text(s)`.
Детерминизм означает одинаковый результат при той же policy и runtime.

Tradeoffs: lowercase теряет информацию о регистре; удаление markup может принять
запись вида `<USD>` за tag. Минимальная policy не обещает улучшения macro F1.
Извлечение текста из script/style и полноценный HTML scraping не входят в scope.
Примеры ожидаемого поведения (контракт, пока не измеренный результат):
`"  Profit is NOT down -5% &amp; costs £2.50.  "` →
`"profit is not down -5% & costs £2.50."`;
`"<b>No</b> growth"` → `"no growth"`.

### Таблица, ошибки и исключения

Вход требует `text`, `sentiment` по контракту canonical loader (§4).
Исходная таблица не изменяется. Будущий processed результат содержит пять колонок:

| Колонка | Контракт |
| --- | --- |
| `source_row_id` | Integer, 0-based позиция строки в исходном canonical CSV, до фильтрации; не pandas index |
| `text` | Исходная строка без изменения |
| `sentiment` | Исходная метка `negative`, `neutral` или `positive` |
| `text_clean` | Непустой результат `minimal-v1` |
| `label_id` | Integer: `0=negative`, `1=neutral`, `2=positive`; добавляется в D2-04 |

Фиксированный порядок колонок — как в таблице. Source IDs уникальны и сохраняют
относительный порядок; после исключений возможны пропуски в нумерации.
Дополнительные входные поля не входят в canonical processed схему.
D2-03 выдаёт первые четыре колонки; финальные пять доступны после D2-04.

Отсутствующий файл, пустой входной dataset, отсутствующие обязательные колонки,
нестроковое непустое значение text/sentiment и неизвестный непустой класс —
ошибки контракта. Их не превращать в исключённые строки или fallback labels.
Загрузчик уже выполняет часть этой валидации; табличный preprocessing также
проверяет вход, поскольку может вызываться напрямую с DataFrame.

Для допустимых, но непригодных строк записывать все применимые причины:
`missing_text`, `missing_sentiment`, `blank_text` (исходный whitespace-only),
`empty_after_cleaning` (исходный непустой текст стал пустым).
`blank_text` и `empty_after_cleaning` взаимоисключающие; missing не преобразуется
в `"nan"`/`"None"`. Очистка вызывается только для непустого строкового текста.
Каждая исключённая строка учитывается один раз с source ID и списком причин;
counts по отдельным причинам могут пересекаться и не обязаны суммироваться в
число исключённых. Всегда `N_input = N_retained + N_excluded`.
Если исключены все строки — явная ошибка с диагностикой, не готовый ML dataset.

Отчёт показывает counts до/после по классам, missing labels отдельно, source IDs
и причины исключений, изменения текстов и число строк. Raw/interim неизменны;
processed CSV и reports записываются только явным действием в D2-05.
Дубликаты сохраняются; конфликты и новые совпадения после cleanup проверяются
в D2-04 без автоматического выбора метки. Day03 должен исключить пересечение
групп совпадающих raw/clean texts между splits.

API D2-03: `preprocess_news(df: DataFrame) -> tuple[DataFrame, PreprocessingReport]`.
Вызывать на canonical CSV в исходном порядке: функция не может восстановить
позиции уже переставленных/выбранных строк. Повторяющиеся имена колонок запрещены;
произвольный, в том числе неуникальный pandas index допустим и не используется как ID.
`PreprocessingReport` содержит policy, `n_input`, `n_retained`, `n_excluded`,
`excluded_rows` (source ID + tuple всех причин), `reason_counts`,
`class_counts_before`, `class_counts_after`, `missing_sentiment_before` и
`changed_source_row_ids` — IDs сохранённых строк, где `text_clean != text`.
Все причины представлены в counts, включая нулевые. Counts классов имеют
порядок `negative`, `neutral`, `positive`; missing labels учитываются отдельно.
При полностью исключённом входе `EmptyProcessedDatasetError(ValueError)`
сохраняет диагностику в `.report`. `dataclasses.asdict(report)` даёт структуру
для будущего сохранения отчётов, без записи файлов на этом шаге.
Прежний `clean_news_data` делегирует обработку этому API и возвращает только
таблицу; при исключениях выдаёт warning с counts и рекомендацией получить полный
report. Для нового кода использовать `preprocess_news`. `add_basic_features`
остаётся заготовкой и в Day02 notebook не вызывается.

### Метки и duplicate/conflict policy (D2-04)

`features/labels_duplicates.py` содержит фиксированный mapping из CLASS_ORDER:
`negative=0`, `neutral=1`, `positive=2`. Это category IDs, не ordinal scores.
`add_label_ids(df) -> DataFrame` возвращает копию с integer `label_id`, сохраняет
строковый `sentiment` и порядок; unknown/missing labels — ошибки без fallback.
Mapping не зависит от порядка строк и набора присутствующих классов.

`analyze_duplicates(processed_df) -> DuplicateReport` проверяет четыре колонки
D2-03, непустые строковые тексты, классы и уникальные nonnegative integer source IDs.
Отчёт содержит `raw_groups` и `clean_groups`: только группы размера >=2,
в порядке первого появления. В `DuplicateGroup` сохраняются ключ `text`,
source IDs и соответствующие им метки в порядке входа. `conflicting` означает
наличие нескольких меток. `new_after_cleaning` в clean group означает, что
несколько разных raw texts превратились в одинаковый clean text; группа может
содержать также уже существовавшие raw duplicates.
`raw_duplicate_rows`/`clean_duplicate_rows` считают строки сверх первой в каждой
группе, независимо от метки; raw/clean counts нельзя складывать — они пересекаются.
Диагностика не меняет данные и доступна даже при conflicts.

`prepare_labeled_news(processed_df) -> tuple[DataFrame, DuplicateReport]`
добавляет `label_id` к результату D2-03, сохраняя все строки, включая одинаковые
тексты с одной меткой. При raw или clean conflicts вызывает
`DuplicateLabelConflictError(ValueError)` с полным отчётом в `.report`.
Никакого majority vote, удаления конфликтной группы или выбора первой метки.
Будущая команда финальной подготовки D2-05 должна использовать этот API,
чтобы unresolved conflicts блокировали сохранение финального dataset.

Policy совпадений определяется точным равенством raw/clean strings, без fuzzy
matching. В Day03 вся группа совпадающих raw или clean texts должна оказаться
в одном split. Для результатов `minimal-v1` одинаковые raw texts имеют одинаковый
clean text, поэтому clean groups включают raw groups. Реализация split в Day02
отсутствует; сохранение repeats может влиять на веса примеров, это ограничение
текущей policy, а не скрытая балансировка классов.

### Артефакты и offline подготовка (D2-05)

`data.prepare_day02.prepare_day02(root: Path) -> Day02Artifacts` проверяет локальный
архив по pinned checksum, сверяет canonical CSV с конвертацией member и expected
row count, затем вызывает `preprocess_news` и `prepare_labeled_news`.
Download отсутствует; при отсутствии архива/CSV требуется отдельная подготовка
источника. Schema/source/conflict errors возникают до записи outputs.

Явная команда из корня:
`PYTHONPATH=src venv/bin/python -m finnews_sentiment.data.prepare_day02`.
CLI также принимает `--root`; default root определяется по расположению модуля.
Raw/interim/config читаются, не перезаписываются.

Выходы (локальные, исключены из Git существующим `.gitignore`):

- `data/processed/financial_phrasebank_75agree.csv`: UTF-8, без pandas index,
  пять колонок §7 в фиксированном порядке, integer source IDs/label IDs,
  сохранённые исходные тексты/метки, строковый `text_clean`.
- `reports/day02/preprocessing.json`: все поля PreprocessingReport, включая
  исключённые source IDs/причины, class counts и changed retained IDs.
- `reports/day02/duplicates.json`: raw/clean groups с source IDs и метками,
  counts повторов сверх первого, новых clean groups и conflicts.
- `reports/day02/provenance.json`: источник/revision/subset/license/citation,
  archive/member/encoding, checksums archive/canonical/config/processed CSV,
  относительные пути, policy version, duplicate/ID policy, class order/mapping,
  output schema/row count и фактические Python/pandas/NumPy versions.

CSV roundtrip до записи проверяется с явными dtypes для текстовых/числовых полей
и `keep_default_na=False`: строки `NA`, `nan`, numeric-only text не должны
менять смысл при чтении. При неизменных входах, policy и runtime повторный
запуск даёт одинаковые bytes всех четырёх outputs (без текущего timestamp).
Перед записью проверяется неизменность прочитанных raw/interim bytes.
Запись четырёх файлов не является единой транзакцией; I/O failure во время
записи может потребовать повторного запуска. Training, splits и inference
на этом шаге не выполняются. Команда/инструкции сохраняются в Git.

### Учебный checkpoint D2-01

`notebooks/day02_preprocessing.ipynb` содержит root/src setup, загрузку реального
локального CSV без download, shape/head, порядок классов, policy и будущую схему,
интерпретацию и вопросы самопроверки. Контрольная загрузка и checksum подтверждают
неизменность таблицы и файла. Code cells выполняются в чистом venv kernel из
корня и `notebooks/`; type gate проверяет их вместе с Python sources.
Завершение D2-01 не означает, что очистка или processed dataset уже реализованы.

## 8. Контракт day03 — воспроизводимый baseline (`baseline-v1`)

### Цель, границы и порядок реализации

Day03 строит первую рабочую модель и сохраняет точку отсчёта для Day04.
`classification_day03.txt` требует train/test и test metrics. Применяется уже
согласованное решение §2: на Day03 выделяются train/validation/held-out test,
baseline оценивается на validation, финальный test — только после заморозки
модели/параметров в D5-04. Validation metrics не называются test metrics.
Структурная проверка размеров/классов test разрешена; его тексты не используются
для fit, transform, predict, score, ручного анализа ошибок или выбора настроек.

Поток данных: processed DataFrame `(N, 5)` → группы совпадающих текстов →
train/validation/test → train-only TF-IDF `(N_train, V)` → LogReg → validation
predictions `(N_valid,)` → metrics и сохранённая baseline reference.
DummyClassifier обучается на тех же train labels и оценивается на тех же
validation rows. Предсказания — category IDs, не probabilities или sentiment scores.

Пункты PLAN: D3-01 runtime/input/config; D3-02 splits; D3-03 unfitted estimators;
D3-04 fit/predict; D3-05 evaluation; D3-06 artifacts/runner; D3-07 приёмка.
Не входят: новые preprocessing transformations, additional features, balancing,
tuning/CV, другие классификаторы, test evaluation, error analysis и inference bundle.
Стартовые train/predict — непринятые заготовки; новая реализация не должна
использовать их скрытый row split, dropna, astype(str), повторную очистку
или add_basic_features. `models/predict.py` остаётся вне scope Day03.

### Runtime, вход и baseline config (D3-01)

Работать в существующем `venv/`, без пересоздания и обновления всего stack.
`requirements-day03.txt` включает Day01 runtime и минимальные проверенные
дополнения для Day03 (scikit-learn и необходимые зависимости); фактические
pins выбираются после проверки совместимости текущего Python/runtime.
Future packages из исходного `requirements.txt` не устанавливаются автоматически.
Импорт modules не читает dataset/resources, не скачивает и не обучает модель.

Вход: `data/processed/financial_phrasebank_75agree.csv` и
`reports/day02/provenance.json` по §7. Loader читает UTF-8 с явными текстовыми
и integer dtypes, `keep_default_na=False`, затем проверяет:

- ровно пять колонок и их порядок §7; непустой dataset, непустые строковые
  text/text_clean, известные sentiments; все три класса представлены;
- unique nonnegative integer source IDs в возрастающем исходном порядке,
  integer label IDs и точное соответствие `negative=0, neutral=1, positive=2`;
- policy `minimal-v1`, class order/mapping и schema/rows в provenance;
  SHA-256 processed CSV совпадает с Day02 `output.sha256`, источник и revision
  согласованы с `configs/dataset.json`; некорректный provenance — ошибка;
- raw/clean duplicates не конфликтуют по меткам; одинаковый raw text имеет
  одинаковый text_clean, иначе подготовка не согласована с policy.

Missing files/schema/types/checksums, пустые тексты, unknown labels и conflicts
вызывают информативную ошибку до split/fit. Никакого удаления строк/дубликатов,
fallback mapping, преобразования missing в строки или изменения входных файлов.
Day03 использует проверенный результат Day02, не запускает очистку повторно.
При отсутствии входа пользователь запускает отдельную offline команду §7.

`configs/train_baseline.json` становится реальным контрактом runner, а не
неиспользуемым шаблоном: `input_path`, `provenance_path`, `text_col=text_clean`,
`target_col=label_id`, `random_state=42`, `test_size=0.2`, `validation_size=0.2`
(доля outer-train), TF-IDF/LogReg/Dummy параметры и output directories ниже.
Все пути относительно корня проекта, числовые значения
и обязательные keys валидируются явно. Прежний model_path не требует записи
joblib на Day03. Effective config сохраняет также используемые defaults.

API D3-01: `data.baseline_config.load_baseline_config(path: Path) -> BaselineConfig`
и `data.load_processed.load_processed_news(csv_path: Path, provenance_path: Path,
dataset_config_path: Path) -> DataFrame`. `BaselineConfig` — typed dataclass:
пути — относительные `Path`, настройки estimators — TypedDict; JSON ngram array
становится tuple. Это проверка config, не создание Pipeline. Fixed policy/settings
и schema keys проверяются явно, числовые настройки — на тип/диапазон/finite,
seed — nonnegative integer в допустимом диапазоне и одинаковый для LogReg.
`validate_processed_news(df: DataFrame) -> None` проверяет таблицу без изменения
её значений/index/порядка; pandas index не используется как source ID.

Runtime D3-01: Python 3.14.6, scikit-learn 1.9.1, SciPy 1.18.1, joblib 1.6.0,
threadpoolctl 3.7.0; существующие Day01 pins сохраняются в requirements-day03.txt.
В закреплённой версии LogReg L2 задаётся `l1_ratio=0.0` (deprecated `penalty`
не передаётся). Проверка runtime — импорт и согласованность зависимостей, без fit.

Loader сверяет bytes processed CSV и dataset config с provenance, источник
со всеми полями dataset config, metadata schema/rows/path, class order/mapping,
duplicate/ID policy и обязательные versions. Archive hash сверяется с config;
canonical hash проверяется на корректный формат. Raw/interim bytes loader
не перечитывает: связь с ними основана на проверенном Day02 provenance, это
не повторная проверка содержимого архива. CSV records имеют ровно пять полей;
заголовок проверяется до pandas, чтобы duplicate columns не переименовывались
молча. Integer поля сначала читаются как строки с keep_default_na=False,
проверяются на integer literals и затем явно приводятся к int64: float/boolean
IDs и missing не должны усекаться или маскироваться. После этого проверяются
все table invariants и верхняя граница source IDs относительно source row count.

### Group-aware splits (D3-02)

Одна группа — точное совпадение `text_clean`; при `minimal-v1` она включает
совпадающие raw texts. Ни fuzzy matching, ни deduplication не выполняются.
group ID = минимальный `source_row_id` в группе; группы сначала сортируются по
group ID, каждая имеет ровно один label ID. Ошибки raw → clean и conflicts
блокируют split, не решаются выбором первой/большинства меток.

Алгоритм `group-stratified-v1`:

1. На уровне уникальных групп вызвать stratified train_test_split с
   test_size=0.2, random_state=42, stratify по единственной метке группы.
2. Оставшиеся outer-train groups снова упорядочить по group ID; тем же способом
   выделить validation_size=0.2 от outer-train, random_state=42.
3. Распространить назначение группы на все её строки. Manifest имеет одну строку
   на retained source ID в исходном порядке; таблицы splits сохраняют этот порядок.

Номинальные доли групп train/validation/test ≈64%/16%/20%. Rounding и разные
размеры duplicate groups дают отличия в долях строк; не выдавать эти числа за
точные размеры. Stratification сохраняет распределение классов приблизительно
по группам; фактический баланс строк тоже показывается, без oversampling/weights.

Инварианты: все source IDs покрыты ровно один раз; splits попарно disjoint по
source IDs, raw texts и clean texts; каждая группа целиком в одном split;
в каждом split есть три класса. Недостаточная возможность stratification или
отсутствующий класс после split — ошибка с counts строк/групп по классам;
не отключать stratification, не менять seed поиском удобного split.
Одинаковые данные/config/runtime дают одинаковое назначение. Manifest фиксируется
в D3-06 и повторно используется Day04/Day05, без нового случайного разделения.

### Estimators и fit boundary (D3-03…D3-04)

`models/train_model.py` создаёт `sklearn.pipeline.Pipeline` с шагами
`tfidf` (TfidfVectorizer) и `clf` (LogisticRegression). Каждый builder возвращает
новый unfitted estimator; D3-03 только строит его, D3-04 впервые вызывает fit.
Параметры baseline фиксируются до просмотра validation metrics:

| Компонент | Baseline settings |
| --- | --- |
| TF-IDF | max_features=5000, ngram_range=(1, 2), analyzer=word, lowercase=False (Day02 уже lower), stop_words=None, min_df=1, max_df=1.0 |
| TF-IDF weighting | use_idf=True, smooth_idf=True, sublinear_tf=False, norm=l2; стандартный word token_pattern, strip_accents=None |
| LogReg | solver=lbfgs, C=1.0, L2 regularization в API закреплённой версии, max_iter=1000, tol=1e-4, class_weight=None, random_state=42 |
| Dummy | strategy=most_frequent; обучается на train label frequencies, без доступа к validation labels |

Стандартный word tokenizer не сохраняет всю финансовую пунктуацию как features,
игнорирует односимвольные tokens и может разбивать contractions. Сохранение `%`,
валют и отрицаний при preprocessing не гарантирует их отдельного представления
в TF-IDF. Это явно объясняется как ограничение baseline; custom tokenizer и
изменение representation — будущая гипотеза, не автоматическое расширение Day03.

Vocabulary и IDF fit только по train.text_clean; LogisticRegression fit по train
features/label IDs. Матрица остаётся sparse, `(N_train, V)`, `V <= 5000`;
`max_features` — верхняя граница, не обещание ровно 5000 столбцов.
Validation проходит только через fitted Pipeline.predict/transform; vocabulary
и IDF после predict не меняются. Если возникнут CV/experiments в Day04,
TF-IDF должен fit внутри Pipeline каждого fold, с group-aware folds.

Dummy получает constant feature array `(N_train, 1)`, predicts `(N_valid,)`.
Это reference для majority-class поведения, не дополнительный обучаемый feature.
Training API принимает готовые splits, не создаёт новые, не молча исключает строки
и не смешивает training с evaluation. До fit проверяет schema, class coverage,
group/source disjointness; обученные classes_ соответствуют `[0, 1, 2]`.
Validation outputs сохраняют точную привязку к source IDs/порядку строк.

Empty vocabulary, неверный config, malformed splits и nonconvergence — явные
диагностики. ConvergenceWarning не скрывается: прежде чем закрыть пункт,
разобраться с причиной и зафиксировать необходимую коррекцию solver/runtime
settings, без выбора качества по validation/test. Score improvement не является
критерием приёмки, сравнение с Dummy не гарантирует победу baseline.

### Оценка на validation (D3-05)

`models/evaluate.py` принимает aligned true/predicted label IDs; входы непусты,
одной длины, принадлежат `[0, 1, 2]`. Основная метрика — macro F1 с явными
labels=[0, 1, 2]; дополнительно accuracy и precision/recall/F1/support каждого
класса в порядке `negative`, `neutral`, `positive`, zero_division=0 явно.
Support sums=N_valid. Отсутствие predicted класса даёт нулевые значения по
принятой policy и обсуждается, не исчезает из отчёта.

Baseline и Dummy оцениваются на одной последовательности validation source IDs.
Macro F1 даёт классам одинаковый вес; accuracy учитывает каждую строку и может
скрыть слабость на редких классах. Повторяющиеся строки сохранены, поэтому
row metrics взвешивают группы по их размеру. Один split не даёт оценки
статистической устойчивости или гарантии качества на live financial news.
Confusion matrix `(3, 3)` и ручной разбор ошибок остаются Day05 на validation;
на Day03 не выполнять test metrics или настройку по найденным ошибкам.

### Сохранение результатов и воспроизведение (D3-06)

Единый явный runner читает config/проверенные данные, вызывает reusable функции
split/build/train/predict/evaluate и сохраняет результаты. Notebook вызывает эти
же функции/runner. CLI из корня:
`PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.train_model --config configs/train_baseline.json`.
Offline выполнение требует локальных Day02 artifacts; download отсутствует.

| Артефакт | Содержимое |
| --- | --- |
| `data/processed/day03/splits.csv` | source_row_id, group_id, split (`train`, `validation`, `test`), исходный порядок |
| `reports/day03/split_summary.json` | algorithm/seed/fractions, размеры строк/групп и class counts/доли каждого split, подтверждение disjointness |
| `reports/day03/validation_predictions.csv` | source_row_id, true_label_id, baseline_pred_label_id, dummy_pred_label_id в validation порядке; тексты не дублируются |
| `reports/day03/metrics.json` | evaluation_split=validation, N_valid, class order/mapping, baseline и Dummy metrics §8 |
| `reports/day03/config.json` | effective config с параметрами estimators и splits, policy baseline-v1 |
| `reports/day03/provenance.json` | source/revision/subset, Day02 policy и checksum/provenance, checksums config/processed CSV/manifest/predictions/metrics, seed/algorithm, vocabulary size, classes_, фактические Python/pandas/NumPy/SciPy/scikit-learn versions |

Все outputs локальные согласно `.gitignore`. Python modules в `src/.../models/`
должны оставаться в Git: существующее правило `models/` проверить и при
необходимости сузить до `/models/` уже в D3-03, не игнорировать новый код
в Git или type gate. Старый training CLI не является поддерживаемой командой
Day03; его замена новым runner выполняется в D3-06.
Baseline reference — сохранённые данные сравнения/протокол, а не Day06 inference
bundle. Joblib serialization и load/predict API на Day03 не обязательны.
Никаких test predictions, test metrics или обещанного прироста в артефактах.

До записи проверить согласованность manifest/predictions/metrics и неизменность
input bytes; CSV/JSON должны читаться обратно без изменения IDs/значений/порядка.
Повторный запуск в том же runtime воспроизводит splits, predictions и metrics;
не обещать идентичные binary model bytes между версиями/платформами. Для reports
не включать timestamp в сравниваемое содержимое. Несколько записей файлов не
являются общей транзакцией; I/O failure может потребовать повторного запуска.
В Day04 новые результаты записывать отдельно, сохраняя baseline reference.

### Notebook и приёмка (D3-07)

`notebooks/day03_baseline.ipynb` имеет по разделу D3-01…D3-07 с целью, теорией,
вызовом reusable кода, реальными outputs, интерпретацией и самопроверкой.
Root/src setup предшествует imports; все переменные создаются выше использования.
После каждого ID сохраняются outputs чистого kernel из корня и `notebooks/`
на локальных данных, без зависимости от предыдущей истории kernel или download.

Финальные `reports/day03/observations.md` и `docs/day03.md` описывают фактические
split sizes, feature shapes, baseline/Dummy validation metrics, параметры,
команды воспроизведения, границу test и ограничения. Рост macro F1, превосходство
над Dummy и quality на live news не гарантируются и не являются acceptance gate.

Приёмка: три непересекающихся group-aware splits; fit только на train; рабочие
Pipeline и Dummy; aligned predictions/validation metrics; сохранённая reference;
повторный offline run; type gate без ошибок и содержательные pytest tests;
real-data checks и clean-kernel notebook подтверждены отдельно. Ни один пункт
реализации не отмечается завершённым только на основании написанного кода.
