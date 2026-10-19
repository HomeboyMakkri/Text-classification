# SPEC — Financial news sentiment classification

## 1. Назначение и scope

Учебный single-task mini-product для классификации англоязычных финансовых
предложений. Цель: понять ML workflow и научиться вести агентскую разработку
через явные контракты, небольшие изменения, проверки и воспроизводимые notebooks.

Путь: data → EDA → preprocessing → baseline → experiments → error analysis → inference.
Day01–Day05 завершены; **day06** детализирован в §11, D6-01…D6-06 выполнены.
Transformer fine-tuning, HTTP API, UI, deployment,
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
| Day06 просит сохранить модель и vectorizer | Сохранять один fitted TF-IDF/classifier Pipeline в joblib bundle; preprocessing выполнять общей `preprocess_text`, не старым табличным адаптером |

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
- Day04: минимум три контролируемых направления на одинаковых group-aware CV folds
  внутри Day03 train; подбор через полный Pipeline. CV выбирает вариант, а
  validation используется только для единственного подтверждения финального выбора.
- Day05: анализ ошибок выбранной модели на validation. Финальный test после заморозки
  модели/параметров; анализ test допускается для итогового отчёта без дальнейшей настройки.
- Day06: один bundle включает preprocessing contract, vectorizer, classifier, class order,
  provenance, versions и config. Inference не делает fit и совпадает с training transform.
  Точный контракт — §11; текущий API возвращает label ID/name без probability/score.
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

API D3-02: `data.split_data.split_processed_news(df: DataFrame, *,
test_size=0.2, validation_size=0.2, random_state=42) -> NewsSplits`.
`NewsSplits` содержит manifest `(N, 3)`, `splits` с ключами
`train`, `validation`, `test` и typed `SplitSummary`/`SplitStats`.
Summary фиксирует algorithm/seed/fractions, counts и доли строк/групп,
counts и доли классов с отдельными row/group denominators, disjointness
по source IDs/raw/clean. Вход и pandas index сохраняются; ID определяется
по source_row_id. I/O и fit отсутствуют; артефакты сохраняются в D3-06.

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

## 9. Контракт day04 — контролируемый выбор модели

### Цель, границы и данные

Day04 проверяет три заранее ограниченных направления для baseline `TF-IDF
word (1, 2) + LogisticRegression(C=1)`: заменить диапазон n-grams на word
unigrams `(1, 1)`; заменить classifier на `LinearSVC` с прежним TF-IDF;
подобрать только `LogisticRegression.C` в `{0.1, 1.0, 10.0}` с прежними
features. Направления независимы: не объединять изменения и не расширять
набор моделей/параметров по промежуточным результатам. Baseline входит в
сравнение как неизменный контроль. Минимум три направления — учебная цель,
а не обещание улучшить качество.

Вход — проверенные Day02 processed data/provenance, Day03 effective config и
неизменённые split manifest/reference artifacts. Train из manifest — единственный
источник для CV и fit. Validation не участвует в поиске; после выбора по CV
выбранный полный Pipeline один раз fit на всём Day03 train и оценивается на
Day03 validation для сравнения с сохранённым baseline. Это подтверждение не
меняет выбор. Test сохраняется нетронутым: никакие его labels/texts не входят в
fit, transform, prediction, scoring или выбор Day04; финальная test оценка — D5-04.

### Общие folds, метрики и выбор

Создать один набор `StratifiedGroupKFold(n_splits=3, shuffle=True,
random_state=42)` folds только на Day03 train. `groups` берутся из
`data/processed/day03/splits.csv` (`group_id`), а метки и строки выравниваются
по `source_row_id`. Требуются все три класса в train и каждом fold, точная
покрываемость train без повторов, непересечение групп train/validation/test и
между fold train/validation. Невыполненный контракт — явная ошибка без fallback
на обычную стратификацию. Все кандидаты используют сохранённые одни и те же
индексы folds. На каждом fold заново fit полного `sklearn.pipeline.Pipeline`;
TF-IDF vocabulary/IDF не может быть обучен вне fold train.

Primary score — macro F1 с явными `labels=[0, 1, 2]` и `zero_division=0`;
дополнительно сохранить fold accuracy. Для каждого кандидата показать три fold
scores, среднее и population standard deviation (`ddof=0`) для macro F1 и
accuracy. Разброс трёх folds описателен: std не является доверительным
интервалом или тестом статистической значимости. Для всех моделей фиксирован
порядок классов `0=negative, 1=neutral, 2=positive`.

Выбрать максимальное среднее CV macro F1 среди baseline и допустимых кандидатов.
При точном равенстве сохранить baseline; иначе использовать заранее
фиксированный порядок `word_unigram`, `linear_svc`, `logreg_c_0_1`,
`logreg_c_1_0`, `logreg_c_10_0`. Недопустимые конфигурации не участвуют.
Validation показывает, как выбранная конфигурация выглядит на прежнем
сравнительном split, но не используется для повторного выбора или tuning.

### Конфигурация LinearSVC и ошибки выполнения

Зафиксировать до запуска `LinearSVC(penalty="l2", loss="squared_hinge",
dual="auto", tol=1e-4, C=1.0, multi_class="ovr", fit_intercept=True,
intercept_scaling=1, class_weight=None, verbose=0, random_state=42,
max_iter=5000)`, согласовав значения с установленным Day03 pin
scikit-learn 1.9.1. Параметры задаются явно; изменение любой из них не должно
смешиваться с проверкой замены classifier.

Ошибки fit/score не превращать в нулевые метрики: runner должен остановиться с
контекстом candidate/fold (`error_score="raise"` для search API). Записывать
предупреждения `ConvergenceWarning`; конфигурация с предупреждением о несходимости
недопустима для выбора и должна быть явно обозначена. Не скрывать warning или
повторять её с иными параметрами в рамках того же сравнения. Остальные warnings
также сохранять в run context.

### Артефакты, notebook и приёмка

Day04 не изменяет никакие файлы Day02/Day03. Эффективные параметры, folds,
результаты и выбор принадлежат `reports/day04/` и `configs/experiments_day04.json`;
генерируемые reports локальные/ignored. Финальная схема D4-07 включает config,
fold manifest/summary, CV results, selection с причиной выбора, aligned
validation predictions/metrics и provenance с checksums входов/выходов,
seed, версиями и class order. Не сохранять model bundle, test predictions или
test metrics. Сравниваемые результаты должны воспроизводиться в закреплённом
runtime; время выполнения не входит в побайтное сравнение.

`notebooks/day04_experiments.ipynb` — учебный checkpoint по одному разделу на
ID: цель → краткая теория → вызов кода → фактический вывод → интерпретация →
самопроверка. D4-01 загружает и проверяет реальные входные/reference artifacts,
показывает сохранённые Day03 baseline validation metrics и ничего не fit-ит.
Эксперименты начинаются в следующих ID. Отсутствующие локальные данные или
reference требуют воспроизвести Day02/Day03 указанными в `docs/day02.md` и
`docs/day03.md` offline-командами; synthetic fixtures не подставляются.

Приёмка D4-08 включает проверку фактического coverage type gate: notebook
exports находятся в нескрытой временной директории; `filesAnalyzed` должен
быть не меньше числа exports. Нулевой охват не считается успешной проверкой.
Runtime tests и clean-kernel execution не заменяют статическую проверку типов.
Дополнительно проверяются отдельные cells как цепочка modules с импортом
предыдущего контекста: так guard в одной cell не скрывает слишком широкий
объявленный тип в следующей. Это консервативная проверка cell boundaries,
не полная эмуляция Pylance. `Day04Run.metrics` возвращает вложенные TypedDict
для selected/baseline/per_class/delta; потребители не зависят от межъячеечного
flow narrowing. Диагностика пользователя требует regression case даже при
успешном sequential export.

## 10. Контракт day05 — error analysis и frozen final evaluation

### Цель, порядок и ML boundary

Day05 реализует `classification_day05.txt`: найти ошибки, построить confusion
matrix, разобрать реальные примеры разных классов, зафиксировать failure cases
и будущие улучшения. Диагностика выполняется на прежнем Day03 validation для
выбранного по CV Day04 candidate. Дополнительно D5-04 выполняет уже согласованную
в §6/§8/§9 финальную held-out оценку. Это уточнение исходного задания, которое
само не задаёт split или процедуру финального test.

Порядок D5-01…D5-06 описан в PLAN; prerequisite — приёмка D4-08. D5-01 загружает
reference без обучения, D5-02 строит матрицы, D5-03 фиксирует ручной анализ,
D5-04 замораживает протокол и оценивает test, D5-05 собирает offline report,
D5-06 проверяет весь день. Каждый пункт имеет checkpoint в
`notebooks/day05_error_analysis.ipynb`; создание/выполнение начинается при
реализации, а не при планировании.

Не выполнять новый search, выбор candidate по validation/error analysis,
изменение preprocessing, labels, seed, parameters, splits или thresholds.
Идеи улучшений — отдельные будущие train/CV эксперименты. Текущий held-out test
после открытия не может служить независимой проверкой последующих изменений.
Test analysis допустим только как описание замороженного результата; подробные
ручные failure categories этого дня строятся на validation. Inference bundle,
joblib serialization и new-text prediction API относятся к Day06.

### Проверенные входы и alignment (D5-01)

Входы: Day02 processed data/provenance, Day03 manifest/config/reference и Day04
`selection.json`, effective config, `validation_predictions.csv`, validation
metrics и provenance. Проверять checksums и согласованность по существующим
схемам Day03/Day04; не вводить второй несовместимый loader/config format.
Candidate и параметры брать из сохранённого selection, а не зашивать имя
текущего победителя. Отсутствующие/несогласованные artifacts — ошибка с путём и
командой подготовки; не подменять реальную выборку fixtures и не пересоздавать
Day03/Day04 молча.

Day04 predictions имеют колонки `source_row_id`, `true_label_id`,
`selected_pred_label_id`, `day03_baseline_pred_label_id`. Таблица analysis
содержит `source_row_id`, `group_id`, `text`, `text_clean`, `true_label_id`,
`pred_label_id` (явное отображение из `selected_pred_label_id`), `true_label`,
`pred_label`, `is_error`. Все N_valid строк, исходный manifest order; таблица
errors — её фильтр `true_label_id != pred_label_id` с тем же порядком.

Join по `source_row_id` one-to-one; проверять точное покрытие validation IDs,
уникальность, порядок predictions, совпадение true labels с processed data,
`0=negative, 1=neutral, 2=positive`, непересечение с train/test IDs/groups.
Переставленные predictions reference не исправлять молча. Raw/clean text не
менять; пропуски/неизвестные labels/пустой вход — информативные ошибки.
Нулевое число ошибок — корректный результат. Не обращаться к test текстам для
examples или test predictions/metrics до D5-04. Структурная проверка manifest/
provenance на всём processed CSV не является test evaluation.

### Confusion matrix, metrics и выбор примеров (D5-02…D5-03)

Матрица `(3, 3)` всегда имеет строки true, столбцы predicted, порядок классов
negative/neutral/positive. Сохраняются integer counts и row-normalized доли:
`M[i, j] / support[i]`; для support=0 строка нулевая, отсутствие класса явно
отмечается. Counts sum=N, trace=correct, off-diagonal sum=errors,
row sums=true support, column sums=predicted counts. Нормировка по true support
показывает распределение предсказаний внутри класса, а не precision.
PNG имеют подписи axes, классов и типа нормировки; сохраняются без GUI.

Использовать общую Day03 metric policy: macro F1 с labels=[0,1,2], accuracy,
per-class precision/recall/F1/support, zero_division=0. Не сравнивать размер
ошибок разных классов без support. Row metrics учитывают каждую сохранённую
строку; duplicate groups не становятся независимыми наблюдениями.
Baseline reference можно упомянуть по сохранённым validation metrics; новый
baseline/Dummy test запуск не требуется и не входит в Day05.

Для каждой из шести направленных off-diagonal пар детерминированно выбрать
до трёх validation ошибок по возрастанию `source_row_id`. Не дополнять пустые
пары выдуманными примерами. Показывать raw/clean text, IDs, group и true/pred
labels. Выборка для ручного разбора ограничена 18 строками; число уникальных
groups и повторы явно видны. Все ошибки сохраняются отдельно от examples.

Ручные annotations в tracked `docs/day05_failure_cases.csv`: `source_row_id`,
`primary_category`, `observation`, `hypothesis`, `rationale`. Ровно одна строка
на выбранный ID, без лишних/пропущенных/повторных IDs. `hypothesis` может быть
пустой; observation/rationale непусты. Категории — описательная разметка
просмотренных ошибок, не причинное объяснение модели и не новая ground truth.
Counts категорий имеют denominator = число просмотренных examples, не N_valid
и не все ошибки. Возможные категории и self-check вопросы — в PLAN.

Выводы ссылаются на конкретные IDs и направления путаницы. Различать видимое
наблюдение, возможное объяснение и будущую проверку. Не объявлять исходный label
ошибочным и не исправлять его по несогласию модели. Не интерпретировать
LinearSVC decision score как probability; confidence/calibration не нужны.

### Frozen protocol и финальный test (D5-04)

До test predict/metrics записать frozen protocol: candidate, полные параметры
TF-IDF/classifier и preprocessing policy, seed, class mapping/order, metric
policy, checksums processed data/manifest/Day04 selection/config/reference,
train IDs (или их однозначный checksum), фактические runtime versions и fit
policy `day03_train_only`. Test IDs задаются прежним manifest, без нового split.

Day04 не сохраняет fitted Pipeline. Для первого запуска D5-04 создать exact
выбранный Pipeline прежним builder и fit только на Day03 train. Vocabulary/IDF
fit внутри Pipeline; sparse shape `(N_train, V)`, predictions `(N,)`.
Validation не добавляется к train: final test оценивает тот же алгоритм и fit
policy, чьи validation ошибки разбирались. Рефит на train+validation был бы
другим обученным решением и не является default этого проекта.

До обращения к test prediction сверить replay validation predictions с Day04
reference по всем IDs, labels и порядку. Mismatch, checksum/version mismatch,
неверный config или ConvergenceWarning останавливают выполнение до test.
Replay — проверка воспроизведения, не вторая возможность выбрать модель.
После успешного replay один раз вызвать predict на test, затем общие metrics
и confusion matrix. Не менять protocol по результату; не требовать улучшения,
не сравнивать validation/test как paired эксперимент на одинаковых строках.

Сохранить `test_predictions.csv` с `source_row_id`, `true_label_id`,
`pred_label_id` в test manifest order; `test_metrics.json` с
`evaluation_split=test`, N_test, candidate/fit policy, class mapping/order и
общими metrics; test counts/normalized matrix. Labels/texts test не участвуют
в fit и подборе; финальные diagnostics описательны, без tuning.

Повтор D5-04, CLI и clean-kernel notebook использует сохранённый final result:
сверяет protocol/input/runtime/output hashes, IDs/order, пересчитывает metrics
из cached predictions и читает cache без fit/test predict. Несовпадение или
неполный cache — явная ошибка с инструкцией диагностики; автоматическое
перезаписывание, удаление или новый test запуск запрещены. Отдельный final
cache manifest `final_evaluation.json` записывается последним после проверки
roundtrips и hashes всех final outputs; он не включает собственный checksum.
Это маркер завершения, не общая транзакция файлов: прерванный первый запуск
может оставить partial artifacts и требует явного разбора перед повтором.
Нельзя гарантировать exactly-once prediction при аварии до сохранения cache.
Report provenance отдельно фиксирует hashes protocol/cache manifest.

### Артефакты и offline report (D5-05)

| Путь | Назначение |
| --- | --- |
| `reports/day05/validation_analysis.csv` | Все aligned validation строки с текстами/labels и is_error |
| `reports/day05/validation_errors.csv` | Все ошибки, без удаления duplicates |
| `reports/day05/validation_examples.csv` | Детерминированная подвыборка с проверенными ручными annotations |
| `reports/day05/validation_confusion_{counts,normalized}.{csv,png}` | Counts и нормировка по true support |
| `reports/day05/confusion_pairs.csv` | Шесть направленных пар, counts и true support |
| `reports/day05/frozen_protocol.json` | Protocol до test evaluation |
| `reports/day05/test_predictions.csv`, `test_metrics.json` | Замороженная финальная оценка |
| `reports/day05/test_confusion_{counts,normalized}.{csv,png}` | Описательная final test matrix |
| `reports/day05/final_evaluation.json` | Final cache marker, protocol hash и hashes final outputs |
| `reports/day05/analysis_summary.json`, `provenance.json` | Counts/denominators, policies, hashes inputs/outputs, versions, class order |
| `reports/day05/observations.md` | Фактические выводы с явным разделением validation и test |
| `docs/day05_failure_cases.csv`, `docs/day05.md` | Tracked annotations, команды, ключевые выводы и ограничения |

Generated reports локальные/ignored согласно существующей policy; ручные
annotations и ключевые conclusions доступны в Git для review. Hash manifest
не должен содержать циклических ссылок или checksum самого себя.

Reusable analysis в `models/error_analysis.py` (при необходимости небольшие
соседние modules), финальная оценка — отдельная явная функция/команда D5-04.
`models/run_error_analysis_day05.py` собирает report из reference, annotations
и final cache без fit/predict/download/CV. Notebook вызывает те же функции.
Команды реализации документируются в `docs/day05.md`: первый final run
отдельно от report-only; import не вызывает I/O/training. Outputs только Day05,
Day02…Day04 artifacts остаются неизменными. Проверять CSV/JSON roundtrip,
согласованность metrics/counts и input immutability перед записью.

### Приёмка (D5-06)

Результат: реальные validation errors/examples, counts/normalized confusion
matrix, evidence-backed failure cases, гипотезы будущих проверок и отдельный
frozen test report без настройки по test. Все IDs/class order согласованы,
counts имеют denominators; нет обещания роста метрики или live financial quality.
Малый holdout и один split ограничивают выводы; std CV не доверительный интервал.

Offline pytest отдельно проверяет schema/alignment, matrices, deterministic
sampling, annotations, fit boundary/replay/cache и report repeat. Type gate
охватывает Python и экспортированный notebook; compileall/diff check проходят.
Реальные offline artifacts и notebook clean kernels из корня/`notebooks/`
подтверждаются отдельно, с reuse final cache и сохранёнными outputs.
Недоступный запуск явно отмечается; пункт не закрывается по одному наличию кода.
Day06 получает frozen configuration, preprocessing/class contracts и ту же
fit policy, не меняя модель на основании final test результата.

## 11. Контракт day06 — сохранение модели и inference новых текстов

### Цель, границы и порядок

Реализовать пять задач `classification_day06.txt`: сохранение/загрузка лучшей
модели и vectorizer, функция для новых текстов, проверка на новых примерах,
тот же preprocessing и документированный формат входа/выхода.
Prerequisite — приёмка D5-06 и handoff `docs/day05.md`.
Порядок D6-01…D6-08 — в PLAN; каждый пункт реализуется отдельным запросом
с checkpoint в `notebooks/day06_inference.ipynb`. При планировании notebook,
bundle и Python modules не создаются, обучение не запускается.

Day04/Day05 сохраняют конфигурацию и predictions, но не fitted Pipeline.
Поэтому подготовка artifact требует одного явного воспроизводящего fit на
прежнем Day03 train (`day03_train_only`) и validation replay. Это отдельная
build-операция; `load_bundle`, prediction API, CSV CLI и повтор notebook
с готовым bundle никогда не вызывают fit, CV, download или final evaluation.
Не вызывать `run_final_evaluation` ради получения Pipeline: первый запуск
этой функции оценивает test, а cache reuse не возвращает fitted estimator.

Не менять selection, preprocessing, class mapping, seed, row order, retained
train duplicates, параметры или fit policy; не добавлять validation/test
в обучение. Не выполнять новый test transform/predict/score и не пересчитывать
финальную оценку. Проверка структуры/хешей существующих artifacts не является
новой test evaluation. Новые примеры служат smoke demo, не подбором модели.
HTTP service, UI, calibration, новые признаки и финальный README — вне Day06;
финализация README и всего workflow остаётся Day07.

### Подготовка frozen estimator (D6-01…D6-02)

Использовать существующие Day02–Day04 loaders/builders и read-only проверку
Day05 final cache. Входы: processed data/provenance, Day03 config/manifest,
Day04 selection/effective config/reference/provenance, Day05 frozen protocol
и final marker. Отсутствие, partial cache, mismatch checksum/schema/config/
versions/IDs/order — явная ошибка с путём; никаких автоматических перезапусков
предыдущих дней, исправления reference или подмены реальных данных fixtures.

Candidate брать из selection и сверять с frozen protocol; текущий —
`linear_svc`, но default baseline не является fallback. Сверять полные
effective параметры TF-IDF/classifier и preprocessing policy с protocol,
используя уже согласованное JSON-представление dtype/tuples.
Runtime подготовки должен совпадать с frozen protocol. Общую валидацию
переиспользовать без побочного refactor и без новой несовместимой схемы.

Build получает только train/validation данные для model calls. Проверить
`preprocess_text(raw) == text_clean` для этих строк. Fit существующего
Pipeline `tfidf` → `clf` выполняется на clean train в manifest order;
vocabulary/IDF fit внутри Pipeline. Затем exact replay всех validation
predictions относительно Day04: source IDs, true labels, порядок и pred IDs.
Mismatch или `ConvergenceWarning` прекращают build до публикации artifact.
Это проверка воспроизведения замороженного алгоритма и fit policy; она
не доказывает идентичность весов утраченного Day05 fitted объекта.

Текущие проверяемые ориентиры: train 2210 rows / 2206 groups, validation
553 rows / 552 groups, TF-IDF `(2210, 5000)`, classes `[0, 1, 2]`.
Размеры выводить из проверенных данных, не использовать как замену alignment.
Сохранять train IDs или их однозначный checksum, N_train, N_features,
validation replay status и checksum reference. Day02–Day05 inputs неизменны.

D6-02 предоставляет явный `build_inference_bundle(reference)` без записи
на диск. До model calls он повторно проверяет artifacts и переданные snapshots,
raw→clean parity и полные candidate parameters. Перед fit и после replay
сверяются input hashes, preprocessing implementation/policy и runtime.
`BundleMetadata` возвращает `n_train`, `n_validation`, `n_features`, fresh
`validation_replay=exact_match` и `validation_reference_sha256` (checksum
исходного Day04 CSV); contract содержит train IDs в manifest order.
Результат build существует в памяти; D6-03 отдельно сохраняет его на диск.

### Bundle, manifest и загрузка (D6-03)

Предлагаемая reusable логика — `models/inference_artifact.py`; базовая
типизированная структура bundle и результаты подготовки доступны notebook
без зависимости от narrowing в предыдущей cell. Публичные операции:
явная подготовка/build, `save_bundle` и `load_bundle(Path)`; окончательные
signatures зафиксированы в D6-01, без framework/registry:

```python
load_inference_reference(root: Path) -> InferenceReference
build_inference_bundle(reference: InferenceReference) -> InferenceBundle
save_bundle(bundle: InferenceBundle, directory: Path, *, reference: InferenceReference) -> Path
load_bundle(path: Path) -> InferenceBundle
prepare_inference_artifact(root: Path, directory: Path | None = None) -> Path
```

`InferenceReference` содержит отдельные train/validation DataFrames, aligned
validation predictions, split summary только с counts и
`FrozenInferenceContract`; test texts/predictions и estimators в нём отсутствуют.
Contract фиксирует JSON-compatible параметры (ngram tuple → list, dtype → name),
class/fit/preprocessing policy, IDs, runtime и hashes inputs/final cache.
`day05_n_features` / `day05_validation_replay` — проверенная metadata прежнего
cache, не новый Day06 replay. `BundleMetadata` / `InferenceBundle`
возвращаются после fresh Day06 replay и содержат fitted Pipeline. Dataclasses frozen, но
вложенные DataFrames/dicts остаются mutable; callers не меняют snapshots.
SHA-256 preprocessing implementation фиксируется при Day06 preparation;
он не является доказательством исходных bytes кода утраченной Day05 модели.

Один `models/day06/sentiment_bundle.joblib` содержит schema version
`day06-inference-v1`, fitted `Pipeline` целиком (vectorizer вместе с classifier),
preprocessing policy `minimal-v1`, class mapping/order, candidate/full config,
fit policy, seed, train/replay metadata, input hashes и runtime versions.
Preprocessing реализуется импортируемой `preprocess_text`; код функции
не сериализуется как notebook lambda или `__main__` object. Bundle фиксирует
SHA-256 `features/preprocess.py`; loader сверяет policy и implementation hash.
Bundle требует совместимый `src/` и runtime, но для обычного load/predict
не требует dataset, configs или reports предыдущих дней.

Рядом `models/day06/manifest.json`: schema, SHA-256 bundle/fitted state и readable копия
metadata для preflight. До `joblib.load` проверить schema, bundle hash,
preprocessing contract и точное совпадение inference runtime: Python, NumPy,
SciPy, scikit-learn, joblib, threadpoolctl. Остальные версии frozen run
сохранить как provenance; inference не требует matplotlib ради загрузки.
После deserialize проверить metadata agreement, fitted state, steps/types,
classes `[0,1,2]`, dimensions vocabulary/IDF/classifier и effective config.
Ошибки информативны; не дообучать и не выбирать другую модель при отказе.
State checksum связывает vocabulary, IDF, classes, coefficients и intercept;
NumPy integer indices нормализуются только в hash payload, не в самой модели.
Deserialize получает уже проверенные bytes из памяти; после load повторно
проверяется неизменность файлов. Fitted state проверяется без model calls.

Joblib загружает только доверенные локальные artifacts этого проекта:
checksum обнаруживает порчу, но не удостоверяет автора pickle. Это ограничение
описать в developer docs; отдельный approval flow для каждого load не нужен.

Save проверяет roundtrip и validation replay загруженного объекта во временном
каталоге до публикации; итоговый manifest записывается последним. Пара файлов не является общей
транзакцией: partial output блокирует автоматический rebuild/overwrite.
Готовый artifact повторно используется после проверки frozen inputs и metadata
без fit. Несовместимый artifact требует явной диагностики, не удаления молча.
Для нового build разрешён явно заданный новый output directory.
Не требовать одинаковых joblib bytes после независимых builds; проверять
hash сохранённого файла и эквивалентность его состояния/предсказаний.

Явная команда D6-03: `PYTHONPATH=src venv/bin/python -m
finnews_sentiment.models.inference_artifact` из корня; `--root` и
`--output-dir` задают project root/новый каталог. Относительный CLI output
считается от root. `prepare_inference_artifact` проверяет frozen reference,
строит/сохраняет только при отсутствии output; complete artifact используется
без fit/predict/dump после проверки contract/counts. CLI делегирует canonical
Python module, чтобы serialized types не зависели от `__main__`.
Все команды/ограничения — `docs/day06.md`. Notebook читает готовую пару,
а D6-02 section показывает recorded build evidence без повторного обучения.

### Вход и выход prediction API (D6-04)

`predict_texts(bundle, texts: Sequence[str]) -> list[SentimentPrediction]`
в `models/predict.py` принимает непустую последовательность raw English texts.
Одна строка целиком не является batch: для одного текста использовать `[text]`.
`str`/`bytes` вместо batch, нестроковый элемент (включая None/NaN), пустая
последовательность и пустой после `preprocess_text` элемент — явные ошибки
с индексом элемента. Сначала валидировать весь batch, затем предсказывать:
никакого `astype(str)`, скрытого удаления строк или частичного результата.

Каждый typed result содержит `text`, `text_clean`, `pred_label_id: int`,
`pred_label: str`. Ровно N результатов в исходном порядке, duplicates сохранены;
`0=negative`, `1=neutral`, `2=positive`. True labels для inference не нужны.
Raw `(N,)` → общая `preprocess_text` → clean `(N,)` → сохранённый TF-IDF
`(N, V)` sparse → classifier predictions `(N,)` → typed records.
`SentimentPrediction` — frozen dataclass с четырьмя скалярными полями.
API делает один predict после проверки всего batch и fitted bundle;
проверяет integer class IDs/shape/length результата до возврата records.
Не вызывать `clean_news_data`, `preprocess_news` или `add_basic_features`:
это табличные training/inspection контракты, а не обработка unlabeled batch.

Не выполнять fit/fit_transform/partial_fit; vocabulary, IDF, classifier weights
и class order неизменны после predict. Допустимый непустой текст без известных
features даёт нулевой TF-IDF row и штатный class prediction; это ограничение
bag-of-words, не повод fit нового vocabulary. Не обещать abstention/confidence.
Текущий LinearSVC не даёт `predict_proba`; scores, вероятности и calibration
не входят в выход API. Другой classifier ради probabilities не обучать.
D6-06 `predict_for_file`/CSV CLI используют тот же `load_bundle` и
`predict_texts`, без прежнего unchecked loader/tabular preprocessing.
Python API и файловый контракт описаны в `docs/day06.md`.

### Roundtrip, CSV adapter и новые примеры (D6-05…D6-07)

Roundtrip отдельно подтверждает: exact raw→clean parity со всеми train/validation
строками; validation pred IDs до save, после load и через public raw-text API
совпадают с reference по всем IDs/порядку. Дополнительно сравнить vocabulary,
IDF, classes и classifier weights до/после serialization и нескольких predict.
Новый процесс читает только bundle/manifest, совместимый src/runtime и новые
тексты: без чтения training artifacts, fit, network или test predictions.
Offline fixtures проверяют те же failure modes; real replay фиксировать отдельно.

D6-05: `models/inference_consistency.py` предоставляет
`check_inference_consistency(root: Path) -> ConsistencyRun` и CLI
`PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.inference_consistency`.
Typed result содержит `report_path` и `ConsistencyPayload`; report связывает
input/artifact/source hashes, runtime, counts, aligned prediction hashes,
state до/после и standalone evidence. Aligned hash включает source IDs,
true label IDs и predicted IDs в исходном validation порядке.
Pre-save parity опирается на D6-03 manifest/state checksum/build replay;
report явно называет это recorded evidence, не новым save/roundtrip.

Child запускается через текущий Python с `-I -B` во временной cwd с копиями
только src/artifact pair. Guards блокируют training/serialization/network,
artifact reads при import и чтение вне isolated/runtime/system paths;
data/configs/reports не копируются и их чтение запрещено. Проверяются
установленные guards, import purity и полное совпадение новых API records
parent/child. Это Python audit verification, не OS security sandbox.

Report записывается только после всех checks, через replace одного JSON file.
Ошибка сохраняет прежний report, одинаковые inputs/source/runtime дают
byte-identical content без timestamp/temp path/PID/self-checksum; совпадающий
report не переписывается. Source checksum не включает сам report.

Сохранить привычный CSV entrypoint в `models/predict.py`:
`PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.predict
--model models/day06/sentiment_bundle.joblib --input <input.csv> --output <output.csv>`.
`--model` требует соседний manifest; CLI вызывает тот же `load_bundle`/`predict_texts`.
Вход UTF-8 CSV с обязательной `text`, без обязательных sentiment/label/provider.
Пустой файл/table, missing/blank/non-string text, дублированные column names
или уже существующие output columns — ошибки; текстовые значения вроде `NA`
не превращать незаметно в NaN. Дополнительные columns и raw text сохраняются.
Выход: прежние columns плюс `text_clean`, `pred_label_id`, `pred_label`, без pandas
index, столько же строк и тот же порядок. Не требовать уникального text/ID.
Не перезаписывать input или существующий output; входные ошибки не создают
готовый output. Пути `Path`, directories относительно выбранного project root
или явно переданные caller; личных абсолютных путей в коде нет.
`predict_for_file(model_path: Path, input_path: Path, output_path: Path) -> None`
читает CSV стандартным strict csv.reader как строки: без type/NA inference,
с проверкой уникального непустого header и exact field count каждой строки.
Header-only/blank physical rows и malformed CSV не игнорируются молча.
Output сохраняет parsed field values, не исходное byte formatting: UTF-8,
CRLF row terminator и корректное quoting в том числе для lone CR внутри поля.
Полный output сериализуется до публикации; input bytes сверяются после predict.
Публикация через temporary file и exclusive hard link в том же каталоге
не заменяет чужой output даже при гонке; temporary file удаляется при ошибке.
Existing output (включая symlink) и input/output aliases/hardlinks запрещены.
CLI использует paths относительно cwd, сообщает ошибки через argparse (exit 2,
без traceback), не запускает build/download и не создаёт output на input error.

В tracked `docs/day06_examples.csv` подготовить небольшой набор новых вручную
составленных английских финансовых фраз, не скопированных из dataset.
Показать growth/loss/neutral event, отрицание/сравнение, числа/валюты/HTML,
неизвестную лексику; не требовать трёх разных predicted классов.
Ручные examples — unlabeled smoke inputs, не synthetic замена real-data replay
и не ground truth для accuracy/F1. Показывать фактический output без правки
примеров/модели ради красивого результата. Дубликаты/невалидный ввод отдельно
демонстрируют контракт API, а не качество модели.

### Артефакты и приёмка (D6-08)

| Путь | Назначение |
| --- | --- |
| `models/day06/sentiment_bundle.joblib`, `manifest.json` | Локальный ignored artifact для load/predict |
| `reports/day06/consistency.json` | Input/bundle hashes, runtime, raw-clean/validation/state/standalone checks |
| `reports/day06/demo_predictions.csv` | Фактические результаты новых examples |
| `docs/day06_examples.csv`, `docs/day06.md` | Tracked demo inputs, контракты, build/load/CSV команды и ограничения |
| `notebooks/day06_inference.ipynb` | Восемь sections, сохранённые outputs и самопроверка |

Generated model/report outputs остаются ignored по существующей policy;
ручные docs/examples и reusable code отслеживаются Git. Consistency report
не содержит собственного checksum и не выдаёт smoke checks за ML quality.

Приёмка: полный offline pytest, type gate (project + sequential notebook
exports + cell-boundary modules), compileall/diff check; отдельно реальный
train-only build/replay/roundtrip, standalone process, CSV API equivalence и
notebook из корня/`notebooks/` в чистых kernels. После появления bundle notebook
использует reuse: build вынесен в явную команду, дальнейшие sections не требуют
повторного fit. До D6-03 временный D6-02 checkpoint может явно воспроизводить
train-only fit; после D6-03 этот section показывает проверенные bundle metadata.
Готовый notebook проверять с запрещёнными fit/CV/download/final evaluation;
Day02–Day05 files и bundle/manifest неизменны по SHA-256.
Не обещать переносимость pickle между версиями, идентичные веса утраченной
Day05 модели или качество на live news. Непроведённые checks указывать явно.
