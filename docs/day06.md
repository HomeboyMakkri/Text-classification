# Day06 — сохранение модели и проверенная загрузка

D6-01…D6-08 завершены: frozen input loader, explicit train-only build, joblib bundle/
manifest, typed raw-text API, consistency/standalone verification, CSV CLI
и сохранённые smoke examples. Итоговая приёмка подтверждена 2026-10-19.

## Подготовка artifact

Входы — принятые локальные Day02–Day05 artifacts; команды их подготовки и
frozen final evaluation описаны в `docs/day03.md`–`docs/day05.md`.
Использовать существующий `venv/`, `requirements-day03.txt` и dev requirements;
не обновлять runtime, чтобы обойти ошибку version mismatch.
Из корня проекта:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.inference_artifact
```

Команда `prepare_inference_artifact(root, directory=None)`:

1. Проверяет frozen inputs и reference прежнего решения.
2. Если artifact отсутствует, вызывает `build_inference_bundle(reference)`:
   fit только на прежнем clean train в manifest order, exact validation replay.
3. `save_bundle(bundle, directory, reference=reference)` проверяет fitted state,
   staging roundtrip и повторный validation replay загруженного объекта.
4. Публикует bundle и последним manifest. Test не предсказывается.

По умолчанию создаются ignored local files:

- `models/day06/sentiment_bundle.joblib`: fitted `tfidf` → `clf` Pipeline и
  typed metadata с frozen configuration/provenance/preprocessing/classes.
- `models/day06/manifest.json`: schema `day06-inference-v1`, checksum bundle,
  checksum fitted vocabulary/IDF/classes/weights и readable metadata.

Первый build делает явный fit; повтор той же команды с готовой согласованной
парой читает и проверяет её без fit/predict/joblib.dump и без изменения bytes.
Несовпадение inputs/config/runtime или повреждённая/неполная пара — ошибка,
не разрешение заново обучить модель или пересчитать test.

Для отдельной новой сборки указать **новый** output directory:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.inference_artifact \
  --root . --output-dir models/day06_alternative_build
```

Относительный `--output-dir` CLI считается от `--root`. Python API принимает
явный `Path` output directory. Existing files не перезаписываются. Запись пары
не атомарна целиком: interruption после bundle, но до manifest оставит partial
output; его необходимо диагностировать отдельно. Checksum не содержит timestamp
или self-checksum; независимые builds не обязаны иметь одинаковые joblib bytes.

## Загрузка без обучения

```python
from pathlib import Path
from finnews_sentiment.models.inference_artifact import load_bundle

bundle = load_bundle(Path("models/day06/sentiment_bundle.joblib"))
print(bundle.metadata.contract.candidate_id)
print(bundle.metadata.n_train, bundle.metadata.n_features)
print(bundle.metadata.validation_replay)
```

Для Python/script нужен `src` в `PYTHONPATH`, как в команде выше.
`load_bundle` требует соседний `manifest.json`, совместимый src/runtime;
dataset, configs и reports не читает. До `joblib.load` проверяются schema,
metadata/preprocessing policy/implementation SHA-256, exact inference versions
и checksum bytes. Deserialization читает уже проверенные bytes из памяти.
После неё проверяются typed object/metadata agreement, Pipeline steps/types,
fitted state, classes `[0,1,2]`, vocabulary/IDF/weights dimensions, effective
parameters и checksum state. Никаких fit, predict или автоматического repair.

Joblib следует загружать только из доверенного локального источника:
pickle может исполнять код. SHA-256 обнаруживает порчу, но не удостоверяет
автора artifact. Точные inference versions: Python, NumPy, SciPy,
scikit-learn, joblib, threadpoolctl. Остальные frozen versions — provenance;
matplotlib не требуется для standalone load. Переносимость между версиями
не обещается. Preprocessing выполняется импортируемой `preprocess_text`,
а не сериализованной notebook lambda или `__main__` class.

## Raw-text batch API (D6-04)

```python
from pathlib import Path
from finnews_sentiment.models.inference_artifact import load_bundle
from finnews_sentiment.models.predict import predict_texts

bundle = load_bundle(Path("models/day06/sentiment_bundle.joblib"))
results = predict_texts(bundle, ["<b>No</b> growth in revenue; costs rose by €2.5 million."])
print(results[0].text_clean)
print(results[0].pred_label_id, results[0].pred_label)
```

Вход — непустой `Sequence[str]` (обычно list/tuple); один текст — `[text]`.
Bare str/bytes, non-sequence batch, empty batch, None/NaN/non-string, blanks
или empty-after-clean elements вызывают TypeError/ValueError до predict;
индекс невалидного элемента указан в ошибке. Весь batch проверяется заранее,
строки не исключаются, missing не превращается в строку `"nan"`.
Literal strings `"nan"`/`"NA"` остаются текстами.

Выход — `list[SentimentPrediction]`, frozen dataclass с полями `text`,
`text_clean`, `pred_label_id: int`, `pred_label: str`. N/order/duplicates и raw
text сохранены; mapping `0=negative, 1=neutral, 2=positive`. True labels не нужны.
Используется та же `preprocess_text`, затем один fitted Pipeline.predict;
не вызываются tabular training cleanup, fit/fit_transform/partial_fit.
Vocabulary/IDF/weights/classes остаются прежними. Непустой OOV-only text
допускается: zero-feature row получает штатное предсказание classifier.
Это не понимание неизвестной лексики и не confidence/abstention mechanism.
Scores/probabilities в records отсутствуют, LinearSVC не даёт predict_proba.

Preprocessing `minimal-v1` декодирует HTML entities, убирает полные HTML tags,
приводит к lowercase и нормализует whitespace. Отрицание, числа, валюты и
пунктуация сохраняются. Stopwords/stemming/lemmatization здесь не применяются;
на новых текстах vocabulary/IDF не пересчитываются.

## CSV inference (D6-06)

Из корня проекта, с готовым trusted local artifact, существующим input и
новым output path:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.predict \
  --model models/day06/sentiment_bundle.joblib \
  --input path/to/new_news.csv --output path/to/predictions.csv
```

Пути относительно cwd; можно передать абсолютные. `--model` требует соседний
manifest.json, не запускает build/download и не читает training references.
Python adapter: `predict_for_file(model_path: Path, input_path: Path,
output_path: Path) -> None`, импорт из `finnews_sentiment.models.predict`.

| Column | Input | Output |
| --- | --- | --- |
| `text` | Обязательный непустой raw text | Сохранён без очистки |
| Любые дополнительные columns | Необязательные строки | Сохранены в прежнем порядке |
| `text_clean` | Зарезервирована, вход с ней запрещён | Тот же minimal-v1 preprocessing |
| `pred_label_id` | Зарезервирована | Integer literal 0/1/2 |
| `pred_label` | Зарезервирована | negative/neutral/positive |

Sentiment/true label/provider/headline не требуются. CSV UTF-8 с comma
delimiter и обычным quote escaping; все fields читаются как строки.
Literal `NA`/`nan`, leading zeros, empty extra fields, currency/quotes/newlines
сохраняются; numeric-looking text тоже остаётся текстом. Duplicates text/IDs
не удаляются и не требуют уникальности. Header должен быть непустым/unique;
число fields в каждой строке должно совпадать с числом columns.

Output содержит header + все N строк в прежнем порядке, без pandas index.
Это сохранение parsed field values, не original byte formatting: output
UTF-8 с CRLF и корректным quoting (включая lone CR внутри field).
Весь batch проверяется до predict, затем целиком сериализуется. Input bytes
повторно сверяются перед публикацией; input/model/manifest не изменяются.
Output directory создаётся после успешной проверки/inference. Existing output
никогда не перезаписывается, в том числе при гонке; сбой публикации не
оставляет частичный CSV или временные files.

Примеры ошибок: `text,text` → duplicate header; input `text_clean` → collision;
последняя строка `<b></b>` → `texts[index] is empty after preprocessing`, весь
batch отвергается; input=output/hardlink → разные files обязательны;
existing output → FileExistsError. CLI печатает понятную error (exit 2) без
traceback и не создаёт output на invalid input/model/path.

## Новые smoke examples (D6-07)

Tracked `docs/day06_examples.csv` содержит восемь вручную составленных
английских inputs с вымышленными названиями компаний и `example_id` 01…08.
Это identifier, не label. Темы: рост прибыли, убыток, календарное событие,
отрицание роста, сравнение с прошлым годом, HTML/валюта, неизменный прогноз,
OOV-only text. До первого inference фразы зафиксированы; raw/clean совпадений
с canonical dataset нет. Проверка читает только text, не использует test labels
и не вызывает test transform/predict/score.

С готовой парой model/manifest выполнить из корня:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.predict \
  --model models/day06/sentiment_bundle.joblib \
  --input docs/day06_examples.csv --output reports/day06/demo_predictions.csv
```

Получится ignored local `reports/day06/demo_predictions.csv`: `(8, 2)` →
`(8, 5)`, все исходные fields плюс cleaned text и actual prediction. Это
генерируемый report; в Git добавляется только input CSV. Нужны существующий
`venv/`, pinned runtime из `requirements-day03.txt` (он включает Day01) и
совместимая trusted artifact pair. Для notebook дополнительно нужны Jupyter/
IPython из Day01; `requirements-dev.txt` нужен для pytest/type gate.

Если report уже существует, CLI откажет в перезаписи. Для проверки повтора
сохранить результат в новый временный каталог и сравнить bytes:

```bash
demo_check_dir=$(mktemp -d)
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.predict \
  --model models/day06/sentiment_bundle.joblib \
  --input docs/day06_examples.csv --output "$demo_check_dir/predictions.csv"
cmp reports/day06/demo_predictions.csv "$demo_check_dir/predictions.csv"
```

Временный каталог можно удалить после проверки. Notebook делает тот же
repeat через TemporaryDirectory, сверяет все fields и public API, автоматически
удаляет временный output. При несовпадении он останавливается, не заменяя report.
Build/reuse модели — отдельная команда из первого раздела; inference не
восстанавливает отсутствующую/повреждённую пару и не обучает новый classifier.

Фактический первый запуск: `01/04/05` → positive, остальные → neutral;
negative не появился, и это допустимо. В частности, фраза `02` про net loss
получила neutral, а `04` с did not increase — positive. Эти ответы показываются
без изменения inputs или модели. Нулевой OOV row (`08`) также дал neutral;
это обычное решение classifier при отсутствии известных признаков.
Здесь нет ground-truth labels, accuracy/F1, confidence или обещания качества
на live news. Bag-of-words может плохо учитывать отрицание и сравнение.
Duplicates/invalid inputs демонстрируются отдельно в D6-04/D6-06 notebook;
smoke examples не заменяют real validation replay D6-05.

## Consistency и standalone check (D6-05)

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.inference_consistency
```

Команда использует готовый bundle и frozen references, ничего не обучает и
не сериализует. Reusable API: `check_inference_consistency(root: Path)`;
typed result содержит `report_path` и `report`.

Проверяются raw-clean parity всех train/validation строк, IDs/truth/order и
все validation predictions загруженного Pipeline/public API против Day04
reference. Aligned SHA-256 включает source IDs, true labels и predictions.
Vocabulary/IDF/classes/weights и metadata не должны изменяться. Evidence
до/после save берётся из проверенного D6-03 manifest/state hash/build replay;
для этой проверки новый before-save объект не создаётся и save не повторяется.

В child копируются только src и два artifact files. Python запускается
через тот же interpreter с `-I -B`, новой cwd и guards: запрет training/
serialization/network, artifact I/O при import и чтения вне copied/runtime/
system paths. Dataset/configs/reports отсутствуют в workspace и запрещены
для чтения; probes проверяют установку guards. Parent/child должны вернуть
одинаковые records на двух новых unlabeled текстах. Это проверка через Python
audit, не OS sandbox. Import checker сам не читает данные и не запускает child.

Только после успеха всех modes записывается ignored
`reports/day06/consistency.json`: input/artifact/source hashes, runtime,
counts, parity, state и standalone evidence. Ошибка не затирает прошлый report;
тот же результат byte-identical и не переписывается. Timestamp, PID,
temporary paths и self-checksum в JSON отсутствуют. Это подтверждение
согласованности, не новая независимая оценка accuracy/F1.

## Notebook и границы ML

`notebooks/day06_inference.ipynb` исполняется из корня/`notebooks/` с чистого
kernel. D6-01 читает frozen references, D6-02 показывает сохранённую build
metadata, D6-03 — verified artifact load, D6-04 — new-text API calls,
D6-05 — full consistency и guarded standalone process.
В D6-06 temporary CSV показывает adapter/API equivalence и failure modes;
temporary inputs/outputs удаляются после показа таблицы.
В D6-07 показаны восемь сохранённых raw/clean/predicted rows и API/CLI parity;
при повторе сохранённый report проверяется без перезаписи.
Готовый notebook не обучает модель и не пишет bundle/manifest. Predict
используется в D6-04 на новых inputs и в D6-05 для validation parity/new-text
standalone comparison. При отсутствии artifact
сначала отдельно выполнить build-команду; notebook не создаёт его автоматически.

Candidate/config/seed/row order/duplicates сохраняются по frozen protocol.
Текущий candidate — `linear_svc`, train 2210 rows / 2206 groups, sparse TF-IDF
`(2210, 5000)`; все 553 validation predictions воспроизводятся. Validation
не включается в train. Test не проходит новые transform/predict/score.
Replay подтверждает воспроизведение алгоритма/fit policy на данном validation,
а не идентичность утраченных весов Day05 или новое независимое качество.

Checkpoint D6-03 (2026-10-18): реальная сборка выполнила один train-only fit
и два exact validation replay (до save и после staging roundtrip).
Два prepare/load repeats с запрещёнными fit/predict/dump/CV/network/final-run
не изменили artifact bytes. Standalone load проверен в отдельном процессе:
читаются только bundle/manifest/preprocess source, matplotlib не импортируется.
Notebook выполнен из двух cwd в чистых kernels с запрещёнными model calls
и записью artifact; все 48 прежних input/final files и пара artifact неизменны.

Offline D6 reference/build/artifact tests: 86 passed, из них 29 новых artifact
tests. Type gate: 49 project files и 61 notebook modules (6 sequential +
55 cell-boundary), 0 errors/warnings. CLI Pyright не является полным запуском
Pylance внутри VS Code. Compileall и diff check успешны; полный pytest suite
в D6-03 не запускался. Local model/report artifacts не добавляются в Git.

Checkpoint D6-04 (2026-10-19): 31 новый prediction test, вместе с artifact
regression — 60 passed. Type gate: 50 project files и 63 notebook modules
(6 sequential + 57 cell-boundary), 0 errors/warnings; это CLI Pyright, не полный
запуск Pylance внутри VS Code. Полный pytest suite здесь не запускался.
Реальный batch/single/OOV inference и invalid-input rejection проверены без
fit/test prediction; fitted state/metadata неизменны. Все восемь notebook
code cells выполнены в чистых kernels из двух cwd; разрешены только new-text
model calls, training/serialization/CV/final-run/network запрещены.
Все 50 input/final/artifact files неизменны; compileall/diff check успешны.
Unlabeled smoke outputs не используются для accuracy/F1 или настройки модели.

Checkpoint D6-05 (2026-10-19): 21 новый consistency test, вместе с prediction
regression — 52 passed. Type gate: 52 project files и 65 notebook modules
(6 sequential + 59 cell-boundary), 0 errors/warnings. Это CLI Pyright,
не полный Pylance внутри VS Code; полный pytest suite не запускался.
Реальные 2210 train + 553 validation raw/clean pairs и все 553 predictions
совпали. Loaded Pipeline/public API/child fitted state согласованы;
serialization evidence reused, fit/test prediction не выполнялись.
Report содержит 30 input/artifact hashes и 26 source hashes, повторяется
byte-for-byte. Все десять notebook code cells выполнены из двух cwd в чистых
kernels с guards. Все 50 прежних input/final/artifact files неизменны;
compileall/diff check успешны. Новая ML quality measurement не заявляется.

Checkpoint D6-06 (2026-10-19): 31 новый CSV test; вместе с prediction и
consistency regression — 83 passed. Type gate: 53 project files и 67 notebook
modules (6 sequential + 61 cell-boundary), 0 errors/warnings. CLI Pyright не
является полным запуском Pylance внутри VS Code; full pytest не запускался.
Real temporary input `(3,3)` дал output `(3,6)`; API/adapter/CLI predictions
совпали, adapter/CLI CSV bytes одинаковы. Separate CLI process и все 12 notebook
code cells (два cwd/clean kernels) проверены с training/network/dump guards.
50 прежних input/final/artifact files и fitted state неизменны; test prediction
не выполнялся. Consistency report regenerated для текущих source hashes.
Compileall/diff check успешны. Persistent demos остаются D6-07.

Checkpoint D6-07 (2026-10-19): восемь новых inputs без canonical raw/clean
совпадений; actual CSV/API parity и byte-identical CLI repeat проверены.
Все 14 code cells исполнены из двух cwd в чистых kernels с training/network/
serialization guards; output и интерпретация сохранены. 83 целевых tests passed;
53 project files + 69 notebook modules (6 sequential / 63 cell-boundary),
0 errors/warnings. CLI Pyright не является полным VS Code Pylance; полный
pytest suite остаётся D6-08. Build-команда проверена как reuse без fit/predict/
dump; Python snippets и demo CLI commands исполнены на реальном bundle.
Все 49 frozen files из snapshot и fitted state неизменны; compileall/diff check
успешны. Новых quality measurements, tuning или test prediction не было.

## Итоговая приёмка D6-08 и handoff Day07

2026-10-19: Day06 принят по `classification_day06.txt` и SPEC §11.

| Критерий | Проверенное свидетельство |
| --- | --- |
| Модель и vectorizer сохранены/загружаются | Fitted Pipeline tfidf → clf, manifest/hash/runtime/state validation; два prepare/load повтора без fit/transform/predict/score/dump |
| Рабочий inference на новых текстах | Typed API + CSV CLI; 8 сохранённых smoke inputs/outputs, API parity и byte-identical repeat |
| Единый preprocessing | minimal-v1; 2210 train + 553 validation raw/clean pairs совпали |
| Согласованность предсказаний | Все 553 validation predictions совпали с reference; guarded standalone process passed |
| Понятный формат использования | Build/load/Python/CSV команды, I/O/errors/reuse и ограничения выше |

Полный `venv/bin/python -m pytest tests -v`: **442 passed, 46 subtests passed**
за 65.97 s. Это offline tests с fixtures; обучение на fixtures не является
новым real-data build. Type gate: **53 project files**, **71 notebook modules**
(6 sequential exports + 65 cell-boundary modules), 0 errors/warnings.
`compileall` и `git diff --check` успешны. CLI Pyright не является полным
запуском Pylance внутри VS Code.

Отдельно все **16 code cells / 8 sections** notebook выполнены из корня и
`notebooks/` в чистых kernels. В real checks training, serialization, CV,
network и final evaluation запрещены; parent TF-IDF transform/predict допускают
только validation/new-text inputs, fitted state проверяется до/после predict.
CLI child дополнительно запрещает чтение training references; D6-05 child
проверяет standalone load/predict без dataset/configs/reports. Сохранённые
outputs показывают фактические результаты. D6-08 section сверяет manifest,
input/source/state checksums и показывает handoff paths/runtime.

Нового real fit/test transform/predict/score не было. Train-only build и
serialization roundtrip evidence остаются D6-02/D6-03: train 2210 rows /
2206 groups, 5000 features, fresh validation replay до save/после staging load.
Приёмка использует существующую пару. Все **50** защищённых files (configs,
data, Day02–Day05 reports, Day05 failure CSV, bundle/manifest) и отдельно
**3** demo/report files неизменны по SHA-256. Replay не доказывает идентичность
утраченных весов Day05 и не является новой независимой оценкой качества.

### Что передаётся Day07

Команды build/reuse, load/API и CSV приведены выше. Для ordinary inference
достаточны совместимый `src/`, pinned runtime, trusted local bundle/manifest
и новые inputs; полный build/consistency workflow дополнительно требует
frozen Day02–Day05 inputs. Generated model/report files остаются ignored.
Текущие локальные SHA-256 (не универсальные hashes для независимых builds):

| Artifact | SHA-256 |
| --- | --- |
| `models/day06/sentiment_bundle.joblib` | `72d5f018ccd5babf9debb597e799ab02a76f8dc551babd216857f9d10cef1300` |
| `models/day06/manifest.json` | `7b155e9fada9f7d4404801e316d52dc0b0242f52c5de1700aa7a1b62392d1080` |
| `reports/day06/consistency.json` | `c4822ea3da21b3b78f0f43f7c59e71094e09517a84ae16a4f6f2dc6664233b03` |
| `docs/day06_examples.csv` | `d99af46c74f82bf0fd999a9ea84e4de66e3ed4b7ddb9dbf93f3f6a7a8a9c363c` |
| `reports/day06/demo_predictions.csv` | `2d689d528e0c4520e5b9e0616cf7206808f959409e83b66ea89d106d5856b6db` |

Fitted state SHA-256:
`7623ab0d4d555aed3c73944f59f85acb23e141ada2794fa826c9d7231383b7f7`.
Frozen choice: `linear_svc`, fit policy `day03_train_only`, seed 42,
`0=negative, 1=neutral, 2=positive`. TF-IDF word ngrams `[1,2]`,
`max_features=5000`, `lowercase=False`, без stopwords; LinearSVC `C=1.0`,
`max_iter=5000`. Полные effective parameters, train IDs, preprocessing hash
и source/config/data provenance — в manifest metadata и consistency report;
`configs/train_baseline.json` и `configs/experiments_day04.json` неизменны.

| Runtime | Frozen version |
| --- | --- |
| Python | 3.14.6 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |
| threadpoolctl | 3.7.0 |
| pandas (workflow/provenance) | 3.0.6 |
| matplotlib (workflow/provenance) | 3.11.2 |

Joblib — только trusted local; checksum не удостоверяет автора. Совместимость
между версиями и качество на live news не обещаются. LinearSVC не даёт
probabilities; OOV/отрицание/сравнение остаются ограничениями demos.

Day07 pending: end-to-end review, итоговый README, общая таблица результатов
и ограничений, воспроизведение workflow. Его roadmap нужно детализировать
перед реализацией; финальный README и Day07 notebook в D6-08 не менялись.

Проверки fixtures отдельно от реального build/roundtrip и clean-kernel runs:

```bash
venv/bin/python -m pytest tests -v
venv/bin/python scripts/check_types.py
venv/bin/python -m compileall -q src tests
git diff --check
```
