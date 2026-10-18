# Day05 — validation error analysis

## D5-03: протокол ручного просмотра

Источник — проверенные Day02–Day04 artifacts и сохранённые validation predictions
`linear_svc`. Split, labels, preprocessing, candidate и параметры не изменялись.
На этапе D5-03 test examples/predictions/metrics не использовались; final evaluation
выполнена отдельно в D5-04 ниже.

`select_validation_examples` берёт до трёх ошибок с минимальными `source_row_id`
для каждой из шести true→pred пар в порядке negative/neutral/positive.
Это детерминированная иллюстративная выборка, не случайная оценка распространённости
причин. Все пары непусты: просмотрено 18 строк из 84 ошибок на 553 validation строках.
Все ошибки сохранены отдельно от examples. Пустая пара при других inputs явно
отмечается `status=empty`; выдуманными строками она не дополняется.

Ручная разметка — `docs/day05_failure_cases.csv`. `observation` описывает текст,
`hypothesis` содержит возможное объяснение (может быть пустой), `rationale`
объясняет выбор primary category. Эти категории не доказывают причину предсказания
и не заменяют исходные labels. Один просмотренный пример не доказывает, что модель
«не понимает отрицания». Decision scores не интерпретируются как вероятности.

## Наблюдения

1. Financial context встречается в разных направлениях. ID 2854 и 2855
   (negative→positive) явно содержат EPS loss; ID 2598 (neutral→negative)
   сообщает об ownership decreased, а ID 1593 (neutral→positive) — о запуске
   продукта. Знак отдельного слова и тип описываемого события не взаимозаменяемы.
   Мы не проверяли веса признаков и не установили, какое слово вызвало ответ.
2. ID 1294 и 1374 (positive→negative) сравнивают текущую/ожидаемую прибыль
   с прошлым убытком. Наличие profit и loss здесь требует временной привязки,
   а не автоматически категории mixed sentiment. ID 2156 (neutral→negative)
   содержит отрицание изменения плана, а не сообщение об убытке.
3. Neutral→positive ID 743 описывает необходимость strengthen operations,
   а не достигнутое улучшение. Neutral→negative ID 1428 даёт абсолютную прибыль
   без базы сравнения. Отсутствующий контекст наблюдаем, но не позволяет
   объявить исходный label неправильным.
4. В positive→neutral ID 16 и 28 указаны рост доходов и удвоение стоимости,
   а ID 59 описывает расширение возможностей сервиса. В ID 321
   (positive→negative) рост продаж указан прямо; причина ошибки остаётся
   `other/unclear`, без приписывания модели доказанного механизма.

Raw и clean text показаны рядом в notebook. У просмотренных examples clean text
соответствует lowercasing; в этих строках нет наблюдаемого удаления отрицания или
числового сигнала. Категории mixed sentiment и preprocessing/representation
limitation не добавлялись ради заполнения словаря.

## Denominators и duplicate groups

Просмотрено **18 строк / 18 уникальных groups**, повторов groups внутри examples
нет; каждая выбранная group встречается в validation один раз. Во всех validation
данных 553 строки / 552 groups, среди 84 ошибок — 84 groups. ID 2854 и 2855 похожи,
но это разные exact-clean-text groups: не объединяем их и не считаем независимость
доказанной только из-за разных IDs.

Primary categories среди **18 просмотренных**: financial context — 12/18 (66,67%),
negation/comparison — 3/18 (16,67%), insufficient context/ambiguous label — 2/18
(11,11%), other/unclear — 1/18 (5,56%). Эти доли не относятся ко всем 84 ошибкам,
553 validation строкам или всей популяции. Ровно одна primary category упрощает
review, но не исключает пересечения возможных объяснений.

## Будущие гипотезы: отдельные train/CV эксперименты

1. Приоритет 1: проверить обработку сравнений и области действия сигналов
   (profit/loss по периодам, отрицание изменения). В отдельном эксперименте
   заранее задать одну альтернативу текущим word features — например,
   признаки сравнения/временной привязки — и сравнить на прежних общих train-only
   group-aware folds. Fit всех learned transformations внутри Pipeline каждого fold;
   оценивать macro F1 и per-class recall, а не успех на выбранных 18 примерах.
2. Приоритет 2: проверить финансовый контекст объекта события (ownership,
   EPS loss, необходимость улучшения против достигнутого результата). Заранее
   сформулировать ограниченный набор contextual features и проверить ablation
   на тех же train-only folds. Рассматривать validation как уже просмотренную;
   подтверждение будущего улучшения требует отдельного согласованного протокола.

Ни один эксперимент здесь не запущен. Текущий candidate и параметры заморожены;
held-out test оценивает прежнее решение, не выбирает эти гипотезы.

## Воспроизведение checkpoint

Запустить `notebooks/day05_error_analysis.ipynb` с чистого kernel из корня проекта
или `notebooks/`. Notebook использует reusable functions без fit/predict/CV и
сохраняет в ignored `reports/day05/` analysis, все errors, annotated examples,
а также D5-02 CSV/PNG. Отсутствующая/неполная ручная разметка вызывает явную ошибку;
она не создаётся автоматически по predictions.

```bash
venv/bin/python -m pytest tests/test_day05_failure_cases.py -v
venv/bin/python scripts/check_types.py
```

Самопроверка: почему 12/18 financial context не означает 66,67% всех ошибок?
Чем гипотеза про исторический loss отличается от наблюдения в тексте?
Почему разные source IDs не гарантируют независимые наблюдения?


## D5-04: frozen final test evaluation

2026-10-18 выполнен первый отдельный final run с `cache_status=created`.
Candidate `linear_svc` и его параметры взяты из Day04 selection. До fit/test
сохранён `reports/day05/frozen_protocol.json`: preprocessing, selected parameters
и полный `get_params` TF-IDF/classifier (dtype записан именем NumPy dtype),
seed=42, class mapping/order, metric policy, input hashes, train/validation/test
IDs, фактические runtime versions и fit policy `day03_train_only`.

Pipeline обучен только на **2210 train строках**, sparse TF-IDF shape
**(2210, 5000)**. Validation не добавлялась в fit. До test predict точное
совпадение всех **553 validation predictions**, IDs, true labels и порядка
с reference Day04 подтверждено; `ConvergenceWarning` не было. После replay
вызван один test predict для **690 строк** прежнего manifest.

| Final test | Результат |
| --- | --- |
| Accuracy | 0.856522 |
| Macro F1 | 0.797050 |
| Negative precision / recall / F1 / support | 0.941176 / 0.571429 / 0.711111 / 84 |
| Neutral precision / recall / F1 / support | 0.858921 / 0.967290 / 0.909890 / 428 |
| Positive precision / recall / F1 / support | 0.821656 / 0.724719 / 0.770149 / 178 |

Это описательные результаты замороженного решения. Macro F1 усредняет F1 трёх
классов с одинаковым весом; accuracy учитывает частоты классов. Test и validation
состоят из разных строк и не образуют paired эксперимент. Итог не используется
для выбора новых параметров, preprocessing, порога или seed. Новые test examples
не анализировались; baseline/Dummy/candidates на test не запускались.

Сохранены `test_predictions.csv`, `test_metrics.json`,
`test_confusion_{counts,normalized}.{csv,png}` и frozen protocol.
`final_evaluation.json` записан последним после roundtrips и проверки записанных
bytes; он хранит protocol hash, runtime versions, replay metadata и hashes шести
outputs, без собственного checksum. Это completion marker, не транзакция всех
файлов. При аварии после test predict, но до завершения cache нельзя обещать
exactly-once prediction.

Явный запуск и повтор той же команды:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.final_evaluation
```

Завершённый cache даёт `cache_status=reused`: проверяются текущие inputs,
protocol/config/versions, output checksums, IDs/order/labels и воспроизведение
metrics/matrices из cached predictions. **Fit/predict отсутствуют**; реальный
повтор отдельно проверен с запрещёнными `Pipeline.fit` и `Pipeline.predict`.
Read-only API `load_final_evaluation(ROOT)` используется notebook и предназначен
для report-only runner D5-05; при отсутствии cache он не обучает модель.

Неполный/повреждённый cache, другой protocol, версии или входы вызывают ошибку
с путём и инструкцией диагностики. Автоматическое удаление, перезапись и новая
test оценка запрещены. При partial artifacts сначала явно разбираются
`frozen_protocol.json`, marker, входные/output hashes; простой повтор команды
не начинает обучение. Exclusive creation protocol резервирует первый запуск;
возобновление незавершённого запуска автоматически не реализовано.

Notebook с чистого kernel читает final cache без новой оценки и показывает
отдельный `evaluation_split=test`, cache status, train shape, per-class metrics
и обе test matrices. Отдельный report-only CLI D5-05 описан ниже; Day06 model
bundle здесь не создаётся.

Самопроверка: почему изменение модели после чтения test нельзя подтвердить
тем же test как независимым? Почему даже совпавшие validation metrics слабее,
чем точное совпадение всех replay predictions?


## D5-05: offline report-only runner

Отдельная команда сборки отчёта:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.run_error_analysis_day05
```

При другом рабочем каталоге передать `--root` с путём к корню проекта;
notebook работает как из корня, так и из `notebooks/`. Reusable API:
`run_error_analysis_day05(ROOT)`. Это report-only путь: он читает проверенный
final cache через `load_final_evaluation`, не вызывает fit/predict/CV/download
и не запускает автоматически команду первой финальной оценки.

Входы: processed data и provenance Day02; прежний Day03 manifest/config/reference;
Day04 selection/config/predictions/metrics/provenance и CV artifacts; ручные
`docs/day05_failure_cases.csv`; восемь frozen final artifacts D5-04. Их hashes
попадают в report provenance. Candidate, параметры, seed, preprocessing,
class order и splits остаются прежними. Даже test PNG являются cache inputs
и не перезаписываются отчётом.

В `reports/day05/` заново собираются 11 report outputs:

- `validation_analysis.csv`: все 553 validation строки;
- `validation_errors.csv`: все 84 ошибки, без удаления duplicates;
- `validation_examples.csv`: 18 детерминированных examples с ручными annotations;
- `validation_confusion_{counts,normalized}.{csv,png}`: true rows / predicted columns;
- `confusion_pairs.csv`: шесть направленных пар с count, true support и долей;
- `analysis_summary.json`: validation/test metrics отдельно, counts/denominators,
  group counts, manual category counts, class order и policies;
- `observations.md`: фактические validation направления, выбранные ID и ручные
  observation/hypothesis/rationale, затем отдельный frozen test результат и ограничения;
- `provenance.json`: input/output SHA-256, runtime versions, policies,
  отдельные hashes frozen protocol и final cache marker.

В provenance 10 output checksums: сам `provenance.json` исключён, циклических
ссылок нет. Final cache и manual annotations являются inputs. Generated reports
локальные/ignored; ключевые выводы и ручная разметка сохранены в tracked docs
выше. Доли categories относятся только к 18 reviewed rows, не к 84 errors
или всем 553 validation строкам.

Runner проверяет исходные схемы/alignment/checksums, final cache и annotations,
согласованность counts/errors/pair denominators, CSV/JSON roundtrip и неизменность
inputs перед записью. CSV сохраняют пустую hypothesis как пустую строку;
ID/order/raw/clean text не меняются. Затем проверяются записанные bytes.
Выходы только Day05; Day02–Day04 и final cache не изменяются. Запись отчёта
последовательная, не общая транзакция: ошибка записи может оставить часть
report outputs. Их можно воспроизвести report-only командой после диагностики;
это не даёт права пересоздавать frozen final cache.

При отсутствующем cache сообщение требует **отдельно** выполнить D5-04 командой
`finnews_sentiment.models.final_evaluation`; при partial artifacts сначала
необходима диагностика. Повреждённый cache, checksum/config/runtime mismatch,
пропущенные/лишние annotations или изменение входа во время сборки завершают
report с ошибкой. Они не превращаются в новую test оценку.

При одинаковых inputs/runtime CSV, JSON, Markdown, PNG и provenance повторяются
byte-for-byte: timestamp не добавляется. Проверки отдельно запрещают model calls
и network, проверяют zero-error fixture, reviewed-error fixture, повтор API/CLI,
пустую hypothesis, input immutability, cache/annotation corruption и import
без I/O. Реальный report и clean-kernel notebook используют сохранённый final cache.

Один split и малый holdout ограничивают обобщение; std CV не доверительный
интервал. Duplicate rows не являются независимыми наблюдениями. Нет обещания
роста метрики или качества для live financial decisions. Day06 inference bundle
и новая настройка модели в D5-05 не входят.

Самопроверка: какие hashes нужны, чтобы связать macro F1 с конкретными inputs,
selection и frozen run? Почему checksum самого provenance создаёт цикл?
Почему повтор отчёта не должен делать новую test prediction?


Checkpoint D5-05 (2026-10-18): реальный report-only CLI использовал cache `reused`;
две API сборки с запрещёнными fit/predict/CV/network дали одинаковые bytes всех
11 report files. Provenance содержит 29 input hashes и 10 output hashes.
34 исходных/final files остались неизменны. Notebook выполнен с чистого kernel
из корня и `notebooks/`: hashes всех 45 input/final/report files совпали до/после.
Offline pytest: 273 passed, 46 subtests passed (15 новых report tests).
Type gate: 45 project files и 53 notebook modules (5 sequential + 48 cell-boundary),
0 errors/warnings; CLI Pyright не является полным запуском Pylance внутри VS Code.
Compileall и diff check успешны. Итоговая приёмка D5-06 остаётся отдельным пунктом.
