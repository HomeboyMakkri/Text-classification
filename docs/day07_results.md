# Итоговые результаты mini-product — D7-03

Проверено **2026-10-25** по локальным accepted artifacts Day03–Day06.
Это сводка прежних измерений, не новый ML experiment. Existing readers
проверили frozen inputs, source IDs/order/labels, CV choice и completed final
cache; Day05 report дополнительно проверен по 29 input + 10 output checksums.
`load_final_evaluation` вернул `cache_status=reused`: только read-only validation
сохранённых predictions/metrics, без fit/test transform/predict/score.
Первый final test run выполнен в D5-04, а не в этом пункте.

Контракты — [SPEC](../SPEC.md), команды и evidence — [Day07](day07.md),
учебные таблицы — [Day07 notebook](../notebooks/day07_review.ipynb).
Источники чисел и SHA-256 перечислены в конце. Generated reports local/ignored;
их отсутствие не запускает автоматический rebuild/final evaluation.

## Данные, протокол и формы

Financial PhraseBank `sentences_75agree`: 3453 англоязычных финансовых
предложения, 3448 exact-clean-text groups. Group-stratified split с seed 42:
train 2210 строк / 2206 groups, validation 553 / 552, test 690 / 690.
Одинаковые raw/clean тексты не пересекают splits; похожие по смыслу предложения
могут оставаться зависимыми. Same-label duplicates сохраняют вес по числу строк.

Классы всегда `0=negative`, `1=neutral`, `2=positive`. Shared preprocessing —
`minimal-v1`, fit policy — `day03_train_only`: vocabulary/IDF и classifier fit
только на train. Baseline — word TF-IDF unigrams+bigrams, max_features=5000,
LogisticRegression C=1; Dummy использует most_frequent.

DataFrame train `(2210, 5)` / validation `(553, 5)` → fitted TF-IDF sparse
`(2210, 5000)` / `(553, 5000)` → predictions `(N,)` → confusion matrix `(3, 3)`.
При CV `N_fold_train` и `V_fold <= 5000` зависят от fold; Pipeline fit внутри
каждого fold. Эти fitted размеры подтверждены прежними metadata, сами
transform/predict в D7-03 не запускаются.

## Day03: baseline на validation

Validation support: negative=67, neutral=344, positive=142; всего N=553.
Каждая строка таблицы использует те же source IDs. Источник —
`reports/day03/metrics.json` и aligned saved predictions, заново сверенные
через `evaluate_predictions` без вызова estimator.

| Модель | N | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| Dummy most_frequent | 553 | 0.622061 | 0.255667 |
| TF-IDF + LogisticRegression baseline | 553 | 0.813743 | 0.720965 |

Dummy предсказывает neutral для каждой строки. Его accuracy равна доле neutral;
macro F1 учитывает также два редких класса с нулевым F1 и показывает слабость
этого ориентира. Macro F1 — среднее трёх class F1 с одинаковым весом;
accuracy — доля правильных ответов по всем строкам.

## Day04: выбор по train-only CV

Три общих `StratifiedGroupKFold` на прежних 2210 train строках / 2206 groups,
seed 42. Vocabulary/IDF fit внутри Pipeline каждого fold. Источники —
`reports/day04/cv_summary.csv`, `cv_fold_summary.csv`, `selection.json`
и проверенная provenance. Std рассчитан с ddof=0.

| Candidate | Mean CV macro F1 | Std macro F1 | Mean CV accuracy |
| --- | --- | --- | --- |
| `baseline` | 0.664277 | 0.026227 | 0.792315 |
| `word_unigram` | 0.669960 | 0.019630 | 0.789599 |
| `linear_svc` | 0.768715 | 0.024064 | 0.836660 |
| `logreg_c_0_1` | 0.381356 | 0.023907 | 0.672852 |
| `logreg_c_1_0` | 0.664277 | 0.026227 | 0.792315 |
| `logreg_c_10_0` | 0.759263 | 0.023086 | 0.832588 |

Три направления: word unigrams вместо unigrams+bigrams; LinearSVC вместо
LogisticRegression; grid LogisticRegression C={0.1, 1, 10}. Шесть строк включают
контроль и повтор его параметров C=1: уникальных configurations пять.
Направления не объединялись в новый общий search.

Выбор — maximum mean CV macro F1; exact ties разрешаются заранее заданным
порядком SPEC с baseline первым. Победил `linear_svc`, tie отсутствует;
validation/test не выбирали candidate. Итог: TF-IDF word ngrams=(1,2),
max_features=5000; LinearSVC C=1, max_iter=5000; полные параметры в selection
и frozen protocol. Разница mean CV macro F1 с C=10 около 0.009452:
три folds и их std не доказывают статистически значимое превосходство.
CV score нельзя сопоставлять с validation score как одинаковые measurements.

## Day04: validation confirmation выбранной модели

После выбора LinearSVC fit на прежнем полном train; validation не добавлялась
в fit. Источники — `reports/day04/metrics.json`/predictions и Day05 checked
validation table. N=553 и support=67/344/142 совпадают с Day03.

| Модель | N | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| TF-IDF + LinearSVC | 553 | 0.848101 | 0.805961 |

По сравнению с baseline на тех же validation IDs абсолютная delta macro F1
**+0.084995**, относительная **+11.79%**.
Расчёт до округления: `0.8059606214204886 - 0.7209651379298799`
= `0.0849954834906087`; `(selected - baseline) / baseline * 100`
= `11.7891253014893%`. Это validation confirmation frozen choice,
а не обещание такого же улучшения test/live quality.

| Класс | Support | Baseline recall / F1 | Selected precision / recall / F1 |
| --- | --- | --- | --- |
| negative | 67 | 0.462687 / 0.601942 | 0.870370 / 0.701493 / 0.776860 |
| neutral | 344 | 0.976744 / 0.888889 | 0.851175 / 0.947674 / 0.896836 |
| positive | 142 | 0.584507 / 0.672065 | 0.827586 / 0.676056 / 0.744186 |

Minority recall вырос относительно baseline, neutral recall снизился;
F1 каждого класса вырос. Это наблюдаемый trade-off на одной validation выборке,
не доказательство механизма обучения или будущего превосходства.

## Day05: frozen held-out test

Источник — checked `reports/day05/test_metrics.json`, frozen protocol и
completed cache. Оценена только выбранная модель с прежним train-only fit.

| Модель | N | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| TF-IDF + LinearSVC | 690 | 0.856522 | 0.797050 |
| LogisticRegression baseline | — | не оценивался | не оценивался |
| Dummy | — | не оценивался | не оценивался |

| Класс | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| negative | 84 | 0.941176 | 0.571429 | 0.711111 |
| neutral | 428 | 0.858921 | 0.967290 | 0.909890 |
| positive | 178 | 0.821656 | 0.724719 | 0.770149 |

Neutral — 428/690 строк, поэтому общий accuracy 0.856522 может сочетаться
с negative recall 48/84 = 0.571429. Macro F1 0.797050 даёт редким классам
равный вес. Validation и test содержат разные строки: их разница не paired
improvement, и нет test baseline для заявления о test delta.
Здесь не выполнялись новые test predictions, просмотр новых test примеров
или настройка по test. Уже открытый test не станет независимой проверкой
будущих изменений модели.

## Validation error analysis

Источник — `reports/day05/confusion_pairs.csv`, counts/normalized matrices,
`validation_examples.csv` и tracked [manual annotations](day05_failure_cases.csv).
Новый summary сверяет матрицы и annotations через existing functions:
`build_validation_confusion`, `select_validation_examples`,
`load_failure_annotations`. Строки матрицы — true, столбцы — predicted;
fractions нормированы на support true-класса, включая правильные ответы.

Validation: **84 ошибки / 553 строки**, 469 правильных ответов;
**552 groups**, среди errors **84 groups**. Один дополнительный retained
duplicate row не удалялся. Похожие ID 2854/2855 — разные exact-text groups;
разные IDs не доказывают независимость наблюдений.

| True → predicted | Errors | True support | Доля true-класса |
| --- | --- | --- | --- |
| negative → neutral | 15 | 67 | 22.39% |
| negative → positive | 5 | 67 | 7.46% |
| neutral → negative | 3 | 344 | 0.87% |
| neutral → positive | 15 | 344 | 4.36% |
| positive → negative | 4 | 142 | 2.82% |
| positive → neutral | 42 | 142 | 29.58% |

Просмотрены **18 строк / 18 groups**: до трёх минимальных source IDs на каждую
направленную пару. Это deterministic illustrative sample, не случайная выборка
для оценки распространённости причин. Среди **18 просмотренных**:

| Primary category | Count / reviewed rows |
| --- | --- |
| mixed sentiment | 0/18 |
| financial context | 12/18 |
| negation/comparison | 3/18 |
| insufficient context/ambiguous label | 2/18 |
| preprocessing/representation limitation | 0/18 |
| other/unclear | 1/18 |

12/18 financial context не означает 12/84 errors или долю во всей популяции.
Нулевой count категории среди 18 примеров не доказывает её отсутствия во всех
ошибках. Категории не меняют исходные labels и не устанавливают причинность.

| Validation source_row_id | Сохранённое наблюдение | Граница интерпретации |
| --- | --- | --- |
| 2854, negative→positive | EPS явно описан как loss | Причина предсказания не установлена; веса признаков не исследовались |
| 2598, neutral→negative | Уменьшилась доля ownership | Тип объекта события важен для интерпретации; это гипотеза о features, не причинный вывод |
| 1294, positive→negative | Ожидаемая прибыль сопоставлена с прошлым loss | Требуется временная привязка; влияние конкретного слова на ответ не доказано |
| 321, positive→negative | Net sales выросли на 28% | Категория other/unclear: механизм ошибки остаётся неизвестным |

Существующие будущие гипотезы Day05 — временные сравнения/область отрицания
и контекст типа финансового события. Для них нужны отдельные заранее заданные
train/CV experiments и новый согласованный evaluation protocol. В D7-03
такие эксперименты не запускались; preprocessing/candidate/parameters frozen.

## Ограничения

- Англоязычные финансовые предложения `sentences_75agree` с agreement ≥75%;
  выводы не распространяются автоматически на другие языки, длинные документы,
  текущие новости или другой domain/time period. Лицензия данных — CC-BY-NC-SA-3.0.
- Один group split, три CV folds, ограниченный grid, retained duplicates и
  возможная зависимость похожих текстов ограничивают статистические выводы.
  CV std не confidence interval; validation уже просмотрена, test уже открыт.
- Class imbalance и minority recall требуют чтения per-class metrics;
  accuracy отдельно недостаточен. Иллюстративные annotations не оценивают
  частоты причин среди всех ошибок, ambiguity не означает ошибочность labels.
- Bag-of-ngrams с V≤5000 ограничивает представление дальнего контекста,
  отрицаний/сравнений и OOV. LinearSVC не предоставляет probabilities;
  score нельзя представлять как уверенность. Unlabeled demo не измеряет качество.
- Joblib используется только для trusted local artifacts с совместимым runtime;
  checksum удостоверяет целостность, не источник/автора, переносимость между
  версиями не гарантирована. Тональность не прогноз доходности и не торговая
  рекомендация.

## Источники и SHA-256

Дата проверки всех entries — **2026-10-25**. Paths относительны корню проекта.
Configs и manual annotations tracked; перечисленные data/metrics/report/model
artifacts local/ignored. Existing validators проверили upstream provenance;
новые ML predictions/models не создавались. Hash fingerprints относятся к этому snapshot.

| Source path | SHA-256 |
| --- | --- |
| `configs/dataset.json` | `fc43038477b9beeb491c505244b7b0a403b8da46f062f61cdc8358126382047d` |
| `configs/train_baseline.json` | `8ffeb9230ae0cbffd6c00939694d028370ea5e825eaabf1cbb374cc025de7315` |
| `configs/experiments_day04.json` | `d9830dd3805252cc523a2fcb6fc22a28638e2c59d31feec11233cb78d908d18a` |
| `data/processed/day03/splits.csv` | `b51d363cac4c00f18d760aa72d3ca6f5272becced6be45d18e763b4b9e557aaf` |
| `reports/day03/provenance.json` | `73f72d587bff42974d4eed6274d6886f03843d264aaf7ef73e801c870d8ad5dd` |
| `reports/day03/split_summary.json` | `8fe5b294800246fa9883def1a9fc615831385ed2c689a2cfd861aa42be4e3bc4` |
| `reports/day03/metrics.json` | `727282dc6860f800b0ed05bcb1d17bef7fff007db424c2bb3647da20bc470436` |
| `reports/day03/validation_predictions.csv` | `b5433913dcfbd94b36effcf759d5d7e8e30af7b04c01f8dfafcde80709400928` |
| `reports/day04/provenance.json` | `ba101b6e2b5d8b388c4f435fa6b1e954df18d8b2f04987b57fcfcb1bd5f3edbc` |
| `reports/day04/cv_summary.csv` | `11fe2b505d8c79797abd484539782d258f497b17d7d60133345afce89f760750` |
| `reports/day04/cv_fold_summary.csv` | `bdb8ae5d8cb94e85e17bf3be78c6390b28fd863fe821d08a14184c155ee090cc` |
| `reports/day04/selection.json` | `83c3c1805900ea148baf88657f7d4b3862f81a4bdabacbe2615a1ce0f1ac3b98` |
| `reports/day04/metrics.json` | `c095b034a2d45d709c27d507802600d34ec7a6ecfda798b7e29a17c27edb0043` |
| `reports/day04/validation_predictions.csv` | `71d74d84970a956d80306899b0928431177ffcfa7498f3f4fe6e275245a2f987` |
| `reports/day05/provenance.json` | `66f094db32335118500f2defa2ac692a53a8dd961988f8ec91d8f24f5c312c3d` |
| `reports/day05/analysis_summary.json` | `471ec8562f8466e41dfdea84afd0b774c16f6017d38e68c4ecec0c7224297d89` |
| `reports/day05/validation_confusion_counts.csv` | `41e1735273854c6cac94c5e92044d511ba89366e32219fca874a98663f04828e` |
| `reports/day05/validation_confusion_normalized.csv` | `76ea2473b12b8ccf7d4aa81ebd7444715814dabf9a1a0ef5d1772a463eb20470` |
| `reports/day05/confusion_pairs.csv` | `126f3d561fd199d15fd7c37c9946dde03d79ba9285a53da81208cc9f1882a770` |
| `reports/day05/validation_examples.csv` | `fec137352a02ebdfcd6e93f912989aaeeef76503cdf0584cb131be03b0efa3e4` |
| `docs/day05_failure_cases.csv` | `7c4c6381ea288b34c2ca90764d3a2e970550d34345d7d5712ce5b4e832f0823f` |
| `reports/day05/frozen_protocol.json` | `4a9a778085b8f082a20374bf46e1a2053a72aa9a6ee872e3e252dbb8a2570a37` |
| `reports/day05/final_evaluation.json` | `1cc8749072afec98f30c8107a92bb7c8d8a526bdc2063e8e7647df68f85e792b` |
| `reports/day05/test_metrics.json` | `ff5b028ae8dbae88cf7f3a8d151edd9a2b1b98b56b803182da7c077b05ce45f9` |
| `reports/day06/consistency.json` | `c4822ea3da21b3b78f0f43f7c59e71094e09517a84ae16a4f6f2dc6664233b03` |
| `models/day06/manifest.json` | `7b155e9fada9f7d4404801e316d52dc0b0242f52c5de1700aa7a1b62392d1080` |
