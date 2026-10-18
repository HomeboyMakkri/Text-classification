# Day04 — контролируемый выбор модели

## Входы и offline запуск

Нужны локальные Day02 processed data/provenance и неизменные Day03 reference
artifacts, включая `data/processed/day03/splits.csv`. При их отсутствии отдельно
выполните подготовку по [day02.md](day02.md) и [day03.md](day03.md).
Day04 ничего не скачивает и не изменяет Day02/Day03 artifacts.

Из корня проекта:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.run_experiments_day04 --root . --config configs/experiments_day04.json
```

Runner проверяет checksums, восстанавливает прежние splits, создаёт общие
train-only folds, сравнивает кандидатов и сохраняет `reports/day04/`:
`cv_folds.csv`, `cv_fold_summary.csv`, `cv_results.csv`, `cv_summary.csv`,
`effective_config.json`, `selection.json`, `validation_predictions.csv`,
`metrics.json`, `provenance.json`. Generated artifacts локальные/ignored;
[observations.md](../reports/day04/observations.md) хранится в Git для review.
Model bundle и test predictions/metrics здесь не создаются.

## Что сравнивается и как выбирается

Фиксированный Day03 split: train 2210 строк / 2206 groups, validation 553 / 552,
test 690 / 690. CV использует только train, три `StratifiedGroupKFold` с seed=42;
внутри каждого fold заново fit полного Pipeline, включая TF-IDF vocabulary/IDF.
Один словарь, обученный до CV, привёл бы к leakage. Train TF-IDF sparse,
`(N_fold_train, V)`, `V <= 5000`; validation predictions имеют shape `(553,)`.

Проверены три независимых направления: word unigrams, замена classifier на
LinearSVC и grid LogisticRegression `C ∈ {0.1, 1, 10}`. Это шесть строк в
сводке, включая baseline и повтор его параметров как `C=1`, то есть пять
уникальных конфигураций. Направления не объединялись.

| Candidate | Mean CV macro F1 | Std (ddof=0) | Mean CV accuracy |
| --- | --- | --- | --- |
| baseline | 0.664277 | 0.026227 | 0.792315 |
| word_unigram | 0.669960 | 0.019630 | 0.789599 |
| linear_svc | 0.768715 | 0.024064 | 0.836660 |
| logreg_c_0_1 | 0.381356 | 0.023907 | 0.672852 |
| logreg_c_1_0 | 0.664277 | 0.026227 | 0.792315 |
| logreg_c_10_0 | 0.759263 | 0.023086 | 0.832588 |

Выбор — максимум mean CV macro F1. Exact ties разрешаются фиксированным
порядком из SPEC §9 с приоритетом baseline. Победил `linear_svc`, tie отсутствует.
Три fold scores показывают разброс, но std не является доверительным интервалом.
Разница mean macro F1 между LinearSVC и `C=10` равна 0.009452; этого недостаточно
для заявления о статистически доказанном превосходстве.

После выбора полный Pipeline обучается на всём прежнем train. Validation только
подтверждает этот выбор: macro F1 0.805961 против baseline 0.720965,
absolute delta +0.084995, relative +11.79%; accuracy 0.848101 против 0.813743.
На полном train vocabulary имеет 5000 features, TF-IDF shape `(2210, 5000)`.
Подробные class trade-offs — в observations; анализ отдельных ошибок и
confusion matrix остаются Day05.

Повтор CLI/notebook повторяет фиксированный эксперимент для проверки
воспроизводимости, включая fit/validation predict, без расширения поиска или
нового выбора по validation. Идентичные результаты ожидаются в закреплённом
runtime; времена fit/score не сравниваются побайтно. Test не передаётся
estimator ни при первом, ни при повторном запуске.

## Notebook и проверки

`notebooks/day04_experiments.ipynb` содержит разделы D4-01…D4-08. Root/src setup
работает из корня и `notebooks/`. Для исполнения используйте отдельные output
файлы, сохраняя открытый пользовательский notebook:

```bash
mkdir -p .cache/day04-checkpoints
venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=text-classification --ExecutePreprocessor.timeout=600 --output-dir=.cache/day04-checkpoints --output=from-root.ipynb notebooks/day04_experiments.ipynb
```

Для второго запуска из `notebooks/`:

```bash
cd notebooks
../venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=text-classification --ExecutePreprocessor.timeout=600 --output-dir=../.cache/day04-checkpoints --output=from-notebooks.ipynb day04_experiments.ipynb
```

Сначала проверяются типы и offline tests:

```bash
venv/bin/python scripts/check_types.py
venv/bin/python -m pytest tests -v
venv/bin/python -m compileall -q src tests
git diff --check
```

Type gate должен печатать число реально проверенных файлов. Исправлен дефект:
экспорты в `.cache` Pyright пропускал, выдавая `0 errors` при `filesAnalyzed=0`.
Теперь временная директория не скрытая, а gate останавливается, если число
проверенных файлов меньше ожидаемого. Регрессионные tests проверяют валидную
ячейку, ошибку типов в notebook и исключённые из анализа exports.
В D4-06 `idxmax()` возвращал слишком общий `Hashable` для `.loc`; теперь
используются числовая позиция `argmax()` и `.iloc`. В D4-07 перед обращением
к вложенным metrics проверяется, что значения `dict[str, object]` — словари.
Runtime success не доказывает корректность статических типов.

## Передача Day05

Сохраните Day04 selection/config/provenance, aligned validation predictions и
прежний Day03 split manifest. Test остаётся закрыт до D5-04. Ошибки validation
разбираются без перенастройки текущего решения; гипотезы улучшений требуют
отдельных будущих train/CV экспериментов. Ни один результат Day04 не гарантирует
качества на новых финансовых новостях или при временном сдвиге данных.

## Свидетельства приёмки D4-08 (2026-10-18)

- Полный offline suite: 190 tests и 46 subtests passed.
- Type gate: 35 Python файлов и 4 notebook exports, 0 errors/warnings;
  coverage notebook подтверждён после исправления пропуска скрытой директории.
- Compileall src/tests/scripts и `git diff --check` прошли.
- Два offline CLI запуска и clean-kernel notebook из root и `notebooks/`:
  9/9 code cells без ошибок в каждом, hashes 9 generated artifacts идентичны,
  11 input hashes не изменились. Полные copies/logs/acceptance hashes сохранены
  локально в `.cache/day04-checkpoints/`. В основной notebook перенесён только
  output нового D4-08; прежние outputs сохранены, пользовательские kernels
  не прерывались. Day05/test evaluation/inference не запускались.
