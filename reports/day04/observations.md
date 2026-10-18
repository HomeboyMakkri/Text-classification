# Day04 — фактические наблюдения

Источник: локальные `cv_summary.csv`, `metrics.json`, `selection.json` и
`provenance.json`, повторно проверенные при приёмке D4-08 (2026-10-18).
Классы: `0=negative`, `1=neutral`, `2=positive`. Данные/seed/splits не менялись.

## Эксперименты и выбор

CV проводится на 2210 train строках / 2206 exact-text groups; validation
отдельно содержит 553 строки, held-out test — 690. Три общих group-aware folds
и fit TF-IDF внутри каждого train fold предотвращают пересечение duplicate
groups и leakage vocabulary/IDF. Все кандидаты используют один протокол.

- Word unigrams: mean macro F1 0.669960 против baseline 0.664277 (+0.005683),
  но mean accuracy ниже: 0.789599 против 0.792315. Из трёх fold macro F1 только
  один улучшился; это небольшой неоднородный сигнал.
- LinearSVC с прежними features: mean/std macro F1 0.768715/0.024064 против
  baseline 0.664277/0.026227. Macro F1 улучшился во всех трёх folds; mean accuracy
  0.836660. Здесь замена classifier дала наибольший mean CV macro F1.
- LogisticRegression C: mean macro F1 0.381356/0.664277/0.759263 для
  C=0.1/1/10. Более слабая regularization C=10 помогла относительно baseline;
  C=0.1 существенно ухудшил результат. C=1 воспроизводит контроль.

Выбран `linear_svc` по фиксированному maximum mean CV macro F1, tie отсутствует.
Три направления дали шесть строк сравнения (пять уникальных конфигураций).
Нет ConvergenceWarning; выбор не изменялся после validation confirmation.
Std трёх folds — описательный разброс, не confidence interval; преимущество
над C=10 на 0.009452 не доказывает статистическую значимость. Ограниченный grid
не доказывает оптимальности C=10 вне трёх значений. Выбор максимума из нескольких
вариантов может адаптироваться к folds даже без прямого leakage.

## Validation confirmation и дисбаланс

На полном train TF-IDF shape `(2210, 5000)`, predictions `(553,)`.
На прежних validation IDs macro F1 выбранного решения 0.805961 против Day03
baseline 0.720965: +0.084995 абсолютной разницы и +11.79% относительно baseline.
Accuracy 0.848101 против 0.813743: +0.034358, или +3.44 процентного пункта.
Это confirmation, а не основание для нового выбора.

| Класс | Support | Baseline precision / recall / F1 | LinearSVC precision / recall / F1 |
| --- | --- | --- | --- |
| negative | 67 | 0.861111 / 0.462687 / 0.601942 | 0.870370 / 0.701493 / 0.776860 |
| neutral | 344 | 0.815534 / 0.976744 / 0.888889 | 0.851175 / 0.947674 / 0.896836 |
| positive | 142 | 0.790476 / 0.584507 / 0.672065 | 0.827586 / 0.676056 / 0.744186 |

Support суммируется в 553. Neutral — 62.21% validation; accuracy сильнее
определяется большинством, тогда как macro F1 одинаково взвешивает три класса.
Recall negative/positive вырос, recall neutral снизился с 0.976744 до 0.947674,
при этом F1 вырос для всех классов. Это наблюдаемый компромисс, а не доказательство
причины ошибок. Повторные строки учитываются как отдельные строки метрик,
поэтому duplicate groups имеют вес по размеру и не независимы.

## Ограничения и следующий день

Три folds и одна validation выборка не дают гарантии generalization или live
качества. Vocabulary ограничен, word n-grams не представляют весь финансовый
контекст; score LinearSVC не probability. Ручные примеры ошибок и confusion
matrix ещё не разбирались: это Day05. Test не использовался для transform,
prediction, scoring или tuning; его финальная оценка остаётся D5-04.

Day05 получает `selection.json`, effective config/provenance, validation
predictions/metrics и прежний Day03 manifest. Inference bundle остаётся Day06.
Команды воспроизведения и исправление type gate — в `docs/day04.md`.

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
