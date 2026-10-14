# Day03 — reproducible baseline

Day03 строит воспроизводимую исходную точку для экспериментов классификации
тональности. Исходные классы — `negative`, `neutral`, `positive`; mapping:
`0=negative`, `1=neutral`, `2=positive`.

## Запуск

Из корня checkout и при наличии локальных Day02 processed data/provenance:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.train_model --config configs/train_baseline.json
```

Команда offline: она не скачивает данные и сохраняет split manifest в
`data/processed/day03/`, а summary, aligned validation predictions, metrics,
effective config и provenance в `reports/day03/`. Runner валидирует входные
checksums и пересчитывает validation metrics из сохранённых predictions.
Для чистой установки сначала отдельно подготовьте входные данные по инструкции
Day02; загрузка источника — отдельное явное действие, не часть baseline runner.

Notebook checkpoint: `notebooks/day03_baseline.ipynb`. В терминале проверки
из корня выполняются командами из `AGENTS.md`: полный pytest suite, Pyright
через `scripts/check_types.py`, compileall и `git diff --check`.

## Результат reference

На `sentences_75agree` (3 453 строки, 3 448 exact-text groups) используется
group-stratified split с seed 42: train 2 210 строк, validation 553, test 690.
Обучение использует только train. TF-IDF fit даёт sparse матрицу
`(2210, 5000)`; конфигурация ограничивает vocabulary 5 000 word unigrams и
bigrams. Validation accuracy/macro F1 составили 0.813743/0.720965 для
LogisticRegression и 0.622061/0.255667 для `DummyClassifier(most_frequent)`.
Подробный class balance и interpretation приведены в
[`../reports/day03/observations.md`](../reports/day03/observations.md).

Эти показатели относятся только к одной validation выборке и не гарантируют
качество для новых финансовых текстов. Test не использовался для fit, выбора
или метрик Day03. Сохраните его до протокола финальной оценки; выбор модели
следует выполнять по validation/CV и заранее заданным критериям. Один split
не показывает разброс результатов, а bag-of-n-grams не понимает дальний
контекст, синонимы и временные сдвиги.
