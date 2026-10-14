# Day03 — baseline observations

## Данные и splits

Источник — Financial PhraseBank `sentences_75agree`, revision
`8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964`; preprocessing — `minimal-v1`.
Manifest содержит 3 453 строки и 3 448 групп. Split выполнен один раз по
`group-stratified-v1` с seed 42. Одинаковые группы не пересекают train,
validation и test; disjointness проверена для source IDs, исходного и очищенного
текста.

| Split | Строки | Группы | negative | neutral | positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 2 210 | 2 206 | 269 | 1 374 | 567 |
| Validation | 553 | 552 | 67 | 344 | 142 |
| Test | 690 | 690 | 84 | 428 | 178 |

Классы в таблице приведены в фиксированном порядке
`negative`, `neutral`, `positive`. Все 5 точных повторов сохранены: 4 строки
сверх первой находятся в train, 1 — в validation. Противоречивых duplicate
labels нет. Fuzzy duplicates не искались, поэтому paraphrases могут оставаться
в разных группах.

## Baseline reference

TF-IDF использует word analyzer, unigram/bigram (`ngram_range=(1, 2)`),
`max_features=5000`, IDF и L2 normalization. Фактическая train матрица sparse,
shape `(2210, 5000)`; словарь обучен только на train. LogisticRegression:
`solver=lbfgs`, `C=1.0`, `max_iter=1000`, `tol=1e-4`, `class_weight=None`,
`random_state=42`; classes `[0, 1, 2]`. ConvergenceWarning не возник.
DummyClassifier использует `most_frequent` и предсказывает нейтральный класс.

| Модель | Validation accuracy | Validation macro F1 |
| --- | ---: | ---: |
| TF-IDF + LogisticRegression | 0.813743 | 0.720965 |
| Dummy `most_frequent` | 0.622061 | 0.255667 |

По классам baseline recall равен 0.462687 (negative), 0.976744 (neutral) и
0.584507 (positive). У Dummy recall neutral равен 1, а negative и positive — 0.
Большая доля neutral объясняет, почему accuracy Dummy выше доли двух меньших
классов, но macro F1 учитывает все три класса равным весом. Baseline reference
сохраняет эти результаты как точку сравнения; значения не являются обещанием
качества на других выборках.

## Ограничения и граница оценки

Это одна фиксированная validation оценка на одном split. Она не характеризует
разброс между split/seed и не подтверждает качество на свежих финансовых
новостях. TF-IDF bag-of-n-grams учитывает только отдельные слова и соседние пары,
но не моделирует дальний порядок, контекст, смысл или временной сдвиг. Стандартный
tokenizer не создаёт самостоятельных признаков для пунктуации валют и `%`,
пропускает односимвольные токены и может разбивать сокращения с `n't`.

Held-out test содержит 690 строк и был использован только при построении и
проверке структуры group-aware manifest. Он не поступал в fit, transform,
predict, score, выбор параметров или таблицу метрик. Day04 может сравнивать
кандидаты по validation/CV согласно своему protocol; test следует сохранить до
заранее определённой финальной оценки. Повторное использование test для выбора
модели превратило бы его в selection data и сделало финальную оценку
оптимистично зависимой от решений.

## Воспроизведение и локальные outputs

Из корня проекта, без download:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.train_model --config configs/train_baseline.json
```

Локальные outputs: `data/processed/day03/splits.csv` и пять файлов в
`reports/day03/` — split summary, validation predictions, metrics, effective
config и provenance (всего вместе с manifest шесть результатов runner).
Provenance фиксирует dataset revision, входные checksums, policy, config,
seed, split algorithm, class mapping, vocabulary size и runtime versions.
Два последовательных CLI запуска в зафиксированном runtime дали побайтно
одинаковые outputs. Эти файлы игнорируются Git; их соответствие проверяется
ноутбуком и runner. Runtime versions из provenance: Python 3.14.7, pandas
3.0.6, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1.
