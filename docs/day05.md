# Day05 — validation error analysis

## D5-03: протокол ручного просмотра

Источник — проверенные Day02–Day04 artifacts и сохранённые validation predictions
`linear_svc`. Split, labels, preprocessing, candidate и параметры не изменялись.
Test examples/predictions/metrics не использовались; final evaluation остаётся D5-04.

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
предстоящий held-out test оценивает прежнее решение, не выбирает эти гипотезы.

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
