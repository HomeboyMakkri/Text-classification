# SPEC — Financial news sentiment classification

## 1. Назначение и scope

Учебный single-task mini-product для классификации англоязычных финансовых
предложений. Цель: понять ML workflow и научиться вести агентскую разработку
через явные контракты, небольшие изменения, проверки и воспроизводимые notebooks.

Путь: data → EDA → preprocessing → baseline → experiments → error analysis → inference.
Сейчас реализуется только **day01**. Transformer fine-tuning, API, UI, deployment,
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
