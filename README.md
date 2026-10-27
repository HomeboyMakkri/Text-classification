# Text-classification — финансовая тональность

Учебный ML/NLP mini-product: классификация англоязычных финансовых предложений
как `negative`, `neutral` или `positive`. Полный путь:
**data → preprocessing → baseline → train-only CV → error analysis → frozen evaluation → inference**.
Переиспользуемая логика находится в `src/finnews_sentiment/`, notebooks объясняют
тот же код и показывают реальные результаты.

Day01–Day06 приняты. Итоговые результаты, demo и инструкция README подготовлены;
проверка training/inference/frozen flows в изолированных копиях D7-06 выполнена;
приёмка всего mini-product D7-07 пока pending. Текущий статус — [PLAN.md](PLAN.md),
контракты — [SPEC.md](SPEC.md), evidence — [docs/day07.md](docs/day07.md).

## Задача и данные

Источник — [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank),
только `sentences_75agree` (agreement ≥75%): 3453 предложения, 3448 exact-text
groups после принятой очистки. Agreement subsets перекрываются: не объединяем их.
Revision, archive SHA-256 и class order закреплены в [configs/dataset.json](configs/dataset.json).
Лицензия данных — **CC-BY-NC-SA-3.0**; Malo et al. (2014),
*Good debt or bad debt: Detecting semantic orientations in economic texts*.

Mapping всегда **`0=negative`, `1=neutral`, `2=positive`**. `minimal-v1`
нормализует HTML/entities, регистр и пробелы, сохраняя отрицания, числа,
проценты и валюты. TF-IDF vocabulary/IDF fit только на train, при CV — внутри
Pipeline каждого fold. Same-label duplicates сохранены в одной группе/split.
Seed=42: train 2210 строк / 2206 groups, validation 553 / 552, test 690 / 690.

## Окружение

Все команды ниже — **Bash, Linux/WSL, cwd=корень checkout**. Проверенный runtime:
Python **3.14.6**, зависимости из [requirements-day03.txt](requirements-day03.txt)
(включает [Day01 pins](requirements-day01.txt)). Полные версии и hashes —
[docs/day06.md](docs/day06.md). `requirements.txt` сохраняет исходный широкий
список учебных библиотек и не является проверенным runtime этого workflow.

**Если `venv/` уже существует, используйте его; не пересоздавайте и не
обновляйте зависимости для обхода version mismatch.** Для нового checkout
сначала выберите Python 3.14.6 как `python`, затем:

```bash
python --version
python -m venv venv
venv/bin/python -m pip install -r requirements-day03.txt
```

Команды создают `venv/` и устанавливают pinned packages; pip требует доступного
package index либо локального cache. Повтор создания окружения не нужен.
Для разработки/проверок дополнительно:

```bash
venv/bin/python -m pip install -r requirements-dev.txt
```

Новая установка dependencies здесь ещё не подтверждена; текущие проверки
используют существующий venv. У inference exact version/source checks:
несовместимость заканчивается ошибкой, не автоматическим refit.

## Быстрый inference с готовой моделью

Нужны совместимые `src/`, runtime, **trusted local pair**:

- `models/day06/sentiment_bundle.joblib` — fitted TF-IDF → LinearSVC Pipeline;
- `models/day06/manifest.json` — соседний manifest той же pair.

Эти files **не хранятся в Git**. Получите оба файла из доверенного принятого
Day06 build и разместите по этим путям. Если pair нет, раздел «Подготовка
artifact» ниже описывает явный build с обязательными frozen references.
Clone сам по себе не даёт готовый bundle или frozen reports; публичного
model download/автоматического восстановления нет. Joblib может исполнять код:
checksum проверяет целостность, но не доверенность автора.

Ordinary inference не читает dataset/configs/training reports и не делает fit.
Примеры [docs/day06_examples.csv](docs/day06_examples.csv) уже tracked, true labels
им не нужны. Для Python batch demo (только чтение и inference, без записи):

```bash
PYTHONPATH=src venv/bin/python - <<'PY'
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from finnews_sentiment.models.inference_artifact import load_bundle
from finnews_sentiment.models.predict import predict_texts

inputs = pd.read_csv("docs/day06_examples.csv", dtype=str, keep_default_na=False)
bundle = load_bundle(Path("models/day06/sentiment_bundle.joblib"))
predictions = predict_texts(bundle, inputs["text"].tolist())
output = pd.DataFrame([
    {"example_id": example_id, **asdict(item)}
    for example_id, item in zip(inputs["example_id"], predictions, strict=True)
])
print(output.to_string(index=False))
PY
```

API принимает непустой `Sequence[str]`; одна строка передаётся как `[text]`.
Typed record: `text`, `text_clean`, integer `pred_label_id`, `pred_label`.
Порядок/duplicates сохраняются, probabilities/confidence не возвращаются.

CSV CLI сохраняет исходные columns/строковые значения и добавляет те же три
prediction columns. Обязательна непустая `text`; true labels не требуются.
Создайте новый output path (последняя команда показывает результат):

```bash
readme_demo_dir=$(mktemp -d)
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.predict \
  --model models/day06/sentiment_bundle.joblib \
  --input docs/day06_examples.csv --output "$readme_demo_dir/predictions.csv"
cat "$readme_demo_dir/predictions.csv"
```

Для своего CSV замените только `--input` и новый `--output`. В примере input
`(8, 2)` → output `(8, 5)`; raw `(8,)` → fitted TF-IDF `(8, 5000)` → labels `(8,)`.
Повтор с тем же output path откажет в перезаписи; для повтора нужен новый path.
Missing/bad pair, неверный CSV, blank text, reserved prediction columns и
совпадающие input/output paths — error (CLI exit 2), без скрытого build или
частичного output. Полный I/O/errors contract — [Day06](docs/day06.md).

Demo responses: `01/04/05 → positive`, остальные `neutral`; OOV `08` имеет
нулевой TF-IDF row и штатный label. Это не gold labels или measurement качества.
Python/API/CLI parity, raw→clean и ограничения показаны в [Day07 demo](docs/day07.md)
и [notebook](notebooks/day07_review.ipynb). Input фразы не менялись ради ответов.

## Подготовка данных и training

Это **отдельный flow**, не prerequisite обычного inference с готовой pair.
Для training/replay используйте отдельную копию checkout с локальным raw
источником. Подготовка/обучение пишут outputs; не смешивайте свежие reports
с принятыми frozen references в текущем checkout.

В Git уже tracked immutable `data/raw/FinancialPhraseBank-v1.0.zip` и
`data/raw/source.json`. `.gitignore` не исключает ранее tracked files.
Canonical/processed CSV, splits и generated reports local/ignored; exceptions —
tracked `reports/day03/observations.md` и `reports/day04/observations.md`.

С существующим pinned archive выполнить **offline**, без training:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.data.prepare_phrasebank --root .
PYTHONPATH=src venv/bin/python -m finnews_sentiment.data.prepare_day02 --root .
```

Первая команда проверяет hash и пишет canonical CSV
`data/interim/financial_phrasebank_75agree.csv`; вторая проверяет archive/CSV,
пишет `data/processed/financial_phrasebank_75agree.csv` и
`reports/day02/{preprocessing,duplicates,provenance}.json`. Повтор явно
пересоздаёт эти outputs; raw не меняется. Missing text/labels/empty-after-clean
отражаются в отчёте, conflicting labels блокируют подготовку.

**Только если архива нет**, download — отдельная явная команда; при наличии
локального архива она использует его и не скачивает заново:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.data.prepare_phrasebank --root . --download
```

Затем повторите Day02 preparation. NLTK resources не нужны. Подробности —
[Day01](docs/day01.md) и [Day02](docs/day02.md).

На готовых Day02 processed/provenance запустить baseline, затем fixed CV:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.train_model \
  --root . --config configs/train_baseline.json
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.run_experiments_day04 \
  --root . --config configs/experiments_day04.json
```

Day03 создаёт `data/processed/day03/splits.csv`, baseline/Dummy validation
predictions/metrics/provenance в `reports/day03/`. Day04 использует прежние
splits, создаёт train-only folds/CV/selection и validation confirmation в
`reports/day04/`. Оба запускают **fit и запись**; повтор снова обучает
фиксированные configurations и пересоздаёт outputs, не является cache-only.
Test estimator не получает. Это воспроизведение прежних экспериментов,
не разрешение нового tuning по уже просмотренным validation/test результатам.
Точные protocols — [Day03](docs/day03.md), [Day04](docs/day04.md).

## Frozen evaluation, report и artifact

Day03/Day04 сохраняют predictions/configs, но не fitted inference Pipeline.
Для следующих команд нужны **уже принятые совместимые frozen Day02–Day05
inputs/reports**, включая completed `reports/day05/final_evaluation.json`,
`frozen_protocol.json` и связанные outputs/provenance. Одного нового training
run до Day04 недостаточно для Day06 build. Получите/сохраните полный snapshot
прежнего принятого workflow; hashes/versions/IDs должны совпадать.

Первый held-out test run — отдельный исторический D5-04 после freeze/replay,
описанный в [Day05](docs/day05.md). Он не часть quickstart или автоматический
fallback. При отсутствии/partial/mismatched cache остановитесь и проверьте
inputs/protocol/hashes/runtime; cache не удаляется и test не переоценивается
ради запуска инструкции. Для read-only проверки существующего результата:

```bash
PYTHONPATH=src venv/bin/python - <<'PY'
from pathlib import Path
from finnews_sentiment.models.final_evaluation import load_final_evaluation

cached = load_final_evaluation(Path.cwd())
print(cached.cache_status, cached.candidate_id)
print(cached.confusion.metrics)
PY
```

Completed cache возвращает `reused`, проверяет сохранённые labels/metrics без
fit/predict/write. При наличии того же cache и tracked manual annotations
отдельный report-only runner пересобирает `reports/day05/` (явная запись):

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.run_error_analysis_day05 --root .
```

Он не обучает и не предсказывает; повтор проверяет inputs и воспроизводит report,
не заменяет frozen final cache. Отдельная команда подготовки inference pair:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.inference_artifact --root .
```

При отсутствии pair эта команда явно fit прежний train-only TF-IDF/LinearSVC,
проверяет exact validation replay и serialization, публикует
`models/day06/{sentiment_bundle.joblib,manifest.json}`. При complete pair —
checked reuse без fit/predict/write. Partial/bad pair даёт error, без удаления
или автоматического overwrite. Validation/test не добавляются к train;
новых test predictions нет. Build не вызывается из ordinary inference.

Отдельная расширенная consistency проверка требует references и готовую pair:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.inference_consistency --root .
```

Она проверяет train/validation raw-clean parity, предсказывает на validation и
новых standalone examples, при необходимости пишет `reports/day06/consistency.json`.
Fit/test prediction отсутствуют. Это повтор проверки согласованности,
не новая независимая оценка качества. Полные prerequisites/reuse и trusted-local
ограничения — [Day06](docs/day06.md), [SPEC §11](SPEC.md).

## Результаты и error analysis

Сводка прежних проверенных измерений; CV выбирал candidate до validation.
Все values округлены до шести знаков; N в CV row — весь train pool.

| Протокол | Модель | N | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| Validation | Dummy most_frequent | 553 | 0.622061 | 0.255667 |
| Validation | TF-IDF + LogisticRegression baseline | 553 | 0.813743 | 0.720965 |
| Train-only CV, mean 3 folds | TF-IDF + LinearSVC | 2210 | 0.836660 | 0.768715 |
| Validation confirmation | TF-IDF + LinearSVC | 553 | 0.848101 | 0.805961 |
| Frozen test | TF-IDF + LinearSVC | 690 | 0.856522 | 0.797050 |

Три направления улучшения: word unigrams; замена LogisticRegression на
LinearSVC; grid LogisticRegression C={0.1,1,10}. Шесть CV rows / пять уникальных
configurations. Выбор — maximum mean CV macro F1, exact ties по frozen order.
У LinearSVC CV std macro F1=0.024064 (ddof=0), это не confidence interval.
Validation macro F1 delta к baseline: **+0.084995 / +11.79%** относительно
baseline, рассчитана до округления. Baseline/Dummy **не оценивались на test**;
разные validation/test строки не дают paired сравнения.

Validation error analysis: 84 errors / 553 строки (84 error groups, всего
552 groups); confusion matrix `(3,3)`, true rows/predicted columns. До трёх
минимальных source IDs на направленную пару дали 18 reviewed rows / 18 groups.
Financial context 12/18 и negation/comparison 3/18 относятся только к reviewed
sample, не всем errors и не доказанным причинам. На test negative recall
48/84=0.571429 при accuracy 0.856522: class imbalance требует per-class metrics.
Подробные CV/class tables, denominators, observations/hypotheses и source
SHA-256 — [docs/day07_results.md](docs/day07_results.md),
annotations — [docs/day05_failure_cases.csv](docs/day05_failure_cases.csv).

## Структура и notebooks

| Путь | Назначение |
| --- | --- |
| [src/finnews_sentiment](src/finnews_sentiment/) | Reusable data/features/models/visualization logic |
| [configs](configs/) | Pinned source, group split/baseline, fixed CV candidates |
| [docs/day07.md](docs/day07.md) | Карта entrypoints, output storage, demo и проверенные checkpoints |
| [tests](tests/) / [scripts/check_types.py](scripts/check_types.py) | Offline contracts / source и notebook type gate |
| `data/`, `models/`, `reports/` | Source/generated artifacts с описанными tracking exceptions |
| `.cache/` | Локальные temporary outputs/evidence, не runtime input |

| День | Notebook | Инструкция |
| --- | --- | --- |
| 01 | [EDA](notebooks/day01_eda.ipynb) | [Day01](docs/day01.md) |
| 02 | [Preprocessing](notebooks/day02_preprocessing.ipynb) | [Day02](docs/day02.md) |
| 03 | [Baseline](notebooks/day03_baseline.ipynb) | [Day03](docs/day03.md) |
| 04 | [Experiments](notebooks/day04_experiments.ipynb) | [Day04](docs/day04.md) |
| 05 | [Error analysis](notebooks/day05_error_analysis.ipynb) | [Day05](docs/day05.md) |
| 06 | [Inference](notebooks/day06_inference.ipynb) | [Day06](docs/day06.md) |
| 07 | [Review/demo](notebooks/day07_review.ipynb) | [Day07](docs/day07.md) |

При наличии локальных prerequisites зарегистрируйте kernel перед запуском:

```bash
venv/bin/python -m ipykernel install --prefix=venv --name=text-classification --display-name="Python (Text-classification venv)"
venv/bin/jupyter notebook notebooks/day07_review.ipynb
```

Выберите kernel **Python (Text-classification venv)**, Run All. Day07 читает
принятые references/cache и выполняет demo; не делает fit/CV/download/test
prediction. Day01–Day04 notebooks могут явно создавать outputs/обучать;
их повтор не является default inference flow. Root/src setup поддерживает cwd
root и `notebooks/`. Для автоматического прогона Day07 с отдельным output:

```bash
mkdir -p .cache/day07-checkpoints
venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=text-classification --ExecutePreprocessor.timeout=180 --output-dir=.cache/day07-checkpoints --output=readme-review.ipynb notebooks/day07_review.ipynb
```

Повтор использует существующую модель, пишет только temporary demo CSV и
указанный output notebook; основной notebook не перезаписывает. Не запускайте
параллельно перезапись artifacts, на которых работает пользовательский kernel.

## Проверки и границы воспроизведения

С установленными dev requirements доступны следующие проверки из корня:

```bash
venv/bin/python scripts/check_types.py
venv/bin/python -m pytest tests -v
venv/bin/python -m compileall -q src tests scripts
git diff --check
```

Type gate проверяет Python и sequential/cell-boundary exports всех notebooks;
CLI Pyright не является полным запуском VS Code Pylance. Offline tests с
fixtures, real-data replay, clean-kernel notebook и fresh dependency install —
отдельные свидетельства. D7-05 проверил README по коду, путям и результатам;
D7-06 исполнил offline training, inference и frozen reuse в отдельных копиях
с existing venv: 20 training outputs совпали побайтно, исходные artifacts
сохранены. Протокол — [docs/day07.md](docs/day07.md#d7-06-проверка-readme-в-изолированных-копиях).
Fresh install/download и другие платформы не проверялись; новый Day06 build
не повторялся (evidence D6-02/D6-03). Полная приёмка **D7-07** пока pending.

## Ограничения

- Английские финансовые предложения исторического agreement subset не
  гарантируют качество на live news, другом domain/time period или языке.
- Один group split, три folds и похожие тексты ограничивают статистические
  выводы. Validation уже просмотрена, test открыт; нельзя улучшать модель
  по test и выдавать тот же test за независимую проверку нового решения.
- Bag-of-ngrams ограничивает контекст, отрицания/сравнения и OOV. Class imbalance
  требует per-class recall/F1; unlabeled demo не измеряет качество.
- LinearSVC не возвращает probabilities; score не confidence. Joblib — только
  trusted local, exact runtime/source compatibility; checksum не удостоверяет автора.
- Тональность предложения не является прогнозом доходности или торговой рекомендацией.

Правила работы с проектом — [AGENTS.md](AGENTS.md). Ниже сохранены исходные
материалы задания. Их учебные заготовки описывают первоначальные варианты
(включая устаревшие preprocessing/train examples), а не актуальные команды
запуска. Для работы используйте инструкцию выше и согласованный SPEC.

---

<b>Финансовые новости — анализ тональности для трейдеров</b>

<b>НЕОБХОДИМО ДЛЯ НАЧАЛА РАБОТЫ:</b>

Вам понадобится датасет с финансовыми новостями и разметкой тональности.

<b>Откуда взять данные:</b>
1. Kaggle: "Financial Sentiment Analysis" или "Stock News Sentiment"
2. Hugging Face: dataset "financial_phrasebank"
3. Или создайте свой датасет из новостных сайтов

<b>Формат данных (CSV файл):</b>
Должны быть минимум две колонки:
• <code>text</code> — текст новости
• <code>sentiment</code> — тональность (positive/negative/neutral или цифры 0/1/2)

Пример:
<code>text,sentiment</code>
<code>"Apple stocks rose 5% after earnings report",positive</code>
<code>"Company announces massive layoffs",negative</code>
<code>"Market remains stable despite volatility",neutral</code>

---

<b>ЧАСТЬ 1: EDA и разбор данных</b>

<b>1.1 Загрузка и первичный анализ</b>

1. Создайте новую папку для проекта

2. Создайте Jupyter ноутбук или Python скрипт

3. Импортируйте библиотеки:
<code>import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns</code>

4. Загрузите ваш CSV файл:
<code>df = pd.read_csv('ваш_файл.csv')</code>

5. Если колонки называются иначе — переименуйте их:
<code>df = df.rename(columns={'headline': 'text', 'label': 'sentiment'})</code>

6. Выведите размерность: <code>print(df.shape)</code>

7. Выведите первые 5 строк: <code>print(df.head())</code>

8. Выведите информацию о типах данных: <code>print(df.info())</code>

9. Проверьте пропуски: <code>print(df.isnull().sum())</code>

10. Удалите строки с пропусками в тексте: <code>df = df.dropna(subset=['text'])</code>

11. Постройте график распределения классов:
<code>sns.countplot(data=df, x='sentiment')
plt.title('Распределение классов')
plt.savefig('sentiment_distribution.png')</code>

<b>1.2 Анализ длин текстов</b>

1. Создайте колонку с длиной текста:
<code>df['text_length'] = df['text'].str.len()</code>

2. Создайте колонку с количеством слов:
<code>df['word_count'] = df['text'].str.split().str.len()</code>

3. Постройте гистограмму длин:
<code>df['text_length'].hist(bins=50)
plt.xlabel('Длина текста')
plt.ylabel('Количество')
plt.savefig('text_length_distribution.png')</code>

4. Выведите примеры для каждого класса:
<code>for sentiment in df['sentiment'].unique():
    print(f'\n=== Класс {sentiment} ===')
    print(df[df['sentiment'] == sentiment]['text'].iloc[0])</code>

<b>ЧАСТЬ 2: Подготовка данных</b>

<b>2.1 Очистка текста</b>

1. Напишите функцию очистки:
<code>import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text</code>

2. Примените к данным:
<code>df['text_clean'] = df['text'].apply(clean_text)</code>

3. Удалите пустые тексты после очистки:
<code>df = df[df['text_clean'].str.len() > 0]</code>

<b>2.2 Признаки</b>

1. Добавьте колонки с признаками:
<code>df['word_count_clean'] = df['text_clean'].str.split().str.len()
df['char_count'] = df['text_clean'].str.len()
df['dollar_count'] = df['text_clean'].str.count('$')</code>

2. Преобразуйте метки классов в цифры, если они текстовые:
<code>df['label'] = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
# или используйте LabelEncoder из sklearn</code>

<b>ЧАСТЬ 3: Baseline модель</b>

<b>3.1 Подготовка признаков</b>

1. Импортируйте нужное:
<code>from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split</code>

2. Создайте TF-IDF векторизатор:
<code>vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))</code>

3. Преобразуйте тексты:
<code>X = vectorizer.fit_transform(df['text_clean'])
y = df['label'].values</code>

4. Разделите на train/test:
<code>X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)</code>

<b>3.2 Обучение baseline</b>

1. Импортируйте и создайте модель:
<code>from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

model = LogisticRegression(max_iter=200, n_jobs=-1)</code>

2. Обучите:
<code>model.fit(X_train, y_train)</code>

3. Сделайте предсказания:
<code>y_pred = model.predict(X_test)</code>

4. Выведите отчет:
<code>print(classification_report(y_test, y_pred))</code>

5. Посчитайте macro F1:
<code>f1 = f1_score(y_test, y_pred, average='macro')
print(f'Macro F1: {f1:.4f}')</code>

6. Сохраните baseline результат:
<code>baseline_f1 = f1
with open('baseline_result.txt', 'w') as f:
    f.write(f'Baseline Macro F1: {f1:.4f}')</code>

<b>ЧАСТЬ 4: Улучшение модели</b>

<b>Цель:</b> улучшить macro F1 минимум на 5%

<b>4.1 Выберите подходы (минимум 3):</b>

<b>Подход 1 — Лемматизация:</b>
<code>import spacy
nlp = spacy.load('en_core_web_sm')

def lemmatize(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc])

df['text_lemm'] = df['text_clean'].apply(lemmatize)</code>

<b>Подход 2 — TF-IDF с триграммами:</b>
<code>vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))</code>

<b>Подход 3 — Другая модель (Random Forest):</b>
<code>from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, n_jobs=-1)</code>

<b>Подход 4 — Другая модель (LinearSVC):</b>
<code>from sklearn.svm import LinearSVC
model = LinearSVC(max_iter=2000)</code>

<b>Подход 5 — Подбор гиперпараметров:</b>
<code>from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=3)
grid.fit(X_train, y_train)
print(f'Лучший C: {grid.best_params_}')</code>

<b>4.2 Реализуйте и протестируйте</b>

1. Для каждого подхода:
   - Подготовьте признаки (если нужно)
   - Обучите модель
   - Посчитайте macro F1
   - Сравните с baseline

2. Сохраняйте лучшие модели:
<code>import joblib
joblib.dump(model, 'best_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')</code>

<b>ЧАСТЬ 5: Сравнение и анализ</b>

<b>5.1 Таблица результатов</b>

Создайте текстовый файл с таблицей:

| Метод | Macro F1 | Улучшение |
|-------|----------|-----------|
| Baseline TF-IDF + LogReg | 0.XX | - |
| [Ваш метод 1] | 0.XX | +X% |
| [Ваш метод 2] | 0.XX | +X% |

<b>5.2 Confusion Matrix</b>

1. Постройте матрицу ошибок:
<code>from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')</code>

<b>5.3 Анализ ошибок</b>

1. Найдите ошибки и сохраните оригинальные тексты до векторизации

2. Выведите несколько примеров ошибок

3. Сохраните анализ в файл <code>error_analysis.txt</code>

<b>ЧАСТЬ 6: Предсказание на новых данных</b>

<b>6.1 Функция для предсказания</b>

1. Напишите функцию:
<code>def predict_sentiment(texts, model, vectorizer, clean_func=None):
    if isinstance(texts, str):
        texts = [texts]
    if clean_func:
        texts = [clean_func(t) for t in texts]
    X = vectorizer.transform(texts)
    predictions = model.predict(X)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
        return predictions, probs
    return predictions</code>

2. Протестируйте на 3-5 новых примерах и выведите результаты

<b>ЧТО СДАТЬ:</b>
• Jupyter ноутбук или Python скрипты с кодом
• Графики EDA (распределение классов, длины текстов)
• Файл с таблицей сравнения методов
• Confusion Matrix график
• Файл с анализом ошибок
• Обученная лучшая модель

---
<i>Удачи с реализацией!</i> 🚀
