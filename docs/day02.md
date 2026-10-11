# Запуск Day02

Используйте существующий `venv/`. Все команды ниже выполняются из корня проекта.
Day02 использует runtime `requirements-day01.txt`; dev-проверки —
`requirements-dev.txt`. NLTK resources для принятого preprocessing не нужны.

При отсутствии исходника подготовьте его отдельной командой:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.data.prepare_phrasebank --download
```

Для offline подготовки processed dataset и отчётов:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.data.prepare_day02
```

Команда проверяет архив и canonical CSV, очищает тексты, исключает непригодные
строки с отчётом, добавляет fixed class IDs и диагностирует дубликаты.
Conflicting labels блокируют запись. Повторы с одной меткой сохраняются.
Raw/interim не изменяются; network и обучение не используются.
При отсутствии архива команда завершится ошибкой, даже если canonical CSV есть:
архив требуется для подтверждения provenance.

Выходы: `data/processed/financial_phrasebank_75agree.csv` и
`reports/day02/{preprocessing,duplicates,provenance}.json`.
Артефакты локальные, не включаются в Git. Повторный запуск при тех же входах
и runtime даёт одинаковые bytes; новые версия policy/runtime могут изменить результат.
Для чтения CSV сохраняйте строковые поля и literal NA/nan:

```python
processed = pd.read_csv(
    path, encoding="utf-8", keep_default_na=False,
    dtype={"source_row_id": "int64", "text": "str", "sentiment": "str",
           "text_clean": "str", "label_id": "int64"},
)
```

Откройте `notebooks/day02_preprocessing.ipynb`, выберите kernel
**Python (Text-classification venv)** и выполните Restart Kernel → Run All.
Setup работает из корня и `notebooks/`; раздел D2-05 явно пересоздаёт outputs.

```bash
venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=text-classification notebooks/day02_preprocessing.ipynb
venv/bin/python scripts/check_types.py
MPLCONFIGDIR=/tmp/text-classification-mpl venv/bin/python -m pytest tests -v
venv/bin/python -m compileall -q src tests
git diff --check
```

Из `notebooks/` offline CLI запускается так:

```bash
PYTHONPATH=../src ../venv/bin/python -m finnews_sentiment.data.prepare_day02
```

Day02 завершён 2026-10-11: notebook содержит D2-01…D2-06 и сохранённые outputs.
Раздел D2-06 создаёт `reports/day02/observations.md` с фактическими выводами,
ручным разбором девяти реальных предложений и ограничениями policy.
CLI создаёт только CSV и три JSON; observations воспроизводится запуском notebook.

Проверки приёмки: pytest — 53 passed, 46 subtests passed; Pyright sources и
notebook exports — 0 errors/warnings; compileall и diff check passed.
Notebook выполнен в двух чистых venv kernels (cwd=root и cwd=notebooks):
по 9/9 code cells, 0 errors. Sandbox запрещает необходимые kernel sockets;
проверенный запуск потребовал разрешённого выхода из sandbox.
Все пять локальных артефактов совпали по bytes между двумя прогонами.

Итоговый dataset `(3453, 5)`, excluded=0, changed retained texts=3447;
class counts 420/2146/887 сохранены. По 5 raw/clean duplicate groups,
новых совпадений и conflicts нет. Ручная проверка не доказывает семантическую
корректность всего корпуса; training и оценка macro F1 ещё не выполнялись.
Day03 требует группового split и fit TF-IDF только на train.
