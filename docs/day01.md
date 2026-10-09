# Запуск day01

Все команды выполняются из корня проекта. Текущий day01 проверен на Python 3.14.6
и версиях `requirements-day01.txt`. Существующее окружение `venv/` уже содержит их.

Для новой копии проекта создайте окружение совместимым Python и установите только
зависимости текущего дня:

```bash
python -m venv venv
venv/bin/python -m pip install -r requirements-day01.txt
```

Подготовьте данные. Первый запуск требует network и скачивает около 682 kB;
последующие используют локальный архив, проверяют SHA-256 и пересоздают canonical CSV.
Параметры источника сохранены в `configs/dataset.json`.

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.data.prepare_phrasebank --download
```

Откройте notebook из interpreter проекта:

```bash
venv/bin/python -m ipykernel install --prefix=venv --name=text-classification --display-name="Python (Text-classification venv)"
venv/bin/jupyter notebook notebooks/day01_eda.ipynb
```

Выберите kernel **Python (Text-classification venv)** и выполните
**Restart Kernel → Run All**. Первая ячейка показывает `sys.executable`; он должен
указывать на `venv/bin/python` проекта. Корень находится автоматически при запуске
из корня или `notebooks/`. Notebook не делает download и не обучает модель.

Для автоматического чистого прогона после регистрации kernel:

```bash
venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=text-classification notebooks/day01_eda.ipynb
```

Получите результат в `reports/day01/`: качество данных, классы, длины, два PNG,
наблюдения и версии runtime. Raw data и reports исключены из Git; проверенный
notebook с outputs сохраняется в репозитории. Успешный запуск не зависит от
результатов предыдущего kernel.

Проверка типов (Pyright + pandas-stubs):

```bash
venv/bin/python -m pip install -r requirements-dev.txt
venv/bin/python scripts/check_types.py
```

Команда проверяет `src`, `tests`, `scripts` и код всех notebooks в порядке ячеек.
Временные `.py` создаются в `.cache/` и удаляются после проверки. Диагностика
показывает имя notebook и строку экспорта; комментарии отмечают номера ячеек.
Для Pylance выберите interpreter `venv/bin/python` через **Python: Select Interpreter**.
Notebook kernel тоже должен использовать это окружение.

Offline проверки:

```bash
MPLCONFIGDIR=/tmp/text-classification-mpl venv/bin/python -m unittest discover -s tests -v
venv/bin/python -m compileall -q src tests
git diff --check
```

Данные: [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank),
`sentences_75agree`; Malo et al. (2014); CC-BY-NC-SA-3.0.
Подмножества по agreement перекрываются; в проекте используется только одно.

Следующий учебный пункт: D2-01 в PLAN. Перед реализацией сформулируем, какую
информацию может сохранить или потерять каждая операция очистки.
