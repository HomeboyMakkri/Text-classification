# Day06 — сохранение модели и проверенная загрузка

D6-01…D6-03: frozen input loader, explicit train-only build, joblib bundle и
manifest. Raw-text batch API и CSV inference появятся в D6-04/D6-06.

## Подготовка artifact

Входы — принятые локальные Day02–Day05 artifacts; команды их подготовки и
frozen final evaluation описаны в `docs/day03.md`–`docs/day05.md`.
Использовать существующий `venv/`, `requirements-day03.txt` и dev requirements;
не обновлять runtime, чтобы обойти ошибку version mismatch.
Из корня проекта:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.inference_artifact
```

Команда `prepare_inference_artifact(root, directory=None)`:

1. Проверяет frozen inputs и reference прежнего решения.
2. Если artifact отсутствует, вызывает `build_inference_bundle(reference)`:
   fit только на прежнем clean train в manifest order, exact validation replay.
3. `save_bundle(bundle, directory, reference=reference)` проверяет fitted state,
   staging roundtrip и повторный validation replay загруженного объекта.
4. Публикует bundle и последним manifest. Test не предсказывается.

По умолчанию создаются ignored local files:

- `models/day06/sentiment_bundle.joblib`: fitted `tfidf` → `clf` Pipeline и
  typed metadata с frozen configuration/provenance/preprocessing/classes.
- `models/day06/manifest.json`: schema `day06-inference-v1`, checksum bundle,
  checksum fitted vocabulary/IDF/classes/weights и readable metadata.

Первый build делает явный fit; повтор той же команды с готовой согласованной
парой читает и проверяет её без fit/predict/joblib.dump и без изменения bytes.
Несовпадение inputs/config/runtime или повреждённая/неполная пара — ошибка,
не разрешение заново обучить модель или пересчитать test.

Для отдельной новой сборки указать **новый** output directory:

```bash
PYTHONPATH=src venv/bin/python -m finnews_sentiment.models.inference_artifact \
  --root . --output-dir models/day06_alternative_build
```

Относительный `--output-dir` CLI считается от `--root`. Python API принимает
явный `Path` output directory. Existing files не перезаписываются. Запись пары
не атомарна целиком: interruption после bundle, но до manifest оставит partial
output; его необходимо диагностировать отдельно. Checksum не содержит timestamp
или self-checksum; независимые builds не обязаны иметь одинаковые joblib bytes.

## Загрузка без обучения

```python
from pathlib import Path
from finnews_sentiment.models.inference_artifact import load_bundle

bundle = load_bundle(Path("models/day06/sentiment_bundle.joblib"))
print(bundle.metadata.contract.candidate_id)
print(bundle.metadata.n_train, bundle.metadata.n_features)
print(bundle.metadata.validation_replay)
```

Для Python/script нужен `src` в `PYTHONPATH`, как в команде выше.
`load_bundle` требует соседний `manifest.json`, совместимый src/runtime;
dataset, configs и reports не читает. До `joblib.load` проверяются schema,
metadata/preprocessing policy/implementation SHA-256, exact inference versions
и checksum bytes. Deserialization читает уже проверенные bytes из памяти.
После неё проверяются typed object/metadata agreement, Pipeline steps/types,
fitted state, classes `[0,1,2]`, vocabulary/IDF/weights dimensions, effective
parameters и checksum state. Никаких fit, predict или автоматического repair.

Joblib следует загружать только из доверенного локального источника:
pickle может исполнять код. SHA-256 обнаруживает порчу, но не удостоверяет
автора artifact. Точные inference versions: Python, NumPy, SciPy,
scikit-learn, joblib, threadpoolctl. Остальные frozen versions — provenance;
matplotlib не требуется для standalone load. Переносимость между версиями
не обещается. Preprocessing выполняется импортируемой `preprocess_text`,
а не сериализованной notebook lambda или `__main__` class.

## Notebook и границы ML

`notebooks/day06_inference.ipynb` исполняется из корня/`notebooks/` с чистого
kernel. D6-01 читает frozen references, D6-02 показывает сохранённую build
metadata, D6-03 — verified artifact load. Готовый notebook не обучает модель,
не вызывает predict и не пишет bundle/manifest. При отсутствии artifact
сначала отдельно выполнить build-команду; notebook не создаёт его автоматически.

Candidate/config/seed/row order/duplicates сохраняются по frozen protocol.
Текущий candidate — `linear_svc`, train 2210 rows / 2206 groups, sparse TF-IDF
`(2210, 5000)`; все 553 validation predictions воспроизводятся. Validation
не включается в train. Test не проходит новые transform/predict/score.
Replay подтверждает воспроизведение алгоритма/fit policy на данном validation,
а не идентичность утраченных весов Day05 или новое независимое качество.

Checkpoint D6-03 (2026-10-18): реальная сборка выполнила один train-only fit
и два exact validation replay (до save и после staging roundtrip).
Два prepare/load repeats с запрещёнными fit/predict/dump/CV/network/final-run
не изменили artifact bytes. Standalone load проверен в отдельном процессе:
читаются только bundle/manifest/preprocess source, matplotlib не импортируется.
Notebook выполнен из двух cwd в чистых kernels с запрещёнными model calls
и записью artifact; все 48 прежних input/final files и пара artifact неизменны.

Offline D6 reference/build/artifact tests: 86 passed, из них 29 новых artifact
tests. Type gate: 49 project files и 61 notebook modules (6 sequential +
55 cell-boundary), 0 errors/warnings. CLI Pyright не является полным запуском
Pylance внутри VS Code. Compileall и diff check успешны; полный pytest suite
в D6-03 не запускался. Local model/report artifacts не добавляются в Git.

Проверки fixtures отдельно от реального build/roundtrip и clean-kernel runs:

```bash
venv/bin/python -m pytest tests/test_day06_artifact.py tests/test_day06_build.py tests/test_day06_reference.py -v
venv/bin/python scripts/check_types.py
venv/bin/python -m compileall -q src tests
git diff --check
```
