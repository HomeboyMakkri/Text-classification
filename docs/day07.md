# Day07 — аудит handoff и read-only checkpoint

## D7-01: результат и границы

Проверено 2026-10-25, после принятого D6-08. D7-01 завершает аудит входов,
а не финализацию всего mini-product. D7-02…D7-07 остаются pending.
Контракт — [SPEC §12](../SPEC.md), порядок — [PLAN](../PLAN.md).
Checkpoint — [day07_review.ipynb](../notebooks/day07_review.ipynb), один раздел
D7-01 и пять code cells с сохранёнными outputs.

Вызваны существующие `load_inference_reference(ROOT)` и
`load_bundle(ROOT / "models/day06/sentiment_bundle.joblib")` из
`finnews_sentiment.models.inference_artifact`. Reference reader проверяет
processed/provenance, manifest/IDs/order/labels/groups, Day04 selection,
полные параметры, runtime и completed final cache. Bundle reader проверяет
manifest, checksum, preprocessing implementation, типы fitted components,
размерности и state hash. Contracts reference/bundle и counts совпали.
Наличие файла отдельно от этих проверок не считается приёмкой provenance.

| Проверенный объект | Результат |
| --- | --- |
| Candidate / fit policy / seed | `linear_svc` / `day03_train_only` / 42 |
| Preprocessing | `minimal-v1` |
| Mapping | `0=negative`, `1=neutral`, `2=positive` |
| Train DataFrame / exact-text groups | `(2210, 5)` / 2206 |
| Validation DataFrame / groups | `(553, 5)` / 552 |
| Saved validation predictions | `(553, 3)`, IDs/order/truth согласованы |
| Test manifest | 690 rows / 690 groups; новых model calls нет |
| Fitted vocabulary size | 5000; TF-IDF transform не вызывался |
| Frozen input/final checksums | 28 проверены reference reader |
| Consistency input/source checksum entries | 30 + 26, сверены с текущими bytes |
| Demo inputs / saved outputs | `(8, 2)` / `(8, 5)`, example IDs/text/order совпали |
| Required / optional inventory | 53 / 10, missing required=0, optional=0 |

`exact_match` replay и standalone `passed` прочитаны из accepted Day06 metadata
и consistency report. В D7-01 они **не выполнялись заново**. Consistency report
согласован с текущими source/input hashes, runtime и checked bundle state.
Выходы demo проверены как сохранённые таблицы; новой prediction/оценки качества
demo нет. Готовая модель загружена только как trusted local joblib;
checksums не устанавливают доверенность автора.

## Inventory и зависимости запуска

Ниже — inventory существующих entrypoints, а не готовый итоговый quickstart.
Полная инструкция и проверка исполнения её команд относятся к D7-05/D7-06.
При отсутствии frozen inputs audit завершается ошибкой без preparation/rebuild.

| Этап | Существующий entrypoint | Входы → выходы / effect |
| --- | --- | --- |
| Source preparation | `data.prepare_phrasebank` | Config + pinned ZIP → canonical CSV; network только отдельным `--download` при отсутствии архива |
| Day02 | `data.prepare_day02` | Canonical source → processed CSV + preprocessing/duplicates/provenance reports; запись |
| Day03 | `models.train_model --config configs/train_baseline.json` | Day02 → train-only baseline, split manifest, validation predictions/metrics; fit и запись |
| Day04 | `models.run_experiments_day04` | Day02/Day03 → fixed train-only CV, selection, validation confirmation; fit и запись |
| Day05 read-only | `models.final_evaluation.load_final_evaluation(ROOT)` | Frozen references + completed final cache → checked cached result; без fit/predict/write |
| Day05 report | `models.run_error_analysis_day05` | Checked references + annotations + completed final cache → report; запись, без новых predictions |
| Day06 build/reuse | `models.inference_artifact` | Frozen Day02–Day05 inputs → bundle/manifest; absent pair делает явный train-only fit, complete pair проверяется без fit |
| Day06 consistency | `models.inference_consistency` | References + bundle → validation/API/standalone parity report; predict и возможная запись, поэтому D7-01 не вызывает |
| Ordinary inference | `models.predict --model … --input … --output …` | Compatible src/runtime + trusted pair + новый CSV → predictions; без fit, output должен отсутствовать |

Module CLI использует префикс `finnews_sentiment.` и `PYTHONPATH=src` из корня.
Raw input → `minimal-v1` → fitted TF-IDF sparse `(N, V)` → classifier → labels
`(N,)`; learned vocabulary/IDF fit только на train. Ordinary inference не требует
dataset/configs/reports; явный artifact build требует frozen references.
`run_final_evaluation` не является reader или способом получить Pipeline:
при первом запуске он оценивает held-out test, в D7-01 он запрещён.

Подробные существующие инструкции: [Day01](day01.md), [Day02](day02.md),
[Day03](day03.md), [Day04](day04.md), [Day05](day05.md), [Day06](day06.md).
При missing/partial cache либо несовпадении hashes/runtime сначала нужна
диагностика; автоматические deletion/download/refit/final-run не выполняются.

Текущий audit требует processed data, Day02–Day05 reports, annotations,
bundle/manifest, Day06 consistency/demo и pinned configs/runtime. Raw archive,
canonical CSV и Day01 EDA outputs optional для read-only audit; для отдельных
сценариев воспроизведения они могут стать обязательными. Полный список и
фактические SHA-256 — в приложении и notebook outputs.

Git storage проверен через `git ls-files`, а не только `.gitignore`:
required inventory содержит 10 tracked files; остальные 43 — local/ignored.
Из 10 optional paths два tracked, восемь local/ignored. Generated bundle,
processed CSV и final cache не включены в Git; текущая рабочая копия их имеет.

## Runtime и ключевые fingerprints

| Component | Verified version |
| --- | --- |
| Python | 3.14.6 |
| pandas | 3.0.6 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |
| threadpoolctl | 3.7.0 |
| matplotlib | 3.11.2 |

Fitted state SHA-256:
`7623ab0d4d555aed3c73944f59f85acb23e141ada2794fa826c9d7231383b7f7`.
Bundle/manifest/file hashes указаны в приложении. Это fingerprints текущего
accepted snapshot, не обещание одинаковых pickle bytes между независимыми builds.

## Наблюдаемые несоответствия и дальнейшие границы

1. **Raw ZIP и source metadata уже tracked.** `git ls-files --stage -- data`
   показывает `data/raw/FinancialPhraseBank-v1.0.zip` и `data/raw/source.json`.
   `.gitignore` с `/data/` не исключает ранее tracked files. SPEC §3 описывает
   архив как не включённый в Git: описание расходится с текущим index.
   D7-01 фиксирует фактическое состояние; archive/index не менялись.
   Согласование документации относится к D7-02.
2. README вводно описывает только Day01; итоговый README — D7-05.
3. SPEC §5 сохраняет историческую карту с templates. Актуализация — D7-02.
4. Day06 demo/consistency сохранены, но новый demo и полноценное воспроизведение
   training/inference здесь не запускаются: D7-04 и D7-06.

Missing inputs не обнаружены. Найденные расхождения документации не нарушают
проверенные data/model contracts. Production modules, configs, data и модель
не исправлялись. Новых ML conclusions, качества на live news и нового replay
по этому аудиту не заявляется.

## Проверки D7-01

| Проверка | Фактический результат |
| --- | --- |
| Целевые offline tests | 65 passed, 14.53 s |
| CLI type gate: project | 53 files, 0 errors/warnings |
| CLI type gate: notebooks | 7 sequential exports + 70 cell-boundary modules = 77 files, 0 errors/warnings |
| Clean kernel, cwd=root | 5/5 code cells, 0 errors, 2.206 s |
| Clean kernel, cwd=notebooks | 5/5 code cells, 0 errors, 2.183 s |
| Protected snapshot | 106 files (source/config/data/reports/model/прежние docs/notebooks/runtime contracts), SHA-256 без изменений |
| Notebook inventory snapshot | 63 paths, SHA-256 до/после совпали |
| Compileall / diff check | passed |

Команды из корня, выполненные на текущем `venv/`:

```bash
venv/bin/python scripts/check_types.py
venv/bin/python -m pytest tests/test_day06_reference.py tests/test_day06_artifact.py tests/test_check_types.py -v
venv/bin/python -m compileall -q src tests scripts
git diff --check
```

Целевые tests проверяют existing reference/bundle failure boundaries и type-gate
coverage, включая ошибки между cells. Они используют fixtures и могут обучать
маленькие fixture models; это не новое обучение на Financial PhraseBank.
Новых tests для Markdown или повторения existing validators не добавлялось.
CLI Pyright не является полным запуском Pylance внутри VS Code.

Два исполнения через nbclient, kernel `text-classification`, использовали
отдельные kernels и временный guard prelude. Guard запрещал model fit,
transform/predict/score, artifact build/save, consistency runner, CV runner,
final evaluation, joblib dump, Python file writes и network connect/urlopen.
В каждом kernel три пробных вызова (fit/write/network) были заблокированы;
дальше при исполнении notebook — **0 forbidden calls**. Пробные вызовы
останавливались mocks до реальной операции. Проверка state/model hashes
выполнялась без estimator transform/predict. Notebook сохранён с outputs
основных пяти cells, guard prelude в учебный код не включён.

Локальные evidence под `.cache/day07-checkpoints/` (ignored):
`before.json`, `audit_initial.json`, `inventory.json`, `types.log`, `pytest.log`,
`from-root.ipynb`, `from-notebooks.ipynb`, `execution.json`, `execution.log`,
`final_audit.json`;
генератор и guarded runner — `create_checkpoint.py`, `execute_checkpoint.py`.
Фактический запуск runner:
`venv/bin/python .cache/day07-checkpoints/execute_checkpoint.py`.
Snapshots/logs относятся к этому checkout и не являются входами обычного
Run All notebook.

Для обычного самостоятельного повторения notebook с отдельным output:

```bash
mkdir -p .cache/day07-checkpoints
venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=text-classification --ExecutePreprocessor.timeout=180 --output-dir=.cache/day07-checkpoints --output=manual-from-root.ipynb notebooks/day07_review.ipynb
```

Из `notebooks/` использовать `../venv/bin/jupyter`, notebook `day07_review.ipynb`
и output-dir `../.cache/day07-checkpoints`. Локальный kernel уже зарегистрирован;
новое окружение/установка packages здесь не проверялись. Полный pytest suite,
новое training replay и inference demo не запускались: их границы заданы
D7-04/D7-06/D7-07. Предыдущие notebooks и пользовательские kernels не менялись.

## Приложение: inventory SHA-256 на 2026-10-25

R — required handoff input, O — optional для read-only audit. Список
фиксирован текущим checkpoint; хеши не включают сами документы Day07.

| Path | Role | Git storage | SHA-256 |
| --- | --- | --- | --- |
| `configs/dataset.json` | R | tracked | `fc43038477b9beeb491c505244b7b0a403b8da46f062f61cdc8358126382047d` |
| `configs/experiments_day04.json` | R | tracked | `d9830dd3805252cc523a2fcb6fc22a28638e2c59d31feec11233cb78d908d18a` |
| `configs/train_baseline.json` | R | tracked | `8ffeb9230ae0cbffd6c00939694d028370ea5e825eaabf1cbb374cc025de7315` |
| `data/interim/financial_phrasebank_75agree.csv` | O | local/ignored | `791b78a0be391241bd6ea600fdd0128caee7c0dbe89eed10dfd00de505fb3c2e` |
| `data/processed/day03/splits.csv` | R | local/ignored | `b51d363cac4c00f18d760aa72d3ca6f5272becced6be45d18e763b4b9e557aaf` |
| `data/processed/financial_phrasebank_75agree.csv` | R | local/ignored | `39dda898a281dc3294be081917b0681524e73edb60f00913183c67714da0c448` |
| `data/raw/FinancialPhraseBank-v1.0.zip` | O | tracked | `0e1a06c4900fdae46091d031068601e3773ba067c7cecb5b0da1dcba5ce989a6` |
| `data/raw/source.json` | O | tracked | `f656fab703cbbf88174a45b0bf11b8bf0e716f32c443c1aa385e850c48d12913` |
| `docs/day05_failure_cases.csv` | R | tracked | `7c4c6381ea288b34c2ca90764d3a2e970550d34345d7d5712ce5b4e832f0823f` |
| `docs/day06_examples.csv` | R | tracked | `d99af46c74f82bf0fd999a9ea84e4de66e3ed4b7ddb9dbf93f3f6a7a8a9c363c` |
| `models/day06/manifest.json` | R | local/ignored | `7b155e9fada9f7d4404801e316d52dc0b0242f52c5de1700aa7a1b62392d1080` |
| `models/day06/sentiment_bundle.joblib` | R | local/ignored | `72d5f018ccd5babf9debb597e799ab02a76f8dc551babd216857f9d10cef1300` |
| `reports/day01/class_distribution.csv` | O | local/ignored | `adf43a007c6dc4b4d843dda6648608437a757c22d2f5927247621d8dc5d5671d` |
| `reports/day01/environment.json` | O | local/ignored | `48835e77974f4565337e27195e6209475f27126df18fa1a59411193c56bae1da` |
| `reports/day01/observations.md` | O | local/ignored | `8917bc443504955898c5d890267ace9273b6e2eec67a06e3fdfc186458185d0e` |
| `reports/day01/quality.json` | O | local/ignored | `076ef9e196f26ff5b1c7a657f5d712a1ca77183cdf954753cc4a2a6b7d45bc64` |
| `reports/day01/sentiment_distribution.png` | O | local/ignored | `de516abcd414a83ae643c5a4a67f883b4d3c4c1f649f28952a63f0429bd766c6` |
| `reports/day01/text_length_distribution.png` | O | local/ignored | `1ad19697e753ce120544bac4da7934137c458af2914207f768efe5aa39de8937` |
| `reports/day01/text_lengths.csv` | O | local/ignored | `1c721056056de087e0e0e4316337ce65e8231fdc98ec767256503b7688413fdf` |
| `reports/day02/duplicates.json` | R | local/ignored | `4ad5260b10d35811eb052e2013b52c1012e13567b1d3d3b4a8271248c34d4526` |
| `reports/day02/observations.md` | R | local/ignored | `681840baebc650749a5f89ee2be6ae142c7fa05a62f458cec4c7bc8a56b6a86a` |
| `reports/day02/preprocessing.json` | R | local/ignored | `770a1fa0fe6bacbff52564932a2e98f0fb56c95792ec834c4b0f36b53f07e665` |
| `reports/day02/provenance.json` | R | local/ignored | `8d9e039f9ff7a76efb727a3e21ce2f07232aedb7b38600cde0e04fda7c79f034` |
| `reports/day03/config.json` | R | local/ignored | `90b5abb6d13b209e69b989d052e9da87bef50a9e936cae39eee8b0b7597a3a93` |
| `reports/day03/metrics.json` | R | local/ignored | `727282dc6860f800b0ed05bcb1d17bef7fff007db424c2bb3647da20bc470436` |
| `reports/day03/observations.md` | R | tracked | `ed91a4b280bf830ef930719cfdc047067760199d5d9ae777d627b147626da2bc` |
| `reports/day03/provenance.json` | R | local/ignored | `73f72d587bff42974d4eed6274d6886f03843d264aaf7ef73e801c870d8ad5dd` |
| `reports/day03/split_summary.json` | R | local/ignored | `8fe5b294800246fa9883def1a9fc615831385ed2c689a2cfd861aa42be4e3bc4` |
| `reports/day03/validation_predictions.csv` | R | local/ignored | `b5433913dcfbd94b36effcf759d5d7e8e30af7b04c01f8dfafcde80709400928` |
| `reports/day04/cv_fold_summary.csv` | R | local/ignored | `bdb8ae5d8cb94e85e17bf3be78c6390b28fd863fe821d08a14184c155ee090cc` |
| `reports/day04/cv_folds.csv` | R | local/ignored | `49fdf48077e896e4edfa44ca8fc5a56bbb48584a7e35697d9d4fcac617679e5c` |
| `reports/day04/cv_results.csv` | R | local/ignored | `0a2a31b6d045810eebd2c34e7b44ae11e6f1e7f491e809e5f59cb2e62fe529c2` |
| `reports/day04/cv_summary.csv` | R | local/ignored | `11fe2b505d8c79797abd484539782d258f497b17d7d60133345afce89f760750` |
| `reports/day04/effective_config.json` | R | local/ignored | `f5bfaa618ee8c2cca217a94956b647b6182940a59a147a0d04f0813cbef495d8` |
| `reports/day04/metrics.json` | R | local/ignored | `c095b034a2d45d709c27d507802600d34ec7a6ecfda798b7e29a17c27edb0043` |
| `reports/day04/observations.md` | R | tracked | `fc881d51edfde7f1ce56e756acedac03e20c0b28e33aa342c151ac03d3c15cd0` |
| `reports/day04/provenance.json` | R | local/ignored | `ba101b6e2b5d8b388c4f435fa6b1e954df18d8b2f04987b57fcfcb1bd5f3edbc` |
| `reports/day04/selection.json` | R | local/ignored | `83c3c1805900ea148baf88657f7d4b3862f81a4bdabacbe2615a1ce0f1ac3b98` |
| `reports/day04/validation_predictions.csv` | R | local/ignored | `71d74d84970a956d80306899b0928431177ffcfa7498f3f4fe6e275245a2f987` |
| `reports/day05/analysis_summary.json` | R | local/ignored | `471ec8562f8466e41dfdea84afd0b774c16f6017d38e68c4ecec0c7224297d89` |
| `reports/day05/confusion_pairs.csv` | R | local/ignored | `126f3d561fd199d15fd7c37c9946dde03d79ba9285a53da81208cc9f1882a770` |
| `reports/day05/final_evaluation.json` | R | local/ignored | `1cc8749072afec98f30c8107a92bb7c8d8a526bdc2063e8e7647df68f85e792b` |
| `reports/day05/frozen_protocol.json` | R | local/ignored | `4a9a778085b8f082a20374bf46e1a2053a72aa9a6ee872e3e252dbb8a2570a37` |
| `reports/day05/observations.md` | R | local/ignored | `abba0e3059b076e59bf63af75e0a2e4ab1c3e8c90d24442b65c38c42e41a318d` |
| `reports/day05/provenance.json` | R | local/ignored | `66f094db32335118500f2defa2ac692a53a8dd961988f8ec91d8f24f5c312c3d` |
| `reports/day05/test_confusion_counts.csv` | R | local/ignored | `052e080b5f0fdbd760119d5bcc284774a4799e7254b290141759af2a9a487143` |
| `reports/day05/test_confusion_counts.png` | R | local/ignored | `5d414697607a37b4d3fb5ce82f331a17c5306522425899d75d265612fb707a69` |
| `reports/day05/test_confusion_normalized.csv` | R | local/ignored | `ba5030951b5169d41895665865fa09aa96aa98f60e048b506cd33011f1fc7dbe` |
| `reports/day05/test_confusion_normalized.png` | R | local/ignored | `aa23fbc7420719a601d8d58ec7ae47c9cf82d90ec434f0ff9239c04f486d7504` |
| `reports/day05/test_metrics.json` | R | local/ignored | `ff5b028ae8dbae88cf7f3a8d151edd9a2b1b98b56b803182da7c077b05ce45f9` |
| `reports/day05/test_predictions.csv` | R | local/ignored | `7125eb8d413279e235974d3dcfcbd808b58001f5f38f91409694fcc8ce42acf4` |
| `reports/day05/validation_analysis.csv` | R | local/ignored | `3b1e4ff70a3d3d853a9d02160489537ce8202f7f822a749ccea33b88e59ac2d7` |
| `reports/day05/validation_confusion_counts.csv` | R | local/ignored | `41e1735273854c6cac94c5e92044d511ba89366e32219fca874a98663f04828e` |
| `reports/day05/validation_confusion_counts.png` | R | local/ignored | `6ca425265794d510e12b6f2546a8c848b82ca6419b1b75dd3b8207653ac12339` |
| `reports/day05/validation_confusion_normalized.csv` | R | local/ignored | `76ea2473b12b8ccf7d4aa81ebd7444715814dabf9a1a0ef5d1772a463eb20470` |
| `reports/day05/validation_confusion_normalized.png` | R | local/ignored | `e6e5e8fdbe768c26278b9ed6ae57ec1b165e42fa20a17725d9dea25980d067a7` |
| `reports/day05/validation_errors.csv` | R | local/ignored | `d8c87a43bdad609a472bdfe8b918b718a10f76f195cc4e7ac0938fcd460d3efa` |
| `reports/day05/validation_examples.csv` | R | local/ignored | `fec137352a02ebdfcd6e93f912989aaeeef76503cdf0584cb131be03b0efa3e4` |
| `reports/day06/consistency.json` | R | local/ignored | `c4822ea3da21b3b78f0f43f7c59e71094e09517a84ae16a4f6f2dc6664233b03` |
| `reports/day06/demo_predictions.csv` | R | local/ignored | `2d689d528e0c4520e5b9e0616cf7206808f959409e83b66ea89d106d5856b6db` |
| `requirements-day01.txt` | R | tracked | `ff26a94cce7d1e207dbaac866513a631a894f472447ca0465c5719de6d7ed6f9` |
| `requirements-day03.txt` | R | tracked | `b09ad0d009144b17e1b035195af998bbc290f9b5487b3328de6a558d24b0feef` |
| `requirements-dev.txt` | R | tracked | `388367b43f04921a11d2b6c82b377c4a8d22bc5952fc9d5cf965ef186c7a5c81` |
