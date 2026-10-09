# PLAN — staged implementation

## Порядок работы

Day01 завершён 2026-10-09. Следующий пункт — D2-01; он ещё не реализован. Если пользователь называет ID, выполняем только его; если день —
его зависимые пункты последовательно. Каждый завершённый ID имеет раздел notebook,
проверку и свидетельство ниже. `[ ]` — ещё не завершено, `[x]` — подтверждено.
Будущие дни — roadmap; детализация до реализации, без опережающих features.

## Day01 — данные и EDA

### D1-00 — аудит, контракты и окружение

- [x] Прочитать все исходники, сформировать AGENTS/SPEC/PLAN.
- Цель: согласовать три класса, границы day01 и учебный рабочий цикл.
- Выход: документы, pinning используемых day01 packages, notebook setup.
- Проверка: interpreter/packages доступны; notebook определяет root и импортирует src.
- Самопроверка: чем SPEC отличается от PLAN и AGENTS? Почему notebook вызывает modules?
- Свидетельство: Прочитаны все файлы задания и исходники; AGENTS/SPEC/PLAN созданы. D1-00 notebook выполнен на Python 3.14.6; версии в requirements-day01.txt и reports/day01/environment.json.

### D1-01 — источник и воспроизводимая загрузка

- [x] Подготовить Financial PhraseBank `sentences_75agree` и canonical CSV.
- Зависит от D1-00. Вход: pinned ZIP; выход: `(3453, 2)` с `text`, `sentiment`.
- Сохранить revision, checksum, license, class order. Download отдельно от EDA.
- Исправить CSV loader под минимальный контракт; сохранить raw data и порядок строк.
- Проверка: hash, archive parsing, CSV roundtrip; missing/invalid schema tests.
- Самопроверка: agreement subset — это train/test split? Почему encoding надо фиксировать?
- Свидетельство: Архив 681890 bytes загружен с pinned revision 8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964; SHA-256 подтверждён. CSV (3453, 2); roundtrip/hash/schema проверки проходят.

### D1-02 — структура и качество данных

- [x] Shape/head/info/dtypes, missing, blanks, duplicates и conflicting labels.
- Зависит от D1-01. Выход: quality summary и `reports/day01/quality.json`.
- Никакой очистки или скрытого удаления строк. Counts имеют явный denominator.
- Проверка: fixtures с NaN, пробелами, duplicates/conflicts; input не изменён.
- Самопроверка: почему отсутствие missing ещё не означает хорошие данные?
- Свидетельство: Реальный quality.json: missing=0, blanks=0, duplicate rows/texts=5, conflicting texts=0. Notebook проверяет неизменность df; edge cases покрыты offline tests.

### D1-03 — баланс классов

- [x] Counts, доли и график классов в фиксированном порядке.
- Зависит от D1-02. Выход: class CSV и sentiment_distribution.png.
- Missing labels учитываются отдельно; доля класса = count / общее число строк.
- Проверка: сумма counts + missing = N; отсутствующий класс имеет count=0.
- Самопроверка: почему accuracy может скрыть плохую работу на редком классе?
- Свидетельство: Реальные counts: negative=420 (12.16%), neutral=2146 (62.15%), positive=887 (25.69%). Сумма=3453; class CSV и PNG сохранены; fixed order/denominator tests проходят.

### D1-04 — длины и примеры

- [x] Символы/слова, describe, histogram и до трёх валидных примеров каждого класса.
- Зависит от D1-02. Выход: text_lengths.csv и text_length_distribution.png.
- Никаких regex cleaning или NLP tokenization; null не становится строкой `nan`.
- Проверка: известные длины fixture; пропуски сохранены; графики сохраняются без GUI.
- Самопроверка: чем слова по пробелам отличаются от tokens? Зачем исследовать хвост длин?
- Свидетельство: Median: 116 символов / 21 слов; max=81 слов. Сохранены text_lengths.csv, histogram PNG, по три примера каждого класса. Проверены missing lengths и сохранение PNG.

### D1-05 — выводы и clean-kernel checkpoint

- [x] Записать фактические наблюдения, ограничения и вопросы для day02.
- Зависит от D1-01…D1-04. Выход: observations.md и полностью выполненный notebook.
- Проверка: unit tests, compile, diff check, real-data run, notebook clean kernel.
- Сохранить версии runtime; повторный offline run не требует download или training.
- Самопроверка: что day01 позволяет заключить, а что требует обученной модели и test?
- Свидетельство: 12/12 unittest passed; compileall passed; git diff --check passed. Реальный notebook: 6/6 code cells executed, 0 errors, outputs сохранены; чистый offline kernel venv. Повторный запуск командой nbconvert из docs/day01.md успешен. Для kernel потребовался выход из sandbox из-за локальных sockets; Pyright gate добавлен отдельным последующим исправлением ниже; Ruff не запускался.

## Roadmap day02–day07

| День | Подзадачи (все pending) | Результат / notebook |
| --- | --- | --- |
| 02 | D2-01 минимальная cleanup policy; D2-02 reusable preprocessing без import side effects; D2-03 labels и duplicate/conflict policy; D2-04 edge cases и выводы | processed dataset/contract, day02_preprocessing.ipynb |
| 03 | D3-01 stable group-aware splits; D3-02 Dummy + TF-IDF/LogReg Pipeline; D3-03 validation metrics; D3-04 config/results/provenance | baseline reference, day03_baseline.ipynb |
| 04 | D4-01 hypotheses/protocol; D4-02 n-grams; D4-03 LinearSVC; D4-04 hyperparameter CV; D4-05 comparison/selection | ≥3 controlled experiments, day04_experiments.ipynb |
| 05 | D5-01 aligned predictions; D5-02 confusion matrix; D5-03 failure categories; D5-04 frozen final test/report | error analysis, day05_error_analysis.ipynb |
| 06 | D6-01 artifact bundle; D6-02 load/predict API; D6-03 roundtrip consistency; D6-04 new-text demos | inference without fit, day06_inference.ipynb |
| 07 | D7-01 end-to-end review; D7-02 runnable README; D7-03 summary/limitations; D7-04 clean-environment acceptance | reproducible mini-product, day07_review.ipynb |

## Шаблон запроса для следующего чата

> Выполни D2-01 из PLAN.md. Объясни цель, ML/NLP понятия и tradeoffs. Реализуй только
> этот шаг, обнови notebook и свидетельство в PLAN. Запусти нужные проверки,
> назови ограничения и предложи commit message; не делай commit.

При работе целым днём замените ID на day02. Крупные решения фиксируем в SPEC;
вопросы пользователя можно обсуждать, не меняя scope реализации.

## Дополнение day01 — Pylance/Pyright quality gate

- [x] Добавить pinned Pyright и pandas-stubs, standard config для venv/src/tests/scripts.
- [x] Исправить ROOT: Path через find_project_root; scalar row selection через iloc;
  word count без неподдерживаемого в stubs list-string accessor.
- [x] Добавить scripts/check_types.py с проверкой code cells всех notebooks и настройки VS Code.
- Проверки: Pyright sources и notebook export — 0 errors, 0 warnings, 0 informations; 12/12 unit tests; compileall и diff check прошли. Исправленный notebook повторно выполнен в чистом kernel, outputs обновлены.
