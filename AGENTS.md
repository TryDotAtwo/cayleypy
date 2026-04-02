# CayleyPy — заметки для агентов и разработчиков

## Репозиторий

- `origin`: `https://github.com/cayleypy/cayleypy.git`
- Локальный клон: каталог `cayleypy` (не родительский `CayleyPy`), там же `.git`.

## Cursor / VS Code

- Открывать в редакторе папку **`…\CayleyPy\cayleypy`**, чтобы встроенный Git, ветки и PR видели репозиторий.
- Если нужен родительский `CayleyPy` в дереве файлов — multi-root workspace: добавить вторую папку `cayleypy`.

## Pull request (типовой поток)

1. На GitHub: **Fork** `cayleypy/cayleypy` в свой аккаунт (если нет прямого push в организацию).
2. Локально: `git remote add fork https://github.com/<ваш_логин>/cayleypy.git` (имя `fork` — по желанию).
3. `git fetch origin` → ветка от актуального `origin/main`: `git checkout -b feature/кратко-описание origin/main`.
4. Коммиты → `git push -u fork feature/кратко-описание`.
5. На GitHub: **Compare & pull request** из вашей ветки в `cayleypy/cayleypy` → `main` (или через `gh pr create`, если установлен GitHub CLI).

При наличии прав записи в `cayleypy/cayleypy` можно пушить в `origin` и открывать PR без форка.

### Pull request из Cursor

- Панель **Source Control** (иконка ветки): коммит, выбор ветки, **Sync/Push** после настройки `remote`.
- После `git push` открыть репозиторий на GitHub → появится баннер **Compare & pull request**, либо вкладка **Pull requests** → **New pull request**.
- Расширение **GitHub Pull Requests** (официальное от GitHub для VS Code/Cursor) позволяет просматривать и создавать PR из боковой панели, если выполнен вход в GitHub в редакторе.
- Установленный **GitHub CLI** (`gh`): в терминале проекта `gh pr create` после push.

## Git (Windows)

- Установка: **Git for Windows** через `winget install --id Git.Git -e` (на этой машине: v2.53.x, каталог `C:\Program Files\Git\`).
- В новых терминалах и после перезапуска Cursor команда `git` обычно доступна из PATH. Если терминал открыт до установки и пишет «git не найден» — закрыть и открыть терминал заново или перезапустить Cursor; обходной путь: полный путь `& "C:\Program Files\Git\bin\git.exe"`.
- Глобальная идентичность коммитов: `git config --global user.name`, `git config --global user.email` (уже заданы на этой системе).
- Рабочий каталог репозитория: `…\CayleyPy\cayleypy` (не вложенная только `cayleypy\cayleypy` без родительского `.git`).

## Kaggle: установка из своего форка

Создать форк на GitHub: репозиторий `cayleypy/cayleypy` → **Fork**. Локально ветка с правками (пример): `feature/bfs-packed-scatter`.

```text
git remote add fork https://github.com/<ваш_логин>/cayleypy.git
git push -u fork feature/bfs-packed-scatter
```

В ноутбуке Kaggle (после включения интернета в настройках сессии):

```text
pip install "git+https://github.com/<ваш_логин>/cayleypy.git@feature/bfs-packed-scatter"
```

Либа ставится как пакет `cayleypy` из корня репозитория (см. `pyproject.toml` / `setup` в проекте).

Форк для тестов (аккаунт `TryDotAtwo`): **https://github.com/TryDotAtwo/cayleypy**. Создание форка и remote: `gh repo fork cayleypy/cayleypy --remote-name fork` (при необходимости вручную: `git remote add fork https://github.com/TryDotAtwo/cayleypy.git`). Ветка с packed scatter: `feature/bfs-packed-scatter`.

```text
pip install "git+https://github.com/TryDotAtwo/cayleypy.git@feature/bfs-packed-scatter"
```

Дымовой тест после установки: `python -m cayleypy.kaggle_fork_smoke` (или консольная команда `cayleypy-kaggle-smoke`). См. модуль `cayleypy/kaggle_fork_smoke.py`.

Сравнение **upstream `main`** и **форка** двумя ячейками Kaggle (git clone + общий `bfs_suite`, JSON после ячейки 1): `cayleypy/kaggle_two_cells_compare.md`. Готовый Jupyter: **`cayleypy/kaggle_upstream_fork_compare.ipynb`**; генератор `tools/_gen_kaggle_nb.py`.

## История изменений этого файла

| Дата       | Изменение |
|------------|-----------|
| 2026-04-02 | Первое заполнение: origin, путь клонирования, открытие папки в IDE, сценарий PR. |
| 2026-04-02 | Секция «Pull request из Cursor»: Source Control, веб-интерфейс GitHub, расширение PR, `gh pr create`. |
| 2026-04-02 | `cayleypy/algo/bfs_distributed.py`: восстановлен из ветки `main` репозитория `cayleypy/cayleypy` (HTTPS raw). Ранее задокументированные локальные оптимизации этого файла сняты. С локальным git: `git fetch origin` → `git checkout origin/main -- cayleypy/algo/bfs_distributed.py`. В среде без `git` в PATH — тот же raw-URL. |
| 2026-04-02 | Секция «Git (Windows)»: установка через winget, PATH, перезапуск терминала, проверка `user.name`/`user.email`, корень репозитория `…\CayleyPy\cayleypy`. |
| 2026-04-02 | `bfs_distributed._bfs_layer_distributed`: вместо сетки буферов `send[num_gpus][num_gpus]` — упаковка строк по `hash % num_gpus` на каждом GPU (`_pack_states_for_scatter`), приём тем же порядком `cat` по `source`; после упаковки `del phase1_results`. Цель — снизить пик VRAM (O(G²) отдельных тензоров → O(G) packed на источник). |
| 2026-04-02 | Секция «Kaggle: установка из своего форка»: `git remote add fork`, `git push -u fork feature/bfs-packed-scatter`, `pip install git+https://...@feature/bfs-packed-scatter`. |
| 2026-04-02 | Форк `TryDotAtwo/cayleypy`, ветка `feature/bfs-packed-scatter` запушена; в секции Kaggle — прямой `pip install` и ссылка на `gh repo fork`. |
| 2026-04-02 | `cayleypy/kaggle_fork_smoke.py`: дымовой тест форка для Kaggle; entry point `cayleypy-kaggle-smoke` в `pyproject.toml`. |
| 2026-04-02 | `cayleypy/kaggle_two_cells_compare.md`: две ячейки — clone `main` / clone форка, сравнение `layer_sizes` с JSON из первой ячейки. |
| 2026-04-02 | Тот же файл: замеры `time_sec`, `cuda_peak_mib` (сумма по GPU), `cpu_tracemalloc_peak_mib` для `lrx5_cpu`; assert только по `layer_sizes`; таблица сравнения времени/памяти fork vs upstream. |
| 2026-04-02 | `cayleypy/kaggle_upstream_fork_compare.ipynb` + `tools/_gen_kaggle_nb.py` — тот же сценарий в формате Jupyter для загрузки на Kaggle. Исключение в `.gitignore` для этого `.ipynb` (остальные `*.ipynb` по-прежнему игнорируются). |
