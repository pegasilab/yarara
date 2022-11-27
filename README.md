# yarara
YARARA working copy

## To install

```bash
cd yarara
git submodule update --init # will also pull the HD110315 data for tests
cd Python
python -m venv .venv # to create the virtual environment that Poetry will use
.venv/bin/python -m pip install --upgrade pip #upgrade the pip version to latest one
poetry install --all-extras # remove --all-extras if you do not want to build the docs
```

## To try it

```bash
cd Python # (if not done already)
poetry run ./run_HD110315.sh
```

## To build the docs

Launch, in the `Python/` code folder (i.e. the one containing `pyproject.toml`):

```bash
cd Python # (if not done already)
poetry run make -C docs clean html
```

The files are then in `Python/docs/build/html`.
