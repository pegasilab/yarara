# yarara
YARARA working copy

## To install

```bash
cd yarara
git submodule update --init # will also pull the HD110315 data for tests
cd Python
python -m venv .venv # to create the virtual environment that Poetry will use
poetry install --all-extras # remove --all-extras if you do not want to build the docs
```

## To try it

```bash
cd Python # (if not done already)
poetry run test/bats/bin/bats test/test_HD110315.bats # will run the full pipeline including RASSINE
```

## To build the docs

Launch, in the `Python/` code folder (i.e. the one containing `pyproject.toml`):

```bash
cd Python # (if not done already)
poetry run make -C docs clean html
```

The files are then in `Python/docs/build/html`.
