# yarara
YARARA working copy

## To install

```bash
git clone git@github.com:pegasilab/yarara.git
cd yarara
git submodule update --init # will also pull the HD110315 data for tests
cd Python
python -m venv .venv # to create the virtual environment that Poetry will use
.venv/bin/python -m pip install --upgrade pip #upgrade the pip version to latest one
poetry install --all-extras # remove --all-extras if you do not want to build the docs
```

All the remaining commands below are run from the /Python subdirectory.

## To try it

```bash
poetry run test/bats/bin/bats test/test_HD110315.bats # will run the full pipeline including RASSINE
```

## To build the docs

Launch, in the `Python/` code folder (i.e. the one containing `pyproject.toml`):

```bash
poetry run make -C docs clean html
```

The files are then in `docs/build/html`.
