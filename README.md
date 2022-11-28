# yarara
YARARA working copy

## To install

```bash
git clone https://github.com/pegasilab/yarara.git
cd yarara
git submodule update --init # will also pull the HD110315 data for tests
python -m venv .venv # to create the virtual environment that Poetry will use
.venv/bin/python -m pip install --upgrade pip #upgrade the pip version to latest one
poetry install --all-extras # remove --all-extras if you do not want to build the docs
```

## To try it

```bash
poetry run test/bats/bin/bats test/test_HD110315.bats # will run the full pipeline including RASSINE
```

## To build the docs

Launch:

```bash
poetry run make -C docs clean html
```

The files are then in `docs/build/html`.
