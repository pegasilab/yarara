# YARARA
![YARARA logo](_static/logo.jpg)

This is a distribution of the YARARA reduction pipeline written by [Michaël Crétignier](mailto:michael.cretignier@gmail.com), 
with some refactoring by [Denis Rosset](mailto:physics@denisrosset.com).

## To install

```bash
git clone https://github.com/pegasilab/yarara.git
cd yarara
git submodule update --init # will also pull the HD110315 data for tests
python -m venv .venv # to create the virtual environment that Poetry will use
.venv/bin/python -m pip install --upgrade pip #upgrade the pip version to latest one
poetry install --all-extras # remove --all-extras if you do not want to build the docs
```

## To try it without fuss

```bash
poetry run test/bats/bin/bats test/test_HD110315.bats
# will run the full pipeline including RASSINE
```

## To run the full pipeline

**Warning**: Do not run RASSINE or YARARA in paths containing spaces. It will fail.

- Datafiles are described [here](data.md)
- Products are described [here](products.md)

Perform first a full installation as described above using `poetry install` in an isolated Python environment.

The installation will bring the RASSINE command line tools and a `run_yarara` command line tool as well.

First, you need your input files in a folder called `STARNAME/data/s1d/INSTRUMENT`. Follow the [RASSINE tutorial](https://pegasilab.github.io/rassine/quickstart.html#how-to-customize-the-above-for-a-different-star).

Note: you do not need to place those files in a subdirectory of `spectra/`, as it is the case for the `HD110315` data used for tests. You can use the instructions below with absolute or relative paths, if you wish. Relative paths are interpreted with respect to the current working directory.

The `harpn.ini`, `harps03.ini`, `parallel` and `run_rassine.sh` files have been copied to the Yarara repository as well. You can simply reuse them.

This command will run all the RASSINE stages.

```bash
poetry run ./run_rassine.sh -l WARNING -c harpn.ini spectra/STARNAME/data/s1d/INSTRUMENT
```

Then, copy all the `RASSINE_*` files from `STACKED/` in a new folder called `WORKSPACE/`.

```bash
mkdir spectra/STARNAME/data/s1d/INSTRUMENT/WORKSPACE
cp spectra/STARNAME/data/s1d/INSTRUMENT/STACKED/RASSINE_* spectra/STARNAME/data/s1d/INSTRUMENT/WORKSPACE
```

Then run the full Yarara pipeline:

```bash
poetry run run_yarara spectra/STARNAME/data/s1d/INSTRUMENT
```

## To build the docs

Launch:

```bash
poetry run make -C docs clean html
```

The files are then in `docs/build/html`.

```{toctree}
:hidden:
:maxdepth: 3
:caption: General information

Home page <self>
data
products
```

```{toctree}
:hidden:
:maxdepth: 3
:caption: API

api
```
