# yarara
YARARA working copy

## To build the docs

First, be sure to have run:

```
poetry install -E docs
```

so that the extra packages for building the documentation are there.

Then, launch, in the Python code folder (i.e. the one containing `pyproject.toml`):

```
poetry run make -C docs clean html
```

The files are then in `docs/build/html`.

The site is built automatically on GitHub when commits are pushed to the `main` branch.
