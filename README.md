# yarara
YARARA working copy

## To install

```
cd Python
poetry install --all-extras # remove --all-extras if you do not want to build the docs
```

## To build the docs

Launch, in the `Python/` code folder (i.e. the one containing `pyproject.toml`):

```
poetry run make -C docs clean html
```

The files are then in `Python/docs/build/html`.
