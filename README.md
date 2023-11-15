# Local development

The easiest way to get a proper environment setup locally is to use conda (miniconda or anaconda).

## Conda environment setup:

``` shell
conda create -n mongo-ds-01 python=3.11 -y
conda activate mongo-ds-01
```

Note the python 3.11 version dependency.

## Install Poetry

``` shell
pip install poetry
```

## Setup dependencies and install module locally:

``` python
poetry insall
```

# Tests

**Running all tests:**

``` shell
poetry run pytest -v
```
Note: See `Running integration tests` below regarding env variables.

**Running unit tests:**

``` shell
poetry run pytest -v -m unit
```

**Running integration tests:**

Note: In order to run integration tests, you will need a Mongo Atlas instance. Set the following environment variables before running:

``` shell
export MONGO_ATLAS_USERNAME="[USERNAME]"
export MONGO_ATLAS_PASSWORD="[PASSWORD]"
export MONGO_ATLAS_HOST="[HOST]"

```

Running these tests will create (and remove) a small collection called `test_80_days` in your database.

Run the tests:

``` shell
poetry run pytest -v -m integration
```

**Tests the require the search index:** 

These tests require the search index to be present. You can run these initially to create the collection, then create the search index in the Atlas console, then on a second run it will use the index. You may need to delete and recreate the index if the collection was deleted.

``` shell
poetry run pytest -v --cov=./src --cov-report=term-missing --override-ini "addopts=" -m search_index
```
