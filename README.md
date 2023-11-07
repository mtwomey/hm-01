# Local development

Conda environment setup:

``` python
conda activate
conda env remove -n mongo-ds-01
conda create -n mongo-ds-01 python=3.11 -y
conda activate mongo-ds-01
conda install pip -y
```

Setup dependencies and install module locally:

``` python
poetry insall
```

# Tests

**Running all tests:**

``` python
poetry run pytest -v
```
Note: See `Running integration tests` below regarding env variables.

**Running unit tests:**

``` python
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

``` python
poetry run pytest -v -m integration
```
