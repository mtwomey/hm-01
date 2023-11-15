# General Information

MongoAtlasDocumentStore is a document store for use with Haystack. It functions in a similar manner as the existing Pinecone, Weaviate, and Qdrant document stores.

There is a demo app showing how it can be used in the demo folder.

## Creating a Mongo Atlas Search Index

To use this document store you must create a vector search index for your collection in the Mongo Atlas web console using the JSON editor:

[https://www.mongodb.com/docs/atlas/atlas-search/field-types/knn-vector/#std-label-fts-data-types-knn-vector](How to Index Vector Embeddings for Vector Search) 

Example vector search index:

```
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 768,
        "similarity": "cosine",
        "type": "knnVector"
      }
    }
  }
}
```

During creation select your collection and ensure that the name of the search matches the collection name.

# Local development

The easiest way to get a proper environment setup locally is to use conda (miniconda or anaconda).

## Conda environment setup:

``` shell
conda create -n mongo-ds-01 python=3.11 -y
conda activate mongo-ds-01
```

You can skip this step if you prefer not to use conda or use a different venv management tool. Note the python 3.11 version dependency.

## Install Poetry

``` shell
pip install poetry
```

## Setup dependencies and install module locally:

CD to the project directory.

``` python
poetry install
```

Alternatively you can run `pip install -r requirements.txt`, but using poetry is recommended.

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

**Tests that require the search index:** 

These tests require the search index to be present. You can run these initially to create the collection, they will fail but the collection will be created. Then create the search index in the Atlas console and run the tests a second time. You may need to delete and recreate the index if the collection was deleted.

``` shell
poetry run pytest -v --cov=./src --cov-report=term-missing --override-ini "addopts=" -m search_index
```

For these tests, you must create a search index in the Mongo Atlas console JSON editor. The index must be named `test_80_days` and be assocaited with the newly created `test_80_days` collection. See `Creating a Mongo Atlas Vector Search Index` above.
