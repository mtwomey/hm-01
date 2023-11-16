
# How to use the demo

## Set your environment variables

``` shell
export MONGO_ATLAS_USERNAME="[YOU MONGO ATLAS USERNAME]"
export MONGO_ATLAS_PASSWORD="[YOUR MONGO ATLAS PASSWORD]"
export MONGO_ATLAS_HOST="[MONGO ATLAS HOST]"
export MONGO_ATLAS_DATABASE="[YOUR MONGO ATLAS DATABASE]"
export MONGO_ATLAS_COLLECTION="80_days"

export OPENAI_KEY="[YOUR OPENAI API KEY]"
```

## Prepare your python environment

Follow the `Local development` directions in the main README.md. This will install all necessary python packages.

## Run the setup

`setup.py` will import and segment the text of the book Around the World in Eighty Days, by Jules Verne. The text of this book is public domain and was obtained from Project Gutenberg. After pre-processing and splitting, the documents will be added to a collection named `80_days` in your database. Embeddings will then be added to the documents in the collection. This will result in a total of 373 documents in the collection.

Once complete, you must create the vector search index in the Mongo Atlas web console using the JSON editor.

For details see this document:

[https://www.mongodb.com/docs/atlas/atlas-search/field-types/knn-vector/#std-label-fts-data-types-knn-vector](How to Index Vector Embeddings for Vector Search) 

Create a search index named `80_days` using the JSON editor in the Mongo Atlas web console. Select the `80_days` collection and use the following JSON definition.

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

## Run the demo

