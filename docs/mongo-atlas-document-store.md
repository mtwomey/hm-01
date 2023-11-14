---
layout: integration
name: Mongo Atlas Document Store
description: Use a Mongo Atlas database with Haystack
authors:
    - name: mtwomey
      socials:
        github: mtwomey
        twitter: mtwomey
        linkedin: mtwomey
pypi: https://pypi.org/project/farm-haystack
repo: https://github.com/deepset-ai/haystack
type: Document Store
report_issue: https://github.com/deepset-ai/haystack/issues
logo: /logos/mongodb.png
---

[MongoDB Atlas Database](https://www.mongodb.com/atlas/database) with vector search can used in Haystack pipelines with the [MongoAtlasDocumentStore](https://docs.haystack.deepset.ai/docs/document_store#initialization)

For a detailed overview of all the available methods and settings for the `MongoAtlasDocumentStore`, visit the Haystack [API Reference](https://docs.haystack.deepset.ai/reference/document-store-api#mongoatlasdocumentstore)

## Installation

```bash
pip install farm-haystack[mongoatlas]
```

## Usage

To use Mongo Atlas as your data storage for your Haystack LLM pipelines, you must have an account with Mongo Atlas. You can initialize a `MongoAtlasDocumentStore` for Haystack:

```
from haystack.document_stores import MongoAtlasDocumentStore

document_store = MongoAtlasDocumentStore(
    mongo_connection_string='YOUR_CONNECTION_STRING'
    database_name='YOUR_DATABASE_NAME',
    collection_name='YOUR_COLLECTION_NAME,
    embedding_dim=768,
)

```

Example connection string: `mongodb+srv://mtwomey:Q4AnLVsSFmogW7rSrM9u@cluster0.dqvvbud.mongodb.net/?retryWrites=true&w=majority`

You must create a vector search index in the Mongo Atlas web console using the JSON editor:

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

During creation, select your collection. Note: The search index name __must__ match the collection name.

### Writing Documents to MongoAtlasDocumentStore

To write documents to your `MongoAtlasDocumentStore`, create an indexing pipeline, or use the `write_documents()` function.
For this step, you may make use of the available [FileConverters](https://docs.haystack.deepset.ai/docs/file_converters) and [PreProcessors](https://docs.haystack.deepset.ai/docs/preprocessor), as well as other [Integrations](/integrations) that might help you fetch data from other resources. Below is an example indexing pipeline that indexes your Markdown files into a Mongo Atlas database.

#### Indexing Pipeline

```python
from haystack import Pipeline
from haystack.document_stores import MongoAtlasDocumentStore
from haystack.nodes import MarkdownConverter, PreProcessor

document_store = MongoAtlasDocumentStore(
    mongo_connection_string='YOUR_CONNECTION_STRING'
    database_name='YOUR_DATABASE_NAME',
    collection_name='YOUR_COLLECTION_NAME,
    embedding_dim=768,
)

converter = MarkdownConverter()
preprocessor = PreProcessor()

indexing_pipeline = Pipeline()
indexing_pipeline.add_node(component=converter, name="PDFConverter", inputs=["File"])
indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["PDFConverter"])
indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PreProcessor"])

indexing_pipeline.run(file_paths=["filename.pdf"])
```

### Using Mondo Atlas in a Query Pipeline

Once you have documents in your `MongoAtlasDocumentStore`, it's ready to be used in any Haystack pipeline. For example, below is a pipeline that makes use of a custom prompt that is designed to answer questions for the retrieved documents.

```python
from haystack import Pipeline
from haystack.document_stores import MongoAtlasDocumentStore
from haystack.nodes import AnswerParser, EmbeddingRetriever, PromptNode, PromptTemplate

document_store = MongoAtlasDocumentStore(
    mongo_connection_string='YOUR_CONNECTION_STRING'
    database_name='YOUR_DATABASE_NAME',
    collection_name='YOUR_COLLECTION_NAME,
    embedding_dim=768,
)
              
retriever = EmbeddingRetriever(document_store = document_store,
                               embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1")
prompt_template = PromptTemplate(prompt = """"Answer the following query based on the provided context. If the context does
                                              not include an answer, reply with 'I don't know'.\n
                                              Query: {query}\n
                                              Documents: {join(documents)}
                                              Answer: 
                                          """,
                                          output_parser=AnswerParser())
prompt_node = PromptNode(model_name_or_path = "gpt-4",
                         api_key = "YOUR_OPENAI_KEY",
                         default_prompt_template = prompt_template)

query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])

query_pipeline.run(query = "What is Mongo Atlas", params={"Retriever" : {"top_k": 5}})
```
