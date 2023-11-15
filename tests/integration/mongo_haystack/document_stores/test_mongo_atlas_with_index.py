import os
import re
import requests
import pytest
import roman
import numpy
import pymongo
from mongo_haystack.document_stores.mongo_filters import (
    _target_filter_to_metadata,
    _and_or_to_list,
    mongo_filter_converter,
)
from mongo_haystack.document_stores.mongo_atlas import MongoAtlasDocumentStore
from haystack.schema import Document
from haystack.pipelines import Pipeline
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents
from haystack.nodes import (
    PreProcessor,
    EmbeddingRetriever,
    PromptNode,
    PromptTemplate,
    AnswerParser,
)

pytestmark = pytest.mark.search_index

mongo_atlas_database = "database01"
mongo_atlas_collection = "test_80_days"

mongo_atlas_username = os.getenv("MONGO_ATLAS_USERNAME")
mongo_atlas_password = os.getenv("MONGO_ATLAS_PASSWORD")
mongo_atlas_host = os.getenv("MONGO_ATLAS_HOST")
mongo_atlas_connection_params = {"retryWrites": "true", "w": "majority"}
mongo_atlas_params_string = "&".join([f"{key}={value}" for key, value in mongo_atlas_connection_params.items()])
mongo_atlas_connection_string = (
    f"mongodb+srv://{mongo_atlas_username}:{mongo_atlas_password}@{mongo_atlas_host}/?{mongo_atlas_params_string}"
)

document_store = MongoAtlasDocumentStore(
    mongo_connection_string=mongo_atlas_connection_string,
    database_name=mongo_atlas_database,
    collection_name=mongo_atlas_collection,
    embedding_dim=768,
)


# Get the book "Around the World in 80 Days" from Project Gutenberg
def get_book_online():
    response = requests.get("https://www.gutenberg.org/ebooks/103.txt.utf-8")
    if response.status_code != 200:
        raise requests.HTTPError(f"HTTP error {response.status_code}")
    else:
        return response.text


def get_book_local():
    with open("test-data/80_days.txt", "r", encoding="utf-8") as file:
        text = file.read()
    return text


get_book = get_book_local


# Divide the book into chapters
def divide_book_into_chapters(book) -> dict:
    lines = book.split("\n")
    current_chapter = None
    chapters = {}
    for line in lines:
        chapter_match = re.match(r"CHAPTER\s+([IVXLCDM]+)\.*", line)
        if chapter_match:
            chapter_roman = chapter_match.group(1)
            chapter_decimal = roman.fromRoman(chapter_roman)
            current_chapter = f"CHAPTER {chapter_decimal}".title()
            chapters[current_chapter] = ""
        if current_chapter:
            chapters[current_chapter] += line + "\n"
    return chapters


book = get_book()
chapters = divide_book_into_chapters(book)
documents = [
    Document(
        content=chapters[f"Chapter {n}"],
        meta={"book": "Around the World in 80 Days", "Chapter": n},
    )
    for n in range(1, len(chapters) + 1)
]


def test_write_documents():
    document_store.delete_documents()

    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        remove_substrings=None,
        split_by="word",
        split_length=200,
        split_respect_sentence_boundary=True,
        split_overlap=0,
        max_chars_check=10_000,
    )

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # Recommended here: https://www.sbert.net/docs/pretrained_models.html
        model_format="sentence_transformers",
        top_k=10,
    )

    processed_documents = processor.process(documents)
    document_store.write_documents(processed_documents)

    assert document_store.get_document_count() == 373
    assert document_store.get_all_documents(return_embedding=True)[0].embedding is None


def test_update_embeddings():
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # Recommended here: https://www.sbert.net/docs/pretrained_models.html
        model_format="sentence_transformers",
        top_k=10,
    )

    document_store.update_embeddings(retriever, batch_size=30)
    assert isinstance(document_store.get_all_documents(return_embedding=True)[0].embedding, numpy.ndarray)


def test_query_by_embedding_default_topk():
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # Recommended here: https://www.sbert.net/docs/pretrained_models.html
        model_format="sentence_transformers",
    )
    embedding = retriever.embed_queries(["How much money was stolen from the bank?"])[0]
    results = document_store.query_by_embedding(query_emb=embedding)
    assert results[0].embedding == None
    assert len(results) == 10


def test_query_by_embedding_default_topk_4():
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # Recommended here: https://www.sbert.net/docs/pretrained_models.html
        model_format="sentence_transformers",
    )
    embedding = retriever.embed_queries(["How much money was stolen from the bank?"])[0]
    results = document_store.query_by_embedding(query_emb=embedding, top_k=4)
    assert results[0].embedding == None
    assert len(results) == 4


def test_query_by_embedding_filtered():
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # Recommended here: https://www.sbert.net/docs/pretrained_models.html
        model_format="sentence_transformers",
        top_k=10,
    )
    embedding = retriever.embed_queries(["Who was Phileas Fogg?"])[0]
    results = document_store.query_by_embedding(query_emb=embedding, filters={"Chapter": 1})
    assert len(results) == 3


def test_query_by_embedding_include_embedding():
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # Recommended here: https://www.sbert.net/docs/pretrained_models.html
        model_format="sentence_transformers",
        top_k=10,
    )
    embedding = retriever.embed_queries(["How much money was stolen from the bank?"])[0]
    results = document_store.query_by_embedding(query_emb=embedding, return_embedding=True)
    assert isinstance(results[0].embedding, numpy.ndarray)
    assert len(results) == 10
