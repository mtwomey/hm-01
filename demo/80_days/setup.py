import os
import sys
import re
import roman
from haystack.nodes import EmbeddingRetriever, PreProcessor
from haystack.schema import Document
from mongo_haystack import MongoAtlasDocumentStore

# Process the book text into Haystack Documents


def get_book():
    with open("80_days.txt", "r", encoding="utf-8") as file:
        text = file.read()
    return text


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


chapters = divide_book_into_chapters(get_book())

documents = [
    Document(
        content=chapters[f"Chapter {n}"],
        meta={"book": "Around the World in 80 Days", "Chapter": n},
    )
    for n in range(1, len(chapters) + 1)
]

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

processed_documents = processor.process(documents)

# Load the Documents into a Mongo Atlas Collection

mongo_atlas_username = os.getenv("MONGO_ATLAS_USERNAME")
mongo_atlas_password = os.getenv("MONGO_ATLAS_PASSWORD")
mongo_atlas_host = os.getenv("MONGO_ATLAS_HOST")
mongo_atlas_database = os.getenv("MONGO_ATLAS_DATABASE")
mongo_atlas_collection = "80_days"
mongo_atlas_connection_params = {"retryWrites": "true", "w": "majority"}
mongo_atlas_params_string = "&".join([f"{key}={value}" for key, value in mongo_atlas_connection_params.items()])
mongo_atlas_connection_string = (
    f"mongodb+srv://{mongo_atlas_username}:{mongo_atlas_password}@{mongo_atlas_host}/?{mongo_atlas_params_string}"
)

embedding_dim = 768
embedding_model = "sentence-transformers/all-mpnet-base-v2"

# Provide option for a different sentance transformer with 1024 dim embeddings
if len(sys.argv) > 1:
    match sys.argv[1]:
        case "1024":
            embedding_dim = 1024
            embedding_model = "BAAI/bge-large-en-v1.5"
            print(f"\nWill use {embedding_model} transformer with dim = {embedding_dim}.\n")

document_store = MongoAtlasDocumentStore(
    mongo_connection_string=mongo_atlas_connection_string,
    database_name=mongo_atlas_database,
    collection_name=mongo_atlas_collection,
    embedding_dim=embedding_dim,
)

document_store.delete_all_documents()
document_store.write_documents(processed_documents)

retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model=embedding_model,
    model_format="sentence_transformers",
    top_k=10,
)

document_store.update_embeddings(retriever, batch_size=60)
