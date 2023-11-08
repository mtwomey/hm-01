import re
from itertools import islice
from typing import Any, Dict, Generator, List, Optional, Set, Union

import numpy as np
import pymongo
from haystack.document_stores import BaseDocumentStore
from haystack.errors import DocumentStoreError
from haystack.nodes.retriever import DenseRetriever
from haystack.schema import Answer, Document, FilterType, Label, Span
from pymongo import ReplaceOne
from pymongo.collection import Collection
from pymongo.write_concern import WriteConcern
from tqdm import tqdm
from . import filters

import json

from mongo_haystack.document_stores.converters import (
    haystack_doc_to_mongo_doc,
    mongo_doc_to_hystack_doc,
)
from mongo_haystack.document_stores.filters import mongo_filter_converter
from pprint import PrettyPrinter

pp = PrettyPrinter(indent=4).pprint

METRIC_TYPES = ["euclidean", "cosine", "dotProduct"]

DEFAULT_BATCH_SIZE = 50

FilterType = Dict[str, Union[Dict[str, Any], List[Any], str, int, float, bool]]


class MongoDocumentStore(BaseDocumentStore):
    def __init__(
        self,
        mongo_connection_string: Optional[str] = None,
        database_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_dim: int = 768,
        return_embedding: bool = False,
        index: str = "document",
        similarity: str = "cosine",
        replicas: int = 1,
        shards: int = 1,
        embedding_field: str = "embedding",
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        recreate_index: bool = False,
        metadata_config: Optional[Dict] = None,
        validate_index_sync: bool = True,
    ):
        self.mongo_connection_string = _validate_mongo_connection_string(mongo_connection_string)
        self.database_name = _validate_database_name(database_name)
        self.collection_name = _validate_collection_name(collection_name)
        self.connection = pymongo.MongoClient(self.mongo_connection_string)
        self.database = self.connection[self.database_name]
        self.similarity = _validate_similarity(similarity)
        self.duplicate_documents = duplicate_documents
        self.embedding_field = embedding_field
        self.progress_bar = progress_bar
        self.embedding_dim = embedding_dim
        self.index = collection_name
        self.return_embedding = return_embedding
        self.recreate_index = recreate_index

        if self.recreate_index:
            self.delete_index()

        # Implicitly create the collection if it doesn't exist
        if collection_name not in self.database.list_collection_names():
            self.database.create_collection(self.collection_name)

    def _create_document_field_map(self) -> Dict:
        """
        [Demanded by base class]
        [Done]
        """
        return {self.embedding_field: "embedding"}

    def _get_collection(self, index=None) -> Collection:
        """
        [Done]
        Returns the collection named by index or returns the collection speicifed when the
        driver was initialized.
        """
        _validate_index_name(index)
        if index is not None:
            return self.database[index]
        else:
            return self.database[self.collection_name]

    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        [Done]
        [Demanded by base class]

        Delete documents from the document store.

        :param index: Collection to delete the documents from. If `None`, the DocumentStore's default collection will be used.
        :param ids: Optional list of IDs to narrow down the documents to be deleted.
        :param filters: optional filters (see get_all_documents for description).
            If filters are provided along with a list of IDs, this method deletes the
            intersection of the two query results (documents that match the filters and
            have their ID in the list).
        :param headers: MongoDocumentStore does not support headers.
        :return None:
        """
        if headers:
            raise NotImplementedError("MongoDocumentStore does not support headers.")

        collection = self._get_collection(index)

        match (ids, filters):
            case (None, None):
                mongo_filters = {}
            case (None, filters):
                mongo_filters = mongo_filter_converter(filters)
            case (ids, None):
                mongo_filters = {"id": {"$in": ids}}
            case (ids, filters):
                mongo_filters = {"$and": [mongo_filter_converter(filters), {"id": {"$in": ids}}]}

        collection.delete_many(filter=mongo_filters)

    def delete_index(self, index=None):
        """
        [Demanded by base class]
        """
        self._get_collection(index).drop()

    def delete_labels():
        """
        [Demanded by base class]
        """
        pass

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        [BaseDocumentStore]
        [Demanded by base class]
        """
        if headers:
            raise NotImplementedError("MongoDocumentStore does not support headers.")

        result = self.get_all_documents_generator(
            index=index,
            filters=filters,
            return_embedding=return_embedding,
            batch_size=batch_size,
        )
        return list(result)

    def get_all_labels():
        """
        [Demanded by base class]
        """
        pass

    def get_document_count(
        self,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        [Demanded by base class]
        [Done]

        Return the number of documents.

        :param filters: Optional filters (see get_all_documents).
        :param index: Collection to use.
        :param only_documents_without_embedding: If set to `True`, only documents without embeddings are counted.
        :param headers: MongoDocumentStore does not support headers.
        """
        if headers:
            raise NotImplementedError("MongoDocumentStore does not support headers.")

        collection = self._get_collection(index)

        if only_documents_without_embedding:
            mongo_filter = {"$and": [mongo_filter_converter(filters), {"embedding": {"$eq": None}}]}
        else:
            mongo_filter = mongo_filter_converter(filters)

        return collection.count_documents(mongo_filter)

    def get_embedding_count(self, filters: Optional[FilterType] = None, index: Optional[str] = None) -> int:
        """
        [P / Q / W Have this]

        Return the number of documents with embeddings.
        """
        return self.collection.count_documents({"embedding": {"$ne": None}})

    def get_all_documents_generator(
        self,
        index: Optional[FilterType] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        """
        [BaseDocumentStore]
        [Demanded by base class]
        """
        if headers:
            raise NotImplementedError("MongoDocumentStore does not support headers.")

        mongo_filters = mongo_filter_converter(filters)

        projection = {}

        if return_embedding is None:
            return_embedding = self.return_embedding

        if not return_embedding:
            projection = {"embedding": False}

        collection = self._get_collection(index)
        documents = collection.find(mongo_filters, batch_size=batch_size, projection=projection)
        for document in documents:
            yield mongo_doc_to_hystack_doc(document)

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        headers: Optional[Dict[str, str]] = None,
        return_embedding: Optional[bool] = None,
        namespace: Optional[str] = None,
    ) -> List[Document]:
        """
        [Demanded by base class]
        """
        pass

    def get_document_by_id():
        """
        [Demanded by base class]
        """
        pass

    def get_label_count():
        """
        [Demanded by base class]
        """
        pass

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = True,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
    ) -> List[Document]:
        """
        [Demanded by base class]
        """

        if headers:
            raise NotImplementedError("MongoDocumentStore does not support headers.")

        if return_embedding is None:
            return_embedding = self.return_embedding

        collection = self._get_collection(index)

        query_emb = query_emb.astype(np.float32)
        # print(query_emb)

        if self.similarity == "cosine":
            self.normalize_embedding(query_emb)

        filters = filters or {}

        results = collection.aggregate(
            [
                {
                    "$search": {
                        "index": self.collection_name,
                        "knnBeta": {
                            "vector": query_emb.tolist(),
                            "path": "embedding",
                            "k": top_k,
                        },
                    }
                },
                {"$set": {"score": {"$meta": "searchScore"}}},
            ]
        )

        documents = []
        for result in results:
            document = Document(id=result["id"], content=result["content"], score=result["score"], meta=result["meta"])
            if return_embedding:
                document.embedding = np.asarray(result["embedding"], dtype=np.float32)
            documents.append(document)

        return documents

    def update_document_meta():
        """
        [Demanded by base class]
        """
        pass

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        labels: Optional[bool] = False,
    ):
        """
        [BaseDocumentStore]
        [Demanded by base class]

        Parameters:

        documents: List of `Dicts` or `Documents`
        index (str): search index name - contain letters, numbers, hyphens, or underscores
        """

        collection = self._get_collection(index)

        duplicate_documents = duplicate_documents or self.duplicate_documents

        field_map = self._create_document_field_map()
        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]

        mongo_documents = list(map(Document.to_dict, document_objects))
        collection.with_options(write_concern=WriteConcern(w=0)).insert_many(mongo_documents, ordered=False)

    def write_labels():
        """
        [Demanded by base class]
        """
        pass

    def update_embeddings(
        self,
        retriever: DenseRetriever,
        index: Optional[str] = None,
        update_existing_embeddings: bool = True,
        filters: Optional[FilterType] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        [P / Q / W Have this]
        """
        document_count = self.get_document_count(
            index=index, filters=filters, only_documents_without_embedding=not update_existing_embeddings
        )

        documents = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=False, batch_size=batch_size
        )

        collection = self._get_collection(index)

        with tqdm(
            total=document_count, disable=not self.progress_bar, position=0, unit=" docs", desc="Updating Embedding"
        ) as progress_bar:
            for _ in range(0, document_count, batch_size):
                document_batch = list(islice(documents, batch_size))
                embeddings = retriever.embed_documents(document_batch)
                self._validate_embeddings_shape(
                    embeddings=embeddings, num_documents=len(document_batch), embedding_dim=self.embedding_dim
                )

                if self.similarity == "cosine":
                    self.normalize_embedding(embeddings)

                mongo_docs_batch = list(map(lambda doc: haystack_doc_to_mongo_doc(doc), document_batch))

                for doc, embedding in zip(mongo_docs_batch, embeddings.tolist()):
                    doc["embedding"] = embedding

                requests = list(map(lambda doc: (ReplaceOne({"id": doc["id"]}, doc)), mongo_docs_batch))

                collection.bulk_write(requests)
                progress_bar.update(len(document_batch))


class MongoDocumentStoreError(DocumentStoreError):
    """Exception for issues that occur in a Pinecone document store"""

    def __init__(self, message: Optional[str] = None):
        super().__init__(message=message)


class ValidationError(Exception):
    """Exception for validation errors"""

    pass


def _validate_mongo_connection_string(mongo_connection_string):
    if not mongo_connection_string:
        raise MongoDocumentStoreError(
            "A `mongo_connection_string` is required. This can be obtained on the MongoDB Atlas Dashboard by clicking on the `CONNECT` button."
        )
    return mongo_connection_string


def _validate_database_name(database_name):
    # There doesn't seem to be much restriction on the name here? All sorts of special character are apparently allowed...
    # Just check if it's there.
    if not database_name:
        raise ValidationError("A `database_name` is required.")
    return database_name


def _validate_collection_name(collection_name):
    # There doesn't seem to be much restriction on the name here? All sorts of special character are apparently allowed...
    # Just check if it's there.
    if not collection_name:
        raise ValidationError("A `collection_name` is required.")
    return collection_name


def _validate_similarity(similarity):
    if similarity not in METRIC_TYPES:
        raise ValueError(
            "Mongo Atlas currently supports dotProduct, cosine and euclidean metrics. Please set similarity to one of the above."
        )
    return similarity


def _validate_index_name(index_name):
    if index_name and not bool(re.match(r"^[a-zA-Z0-9\-_]+$", index_name)):
        raise ValueError(
            f'Invalid index name: "{index_name}". Index name can only contain letters, numbers, hyphens, or underscores.'
        )
    return index_name
