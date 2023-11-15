import re
from typing import Any, Dict, Generator, List, Optional, Union
import numpy as np
import pymongo
from haystack.document_stores import BaseDocumentStore
from haystack.errors import DocumentStoreError
from haystack.nodes.retriever import DenseRetriever
from haystack.schema import Document, FilterType
from haystack.utils import get_batches_from_generator
from pymongo import InsertOne, ReplaceOne, UpdateOne
from pymongo.collection import Collection
from tqdm import tqdm
from .mongo_converters import mongo_doc_to_haystack_doc, haystack_doc_to_mongo_doc
from .mongo_filters import mongo_filter_converter

METRIC_TYPES = ["euclidean", "cosine", "dotProduct"]
DEFAULT_BATCH_SIZE = 50

FilterType = Dict[str, Union[Dict[str, Any], List[Any], str, int, float, bool]]


class MongoAtlasDocumentStore(BaseDocumentStore):
    def __init__(
        self,
        mongo_connection_string: Optional[str] = None,
        database_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_dim: int = 768,
        return_embedding: bool = False,
        index: str = "document",
        similarity: str = "cosine",
        embedding_field: str = "embedding",
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        recreate_index: bool = False,
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
            self._get_collection().create_index("id", unique=True)

    def _create_document_field_map(self) -> Dict:
        """
        [Done]
        [Demanded by base class]
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

        :param index: Optional collection name. If `None`, the DocumentStore's default collection will be used.
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
        [Done]
        [Demanded by base class]
        Deletes the collection named by index or the collection speicifed when the
        driver was initialized.
        """
        self._get_collection(index).drop()

    def delete_labels(self):
        """
        [Done]
        [Demanded by base class]
        """
        raise NotImplementedError("MongoDocumentStore does not support labels (yet).")

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        [Done]
        [BaseDocumentStore]
        [Demanded by base class]
        Retrieves all documents in the index (collection).

        :param index: Optional collection name. If `None`, the DocumentStore's default collection will be used.
        :param filters: Optional filters to narrow down the documents that will be retrieved.
            Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
            operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
            `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
            Logical operator keys take a dictionary of metadata field names and/or logical operators as
            value. Metadata field names take a dictionary of comparison operators as value. Comparison
            operator keys take a single value or (in case of `"$in"`) a list of values as value.
            If no logical operator is provided, `"$and"` is used as default operation. If no comparison
            operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
            operation.
                __Example__:

                ```python
                filters = {
                    "$and": {
                        "type": {"$eq": "article"},
                        "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                        "rating": {"$gte": 3},
                        "$or": {
                            "genre": {"$in": ["economy", "politics"]},
                            "publisher": {"$eq": "nytimes"}
                        }
                    }
                }
                ```
            Note that filters will be acting on the contents of the meta field of the documents in the collection.
        :param return_embedding: Optional flag to return the embedding of the document.
        :param batch_size: Number of documents to process at a time. When working with large number of documents,
                           batching can help reduce memory footprint.
        :param headers: MongoDocumentStore does not support headers.
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

    def get_all_labels(self):
        """
        [Done]
        [Demanded by base class]
        """
        raise NotImplementedError("MongoDocumentStore does not support labels (yet).")

    def get_document_count(
        self,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        [Done]
        [Demanded by base class]

        Return the number of documents.

        :param filters: Optional filters (see get_all_documents for description).
        :param index: Optional collection name. If `None`, the DocumentStore's default collection will be used.
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
        [Done]
        [P / Q / W Have this]

        Return the number of documents with embeddings.
        """
        collection = self._get_collection(index)

        filters = filters or {}

        mongo_filters = {"$and": [mongo_filter_converter(filters), {"embedding": {"$ne": None}}]}

        return collection.count_documents(mongo_filters)

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        """
        [Done]
        [BaseDocumentStore]
        [Demanded by base class]
        Retrieves all documents in the index (collection). Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.

        :param index: Optional collection name. If `None`, the DocumentStore's default collection will be used.
        :param filters: optional filters (see get_all_documents for description).
        :param return_embedding: Optional flag to return the embedding of the document.
        :param batch_size: Number of documents to process at a time. When working with large number of documents,
                           batching can help reduce memory footprint.
        :param headers: MongoDocumentStore does not support headers.
        """
        if headers:
            raise NotImplementedError("MongoDocumentStore does not support headers.")

        mongo_filters = mongo_filter_converter(filters)

        if return_embedding is None:
            return_embedding = self.return_embedding

        projection = {"embedding": False} if not return_embedding else {}

        collection = self._get_collection(index)
        documents = collection.find(mongo_filters, batch_size=batch_size, projection=projection)

        for doc in documents:
            yield mongo_doc_to_haystack_doc(doc)

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        headers: Optional[Dict[str, str]] = None,
        return_embedding: Optional[bool] = None,
    ) -> List[Document]:
        """
        [Done]
        [Demanded by base class]
        Retrieves all documents matching ids.

        :param ids: List of IDs to retrieve.
        :param index: Optional collection name. If `None`, the DocumentStore's default collection will be used.
        :param batch_size: Number of documents to retrieve at a time. When working with large number of documents,
            batching can help reduce memory footprint.
        :param headers: MongoDocumentStore does not support headers.
        :param return_embedding: Optional flag to return the embedding of the document.
        """
        mongo_filters = {"id": {"$in": ids}}

        result = self.get_all_documents_generator(
            index=index,
            filters=mongo_filters,
            return_embedding=return_embedding,
            batch_size=batch_size,
            headers=headers,
        )

        return list(result)

    def get_document_by_id(
        self,
        id: str,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        return_embedding: Optional[bool] = None,
    ) -> Document:
        """
        [Done]
        [Demanded by base class]
        Retrieves the document matching id.

        :param id: The ID of the document to retrieve
        :param index: Optional collection name. If `None`, the DocumentStore's default collection will be used.
        :param headers: MongoDocumentStore does not support headers.
        :param return_embedding: Optional flag to return the embedding of the document.
        """
        documents = self.get_documents_by_id(ids=[id], index=index, headers=headers, return_embedding=return_embedding)
        return documents[0]

    def get_label_count(self):
        """
        [Done]
        [Demanded by base class]
        """
        raise NotImplementedError("MongoDocumentStore does not support labels (yet).")

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
    ) -> List[Document]:
        """
        [Done]
        [Demanded by base class]
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

        :param query_emb: Embedding of the query
        :param filters: optional filters (see get_all_documents for description).
        :param top_k: How many documents to return.
        :param index: Optional collection name. If `None`, the DocumentStore's default collection will be used.
        :param return_embedding: Whether to return document embedding.
        :param headers: MongoDocumentStore does not support headers.
        :param scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                            If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value range will be scaled to a range of [0,1], where 1 means extremely relevant.
                            Otherwise raw similarity scores (e.g. cosine or dot_product) will be used.
        """
        if headers:
            raise NotImplementedError("MongoDocumentStore does not support headers.")

        if return_embedding is None:
            return_embedding = self.return_embedding

        collection = self._get_collection(index)

        query_emb = query_emb.astype(np.float32)

        if self.similarity == "cosine":
            self.normalize_embedding(query_emb)

        filters = filters or {}

        pipeline = [
            {
                "$search": {
                    "index": self.collection_name,
                    "knnBeta": {
                        "vector": query_emb.tolist(),
                        "path": "embedding",
                        "k": top_k,
                    },
                }
            }
        ]
        if filters is not None:
            pipeline.append({"$match": mongo_filter_converter(filters)})
        if not return_embedding:
            pipeline.append({"$project": {"embedding": False}})
        pipeline.append({"$set": {"score": {"$meta": "searchScore"}}})
        documents = list(collection.aggregate(pipeline))

        if scale_score:
            for doc in documents:
                doc["score"] = self.scale_to_unit_interval(doc["score"], self.similarity)

        documents = [mongo_doc_to_haystack_doc(doc) for doc in documents]
        return documents

    def update_document_meta(self, id: str, meta: Dict[str, str], index: Optional[str] = None):
        """
        [Done]
        [Demanded by base class]
        Update the metadata dictionary of a document by specifying its string ID.

        :param id: ID of the Document to update.
        :param meta: Dictionary of new metadata.
        :param index: Optional collection name. If `None`, the DocumentStore's default collection will be used.
        """
        collection = self._get_collection(index)
        collection.update_one({"id": id}, {"$set": {"meta": meta}})

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        [Done]
        [BaseDocumentStore]
        [Demanded by base class]

        Parameters:

        documents: List of `Dicts` or `Documents`
        index (str): search index name - contain letters, numbers, hyphens, or underscores
        :param duplicate_documents: handle duplicate documents based on parameter options.
            Parameter options:
                - `"overwrite"`: Update any existing documents with the same ID when adding documents.
                - `"skip"`: Ignore the duplicate documents.
                - `"fail"`: An error is raised if the document ID of the document being added already exists.

                "overwrite" is the default behaviour.
        """
        if headers:
            raise NotImplementedError("MongoDocumentStore does not support headers.")

        collection = self._get_collection(index)

        duplicate_documents = duplicate_documents or self.duplicate_documents

        field_map = self._create_document_field_map()
        documents = [
            Document.from_dict(doc, field_map=field_map) if isinstance(doc, dict) else doc for doc in documents
        ]

        mongo_documents = list(map(Document.to_dict, documents))

        with tqdm(
            total=len(mongo_documents),
            disable=not self.progress_bar,
            position=0,
            unit=" docs",
            desc="Writing Documents",
        ) as progress_bar:
            batches = get_batches_from_generator(mongo_documents, batch_size)
            for batch in batches:
                operations = [ReplaceOne({"id": doc["id"]}, upsert=True, replacement=doc) for doc in batch]

                match duplicate_documents:
                    case "skip":
                        operations = [UpdateOne({"id": doc["id"]}, {"$setOnInsert": doc}, upsert=True) for doc in batch]
                    case "fail":
                        operations = [InsertOne(doc) for doc in batch]
                    case _:
                        operations = [ReplaceOne({"id": doc["id"]}, upsert=True, replacement=doc) for doc in batch]

                collection.bulk_write(operations)
                progress_bar.update(len(batch))

    def write_labels(self):
        """
        [Done]
        [Demanded by base class]
        """
        raise NotImplementedError("MongoDocumentStore does not support labels (yet).")

    def update_embeddings(
        self,
        retriever: DenseRetriever,
        index: Optional[str] = None,
        update_existing_embeddings: bool = True,
        filters: Optional[FilterType] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        [Done]
        [P / Q / W Have this]
        Updates the embeddings in the document store using the encoding model specified in the retriever.

        This can be useful if you want to add or change the embeddings for your documents (e.g. after changing the
        retriever config).

        :param retriever: Retriever to use to get embeddings for text.
        :param index: Optional collection name. If `None`, the DocumentStore's default collection will be used.
        :param update_existing_embeddings: Whether to update existing embeddings of the documents. If set to `False`,
            only documents without embeddings are processed. This mode can be used for incremental updating of
            embeddings, wherein, only newly indexed documents get processed.
        :param filters: optional filters (see get_all_documents for description).
        :param batch_size: Number of documents to process at a time. When working with large number of documents,
            batching can help reduce memory footprint. "
        """
        filters = filters or {}
        document_count = self.get_document_count(
            index=index, filters=filters, only_documents_without_embedding=not update_existing_embeddings
        )

        if not update_existing_embeddings:
            filters = {"$and": [filters, {"embedding": {"$eq": None}}]}

        documents = self.get_all_documents_generator(
            index=index, filters=filters, return_embedding=False, batch_size=batch_size
        )

        collection = self._get_collection(index)

        with tqdm(
            total=document_count,
            disable=not self.progress_bar,
            unit=" docs",
            desc="Updating Embeddings",
        ) as progress_bar:
            batches = get_batches_from_generator(documents, batch_size)
            for batch in batches:
                embeddings = retriever.embed_documents(batch)
                self._validate_embeddings_shape(
                    embeddings=embeddings, num_documents=len(batch), embedding_dim=self.embedding_dim
                )
                if self.similarity == "cosine":
                    self.normalize_embedding(embeddings)

                mongo_documents = [haystack_doc_to_mongo_doc(doc) for doc in batch]

                for doc, embedding in zip(mongo_documents, embeddings.tolist()):
                    doc["embedding"] = embedding

                updates = [ReplaceOne({"id": doc["id"]}, doc) for doc in mongo_documents]
                collection.bulk_write(updates)
                progress_bar.update(len(batch))


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
