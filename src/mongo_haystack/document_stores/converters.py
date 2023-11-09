from haystack.schema import Document
from typing import Dict


def mongo_doc_to_hystack_doc(mongo_doc) -> Document:
    if "embedding" in mongo_doc:
        embedding = mongo_doc["embedding"]
    else:
        embedding = None

    if "score" in mongo_doc:
        score = mongo_doc["score"]
    else:
        score = None

    return Document(
        id=mongo_doc["id"],
        content=mongo_doc["content"],
        content_type=mongo_doc["content_type"],
        meta=mongo_doc["meta"],
        embedding=embedding,
        score=score,
    )


def haystack_doc_to_mongo_doc(haystack_doc) -> Dict:
    return {
        "id": haystack_doc.id,
        "content": haystack_doc.content,
        "content_type": haystack_doc.content_type,
        "meta": haystack_doc.meta,
    }
