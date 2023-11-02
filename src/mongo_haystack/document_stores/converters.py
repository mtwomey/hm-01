from haystack.schema import Document


def mongo_doc_to_hystack_doc(mongo_doc):
    return Document(
        id=mongo_doc["id"], content=mongo_doc["content"], content_type=mongo_doc["content_type"], meta=mongo_doc["meta"]
    )


def haystack_doc_to_mongo_doc(haystack_doc):
    return {
        "id": haystack_doc.id,
        "content": haystack_doc.content,
        "content_type": haystack_doc.content_type,
        "meta": haystack_doc.meta,
    }
