import pytest
from mongo_haystack.document_stores.mongo_filters import _target_filter_to_metadata, _and_or_to_list, mongo_filter_converter

pytestmark = pytest.mark.integration


def test_and_or_meta_converted():
    test_filter = {
        "$and": [
            {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
            {"_split_id": 0},
            {
                "$or": [
                    {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
                    {"_split_id": 0},
                ]
            },
        ]
    }

    target_outcome = {
        "$and": [
            {"meta.url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
            {"meta._split_id": 0},
            {
                "$or": [
                    {"meta.url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
                    {"meta._split_id": 0},
                ]
            },
        ]
    }
    assert _and_or_to_list(_target_filter_to_metadata(test_filter, "meta")) == target_outcome


def test_mongo_filter_converter_and_or_meta_converted():
    test_filter = {
        "$and": [
            {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
            {"_split_id": 0},
            {
                "$or": [
                    {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
                    {"_split_id": 0},
                ]
            },
        ]
    }

    target_outcome = {
        "$and": [
            {"meta.url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
            {"meta._split_id": 0},
            {
                "$or": [
                    {"meta.url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
                    {"meta._split_id": 0},
                ]
            },
        ]
    }

    assert mongo_filter_converter(test_filter) == target_outcome


def test_mongo_filter_converter_falsey_empty_dict():
    test_filter = {
        "$and": [
            {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
            {"_split_id": 0},
            {
                "$or": [
                    {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
                    {"_split_id": 0},
                ]
            },
        ]
    }

    target_outcome = {
        "$and": [
            {"meta.url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
            {"meta._split_id": 0},
            {
                "$or": [
                    {"meta.url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
                    {"meta._split_id": 0},
                ]
            },
        ]
    }

    assert mongo_filter_converter(None) == {}
    assert mongo_filter_converter("") == {}
    assert mongo_filter_converter({}) == {}
    assert mongo_filter_converter([]) == {}
    assert mongo_filter_converter(0) == {}
