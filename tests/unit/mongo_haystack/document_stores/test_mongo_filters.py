import pytest
from mongo_haystack.document_stores.mongo_filters import _target_filter_to_metadata, _and_or_to_list

pytestmark = pytest.mark.unit

# _target_filter_to_metadata


def test__target_filter_to_metadata_01():
    test_filter = {
        "url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes",
        "_split_id": 0,
    }

    target_outcome = {
        "meta.url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes",
        "meta._split_id": 0,
    }

    assert _target_filter_to_metadata(test_filter, "meta") == target_outcome


def test__target_filter_to_metadata_02():
    test_filter = {
        "$and": [
            {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
            {"_split_id": 0},
        ]
    }

    target_outcome = {
        "$and": [
            {"meta.url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
            {"meta._split_id": 0},
        ]
    }

    assert _target_filter_to_metadata(test_filter, "meta") == target_outcome


def test__target_filter_to_metadata_03():
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


def test__target_filter_to_metadata_leave_id():
    test_filter = {"id": {"$in": ["b714102aa7ac3a9622d0d00caa55fa", "b3de1a673c1eb2876585405395a10c3d"]}}

    target_outcome = {"id": {"$in": ["b714102aa7ac3a9622d0d00caa55fa", "b3de1a673c1eb2876585405395a10c3d"]}}

    assert _target_filter_to_metadata(test_filter, "meta") == target_outcome


# _and_or_to_list


def test__and_or_to_list_01():
    test_filter = {"$and": {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes", "_split_id": 0}}

    target_outcome = {
        "$and": [
            {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
            {"_split_id": 0},
        ]
    }

    assert _and_or_to_list(test_filter) == target_outcome


def test__and_or_to_list_02():
    test_filter = {
        "$and": {
            "url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes",
            "_split_id": 0,
            "$or": {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes", "_split_id": 0},
        }
    }

    target_outcome = {
        "$and": [
            {"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"},
            {"_split_id": 0},
            {"$or": [{"url": "https://en.wikipedia.org/wiki/Colossus_of_Rhodes"}, {"_split_id": 0}]},
        ]
    }

    assert _and_or_to_list(test_filter) == target_outcome
