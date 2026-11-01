"""Tests for the offline retrieval evaluation script (eval/evaluate.py).

Scoring and QA-set validation are pure functions and run without a store.
The end-to-end test reuses the suite's offline TestClient fixture.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.evaluate import (  # noqa: E402
    first_hit_rank,
    load_qa_set,
    run_evaluation,
    score_results,
)
from tests.conftest import SAMPLE_TXT  # noqa: E402

SOURCES = [
    {"text": "Refunds within 30 days of purchase.", "score": 0.9},
    {"text": "Shipping takes 3 to 5 business days.", "score": 0.7},
]


def test_first_hit_rank_first_second_and_miss():
    assert first_hit_rank(SOURCES, "30 days") == 1
    assert first_hit_rank(SOURCES, "3 to 5 business days") == 2
    assert first_hit_rank(SOURCES, "warranty") is None
    # Matching is case-insensitive.
    assert first_hit_rank(SOURCES, "REFUNDS") == 1


def test_score_results_hit_at_1_hit_at_k_and_mrr():
    per_question = [{"rank": 1}, {"rank": 2}, {"rank": None}]
    summary = score_results(per_question, top_k=2)
    assert summary["total"] == 3
    assert summary["hit@1"] == pytest.approx(1 / 3)
    assert summary["hit@k"] == pytest.approx(2 / 3)
    assert summary["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert summary["misses"] == 1
    with pytest.raises(ValueError):
        score_results([], top_k=4)


def test_load_qa_set_rejects_bad_sets(tmp_path):
    good = [{"id": "q1", "question": "Q?", "expected": "answer"}]
    path = tmp_path / "qa.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    assert load_qa_set(str(path)) == good

    for bad in (
        [],
        [{"id": "q1", "question": "Q?"}],  # missing expected
        [{"id": "q1", "question": " ", "expected": "answer"}],  # blank question
        [
            {"id": "q1", "question": "Q?", "expected": "a"},
            {"id": "q1", "question": "Q2?", "expected": "b"},
        ],  # duplicate id
    ):
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError):
            load_qa_set(str(path))


def test_run_evaluation_end_to_end_offline(client):
    # One question per handbook chunk: each must retrieve its chunk.
    qa_set = [
        {"id": "q01", "question": "How many days to request a refund?", "expected": "30 days"},
        {"id": "q06", "question": "How long does standard shipping take within Europe?", "expected": "3 to 5 business days"},
        {"id": "q11", "question": "What are the password requirements?", "expected": "12 characters"},
    ]
    report = run_evaluation(client, qa_set, SAMPLE_TXT, top_k=4)
    assert report["summary"]["total"] == 3
    assert report["summary"]["misses"] == 0
    assert all(item["rank"] is not None for item in report["results"])
    # Ranking quality, not just coverage: at least two of three first.
    assert report["summary"]["hit@1"] >= 2 / 3


def test_run_evaluation_rejects_bad_top_k(client):
    with pytest.raises(ValueError):
        run_evaluation(client, [], SAMPLE_TXT, top_k=0)
