import math

from app.vectorstore.similarity import cosine_similarity


def test_identical_vectors_score_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_orthogonal_vectors_score_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_opposite_vectors_score_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0


def test_magnitude_does_not_affect_score():
    # Same direction, different magnitude -> still 1.0 (normalized).
    assert math.isclose(cosine_similarity([1.0, 1.0], [5.0, 5.0]), 1.0)


def test_zero_vector_scores_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
