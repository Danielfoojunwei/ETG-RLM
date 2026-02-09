"""Human evaluation protocol for ETG (Section 2.2 of experimental design).

Implements the human evaluation framework:

Task 1: Faithfulness Rating (5-point scale)
    5 = Fully Faithful: All claims supported by source
    4 = Mostly Faithful: Minor unsupported details
    3 = Partially Faithful: Mix of supported/unsupported
    2 = Mostly Unfaithful: Majority unsupported
    1 = Completely Unfaithful: Largely fabricated

Task 2: Pairwise Preference (ETG vs. baseline)
    Annotators judge: (a) More helpful, (b) More trustworthy, (c) Overall better

Inter-Annotator Agreement: Fleiss' Kappa (target kappa >= 0.6)
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Faithfulness rating (Task 1)
# ---------------------------------------------------------------------------


class FaithfulnessRating(IntEnum):
    """5-point faithfulness scale (Section 2.2, Task 1)."""

    COMPLETELY_UNFAITHFUL = 1
    MOSTLY_UNFAITHFUL = 2
    PARTIALLY_FAITHFUL = 3
    MOSTLY_FAITHFUL = 4
    FULLY_FAITHFUL = 5


@dataclass(frozen=True)
class FaithfulnessAnnotation:
    """A single annotator's faithfulness rating for one instance.

    Attributes:
        instance_id: the evaluation instance
        system_name: which system produced the answer
        annotator_id: which annotator
        rating: the faithfulness rating (1-5)
    """

    instance_id: str
    system_name: str
    annotator_id: str
    rating: FaithfulnessRating


# ---------------------------------------------------------------------------
# Pairwise preference (Task 2)
# ---------------------------------------------------------------------------


class PreferenceDimension(Enum):
    """Dimensions for pairwise preference judgments."""

    MORE_HELPFUL = "more_helpful"
    MORE_TRUSTWORTHY = "more_trustworthy"
    OVERALL_BETTER = "overall_better"


class PreferenceChoice(Enum):
    """Which system the annotator preferred."""

    SYSTEM_A = "system_a"
    SYSTEM_B = "system_b"
    TIE = "tie"


@dataclass(frozen=True)
class PairwiseAnnotation:
    """A single annotator's pairwise preference for one instance.

    Attributes:
        instance_id: the evaluation instance
        system_a: name of system A (anonymized during annotation)
        system_b: name of system B (anonymized during annotation)
        annotator_id: which annotator
        preferences: dict mapping dimension -> choice
    """

    instance_id: str
    system_a: str
    system_b: str
    annotator_id: str
    preferences: dict[PreferenceDimension, PreferenceChoice]


# ---------------------------------------------------------------------------
# Aggregated human evaluation results
# ---------------------------------------------------------------------------


class FaithfulnessAggregation(NamedTuple):
    """Aggregated faithfulness ratings for one system."""

    system_name: str
    mean_rating: float
    median_rating: float
    n_annotations: int
    rating_distribution: dict[int, int]  # rating -> count


class PreferenceAggregation(NamedTuple):
    """Aggregated pairwise preferences for one system pair."""

    system_a: str
    system_b: str
    dimension: PreferenceDimension
    n_comparisons: int
    a_wins: int
    b_wins: int
    ties: int
    a_win_rate: float


def aggregate_faithfulness(
    annotations: list[FaithfulnessAnnotation],
    system_name: str,
) -> FaithfulnessAggregation:
    """Aggregate faithfulness ratings for a single system.

    Args:
        annotations: all annotations (may include other systems)
        system_name: which system to aggregate for

    Returns:
        FaithfulnessAggregation with mean, median, and distribution.
    """
    relevant = [a for a in annotations if a.system_name == system_name]
    if not relevant:
        return FaithfulnessAggregation(
            system_name=system_name,
            mean_rating=0.0,
            median_rating=0.0,
            n_annotations=0,
            rating_distribution={},
        )

    ratings = sorted(a.rating.value for a in relevant)
    n = len(ratings)
    mean = sum(ratings) / n
    median = float(ratings[n // 2]) if n % 2 == 1 else (ratings[n // 2 - 1] + ratings[n // 2]) / 2.0
    distribution = dict(Counter(ratings))

    return FaithfulnessAggregation(
        system_name=system_name,
        mean_rating=mean,
        median_rating=median,
        n_annotations=n,
        rating_distribution=distribution,
    )


def aggregate_preferences(
    annotations: list[PairwiseAnnotation],
    dimension: PreferenceDimension,
) -> PreferenceAggregation:
    """Aggregate pairwise preferences for a specific dimension.

    Args:
        annotations: all pairwise annotations for a system pair
        dimension: which dimension to aggregate

    Returns:
        PreferenceAggregation with win rates and counts.
    """
    if not annotations:
        return PreferenceAggregation(
            system_a="",
            system_b="",
            dimension=dimension,
            n_comparisons=0,
            a_wins=0,
            b_wins=0,
            ties=0,
            a_win_rate=0.0,
        )

    system_a = annotations[0].system_a
    system_b = annotations[0].system_b

    a_wins = 0
    b_wins = 0
    ties = 0

    for ann in annotations:
        choice = ann.preferences.get(dimension)
        if choice == PreferenceChoice.SYSTEM_A:
            a_wins += 1
        elif choice == PreferenceChoice.SYSTEM_B:
            b_wins += 1
        elif choice == PreferenceChoice.TIE:
            ties += 1

    n = a_wins + b_wins + ties
    a_win_rate = a_wins / n if n > 0 else 0.0

    return PreferenceAggregation(
        system_a=system_a,
        system_b=system_b,
        dimension=dimension,
        n_comparisons=n,
        a_wins=a_wins,
        b_wins=b_wins,
        ties=ties,
        a_win_rate=a_win_rate,
    )


# ---------------------------------------------------------------------------
# Fleiss' Kappa for inter-annotator agreement
# ---------------------------------------------------------------------------


def fleiss_kappa(
    ratings_matrix: list[list[int]],
    n_categories: int | None = None,
) -> float:
    """Compute Fleiss' Kappa for inter-annotator agreement.

    Fleiss' Kappa measures agreement among multiple raters assigning
    categorical ratings to a fixed number of items. Used to ensure
    annotation reliability (target kappa >= 0.6).

    Args:
        ratings_matrix: N x k matrix where N = number of items and
            k = number of categories. Each cell (i, j) is the number
            of raters who assigned category j to item i.
        n_categories: total number of categories (inferred if None)

    Returns:
        Fleiss' Kappa statistic in [-1, 1]. Values:
            kappa >= 0.6: acceptable agreement
            kappa >= 0.8: strong agreement
            kappa < 0.0: less than chance agreement

    Raises:
        ValueError: if the matrix is empty or inconsistent.
    """
    if not ratings_matrix:
        raise ValueError("Ratings matrix is empty")

    n_items = len(ratings_matrix)
    if n_categories is None:
        n_categories = len(ratings_matrix[0])

    if n_categories < 2:
        raise ValueError("Need at least 2 categories")

    # Total raters per item (should be constant)
    n_raters = sum(ratings_matrix[0])
    if n_raters < 2:
        raise ValueError("Need at least 2 raters")

    for row in ratings_matrix:
        if len(row) != n_categories:
            raise ValueError(
                f"Inconsistent number of categories: expected {n_categories}, got {len(row)}"
            )
        if sum(row) != n_raters:
            raise ValueError(
                f"Inconsistent number of raters: expected {n_raters}, got {sum(row)}"
            )

    # Step 1: Compute P_i for each item
    p_items = []
    for row in ratings_matrix:
        sum_sq = sum(r * r for r in row)
        p_i = (sum_sq - n_raters) / (n_raters * (n_raters - 1))
        p_items.append(p_i)

    # P_bar: mean of P_i
    p_bar = sum(p_items) / n_items

    # Step 2: Compute P_j for each category (proportion of all ratings in category j)
    p_categories = []
    total_ratings = n_items * n_raters
    for j in range(n_categories):
        col_sum = sum(row[j] for row in ratings_matrix)
        p_j = col_sum / total_ratings
        p_categories.append(p_j)

    # P_e_bar: expected agreement by chance
    p_e_bar = sum(p_j * p_j for p_j in p_categories)

    # Step 3: Kappa
    if abs(1.0 - p_e_bar) < 1e-10:
        # Perfect agreement by chance -- kappa undefined, return 1.0
        return 1.0

    kappa = (p_bar - p_e_bar) / (1.0 - p_e_bar)
    return kappa


def check_annotator_agreement(
    kappa: float,
    threshold: float = 0.6,
) -> bool:
    """Check whether inter-annotator agreement meets the threshold.

    From the experimental design: if kappa < 0.6, annotators should
    be retrained and the evaluation re-run.

    Args:
        kappa: Fleiss' Kappa statistic
        threshold: minimum acceptable agreement (default 0.6)

    Returns:
        True if agreement is acceptable.
    """
    return kappa >= threshold


@dataclass
class HumanEvalSummary:
    """Complete human evaluation summary.

    Attributes:
        n_instances: number of evaluated instances
        n_annotators: number of annotators per instance
        faithfulness: per-system faithfulness aggregations
        preferences: per-dimension preference aggregations
        fleiss_kappa: inter-annotator agreement score
        agreement_acceptable: whether kappa >= threshold
    """

    n_instances: int
    n_annotators: int
    faithfulness: list[FaithfulnessAggregation] = field(default_factory=list)
    preferences: list[PreferenceAggregation] = field(default_factory=list)
    fleiss_kappa: float = 0.0
    agreement_acceptable: bool = False
