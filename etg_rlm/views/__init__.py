"""Diverse verification view implementations (Section 1 of eval plan).

Each of the N views is a separate pathway for checking whether a claim
is supported by evidence. The implementation plan specifies N=5 views
that differ across retrieval strategy, chunking, and query formulation.
"""

from etg_rlm.views.factory import (
    ViewConfig,
    ViewType,
    create_view,
    create_default_view_suite,
)

__all__ = [
    "ViewConfig",
    "ViewType",
    "create_view",
    "create_default_view_suite",
]
