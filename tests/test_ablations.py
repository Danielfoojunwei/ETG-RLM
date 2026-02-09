"""Tests for the ablation study configurations module."""

import pytest

from etg_rlm.ablations import (
    AblationConfig,
    AblationType,
    RandomPolicy,
    all_ablation_configs,
    make_no_constraint_config,
    make_no_multi_view_config,
    make_policy_ablation_config,
    make_threshold_sweep_configs,
)
from etg_rlm.core import (
    AtomicClaim,
    ClaimStatus,
    ESBGNode,
    EvidenceScopedBeliefGraph,
)
from etg_rlm.pipeline import ETGConfig
from etg_rlm.policy import ActionType


class TestAblationTypes:
    def test_four_types(self):
        assert len(AblationType) == 4

    def test_all_types(self):
        types = {t.value for t in AblationType}
        assert types == {
            "no_multi_view",
            "no_constraint",
            "threshold_sweep",
            "policy_ablation",
        }


class TestNoMultiViewConfig:
    def test_min_views_is_one(self):
        config = make_no_multi_view_config()
        assert config.etg_config.min_views_per_claim == 1

    def test_name(self):
        config = make_no_multi_view_config()
        assert config.name == "ETG-NoMultiView"

    def test_preserves_base_tau(self):
        base = ETGConfig(tau=0.8)
        config = make_no_multi_view_config(base)
        assert config.etg_config.tau == 0.8


class TestNoConstraintConfig:
    def test_tau_is_zero(self):
        config = make_no_constraint_config()
        assert config.etg_config.tau == 0.0
        assert config.etg_config.tau_prime == 0.0

    def test_allows_uncertain(self):
        config = make_no_constraint_config()
        assert config.etg_config.allow_uncertain is True

    def test_name(self):
        config = make_no_constraint_config()
        assert config.name == "ETG-NoConstraint"


class TestThresholdSweepConfigs:
    def test_default_thresholds(self):
        configs = make_threshold_sweep_configs()
        assert len(configs) == 5

    def test_tau_values(self):
        configs = make_threshold_sweep_configs()
        taus = [c.etg_config.tau for c in configs]
        assert taus == pytest.approx([0.5, 0.6, 0.7, 0.8, 0.9])

    def test_custom_thresholds(self):
        configs = make_threshold_sweep_configs(thresholds=(0.3, 0.5, 0.7))
        assert len(configs) == 3

    def test_names(self):
        configs = make_threshold_sweep_configs()
        for config in configs:
            assert config.name.startswith("ETG-tau=")

    def test_ablation_type(self):
        configs = make_threshold_sweep_configs()
        for config in configs:
            assert config.ablation_type == AblationType.THRESHOLD_SWEEP


class TestPolicyAblationConfig:
    def test_name(self):
        config = make_policy_ablation_config()
        assert config.name == "ETG-RandomPolicy"

    def test_preserves_base_config(self):
        base = ETGConfig(tau=0.8, verification_budget=100)
        config = make_policy_ablation_config(base)
        assert config.etg_config.tau == 0.8
        assert config.etg_config.verification_budget == 100


class TestRandomPolicy:
    def _make_esbg(self, n_claims: int = 3) -> EvidenceScopedBeliefGraph:
        g = EvidenceScopedBeliefGraph()
        for i in range(n_claims):
            g.add_node(ESBGNode(
                node_id=f"c{i}",
                claim=AtomicClaim(claim_id=f"c{i}", text=f"Claim {i}"),
            ))
        return g

    def test_selects_action(self):
        policy = RandomPolicy(seed=42)
        esbg = self._make_esbg()
        action = policy.select_action("query", "corpus", esbg, budget_remaining=10)
        assert action.action_type == ActionType.RUN_VIEW
        assert action.target_node_id is not None

    def test_stops_when_no_budget(self):
        policy = RandomPolicy()
        esbg = self._make_esbg()
        action = policy.select_action("query", "corpus", esbg, budget_remaining=0)
        assert action.action_type == ActionType.STOP

    def test_stops_when_empty_graph(self):
        policy = RandomPolicy()
        esbg = EvidenceScopedBeliefGraph()
        action = policy.select_action("query", "corpus", esbg, budget_remaining=10)
        assert action.action_type == ActionType.STOP

    def test_stops_when_all_maxed(self):
        policy = RandomPolicy(max_views_per_claim=1)
        esbg = self._make_esbg(n_claims=1)
        node = esbg.get_node("c0")
        node.view_verdicts = [True]  # already has 1 view
        action = policy.select_action("query", "corpus", esbg, budget_remaining=10)
        assert action.action_type == ActionType.STOP

    def test_deterministic_with_seed(self):
        esbg = self._make_esbg()
        actions1 = []
        policy1 = RandomPolicy(seed=123)
        for _ in range(5):
            a = policy1.select_action("q", "c", esbg, budget_remaining=50)
            actions1.append(a.target_node_id)

        actions2 = []
        policy2 = RandomPolicy(seed=123)
        for _ in range(5):
            a = policy2.select_action("q", "c", esbg, budget_remaining=50)
            actions2.append(a.target_node_id)

        assert actions1 == actions2


class TestAllAblationConfigs:
    def test_generates_all(self):
        configs = all_ablation_configs()
        # 1 + 1 + 5 + 1 = 8
        assert len(configs) == 8

    def test_types_present(self):
        configs = all_ablation_configs()
        types = {c.ablation_type for c in configs}
        assert types == {
            AblationType.NO_MULTI_VIEW,
            AblationType.NO_CONSTRAINT,
            AblationType.THRESHOLD_SWEEP,
            AblationType.POLICY_ABLATION,
        }

    def test_names_unique(self):
        configs = all_ablation_configs()
        names = [c.name for c in configs]
        assert len(names) == len(set(names))
