"""Unit tests for lkg.py (the Living Knowledge Graph).

TestLivingKG below is the module's own original test suite, moved here
unchanged from lkg.py so implementation files in this project don't embed
their own unittest classes (tests/ is where every other module's tests
live -- see test_pipeline.py, test_regressions.py, etc.). The functions
after it cover from_config and observe_entity_transition, both added while
integrating the module into OpenClosedLoopEngine (see CHANGELOG.md) and not
present in the tests that shipped with the original standalone file.
"""
from __future__ import annotations

import unittest

from lkg import (
    AdvancedLKG,
    BetaDistribution,
    DEFAULT_FORGETTING_FACTOR,
    DiscreteMarkovChain,
    LivingKnowledgeGraph,
    Particle,
    ParticleFilter,
)


class TestLivingKG(unittest.TestCase):
    """Unit tests for the Living Knowledge Graph."""

    # ─── BetaDistribution ───────────────────────────────────────────────

    def test_beta_initial_mean(self) -> None:
        b = BetaDistribution()
        self.assertAlmostEqual(b.mean(), 0.5)

    def test_positive_evidence_increases_mean(self) -> None:
        b = BetaDistribution()
        b.update(True, 1.0)
        self.assertGreater(b.mean(), 0.5)

    def test_negative_evidence_decreases_mean(self) -> None:
        b = BetaDistribution()
        b.update(False, 1.0)
        self.assertLess(b.mean(), 0.5)

    def test_variance_matches_formula(self) -> None:
        b = BetaDistribution(2.0, 3.0)
        total: float = 5.0
        expected: float = (2.0 * 3.0) / ((total ** 2) * (total + 1.0))
        self.assertAlmostEqual(b.variance(), expected, places=10)

    def test_decay_moves_toward_prior(self) -> None:
        b = BetaDistribution(5.0, 2.0)
        original_mean: float = b.mean()
        b.decay(0.5, 1.0, 1.0)
        self.assertLess(abs(b.mean() - 0.5), abs(original_mean - 0.5))

    # ─── DiscreteMarkovChain ────────────────────────────────────────────

    def test_markov_rows_normalize(self) -> None:
        mc = DiscreteMarkovChain(["A", "B", "C"])
        mc.observe_transition("A", "B")
        mc.observe_transition("A", "C")
        for s1 in mc.states:
            row_sum: float = sum(
                mc.transition_prob(s1, s2) for s2 in mc.states
            )
            self.assertAlmostEqual(row_sum, 1.0, places=10)

    def test_observed_transitions_change_posterior(self) -> None:
        mc = DiscreteMarkovChain(["A", "B"])
        before: float = mc.transition_prob("A", "B")
        mc.observe_transition("A", "B")
        after: float = mc.transition_prob("A", "B")
        self.assertGreater(after, before)

    def test_sampling_follows_probabilities(self) -> None:
        mc = DiscreteMarkovChain(["A", "B"], seed=42)
        for _ in range(100):
            mc.observe_transition("A", "B")
        counts: dict[str, int] = {"A": 0, "B": 0}
        for _ in range(10000):
            s: str = mc.sample_next("A")
            counts[s] += 1
        self.assertGreater(counts["B"], counts["A"])

    def test_unknown_state_raises(self) -> None:
        mc = DiscreteMarkovChain(["A", "B"])
        with self.assertRaises(ValueError):
            mc.observe_transition("A", "C")

    def test_duplicate_states_raise(self) -> None:
        with self.assertRaises(ValueError):
            DiscreteMarkovChain(["A", "A", "B"])

    def test_get_transition_matrix_rows_sum_to_one(self) -> None:
        mc = DiscreteMarkovChain(["A", "B", "C"])
        mc.observe_transition("A", "B")
        matrix: list[list[float]] = mc.get_transition_matrix()
        for row in matrix:
            self.assertAlmostEqual(sum(row), 1.0, places=10)

    # ─── Particle / ParticleFilter ──────────────────────────────────────

    def test_particle_initialization(self) -> None:
        p = Particle()
        self.assertEqual(len(p.facts), 0)
        self.assertEqual(len(p.entities), 0)
        self.assertEqual(len(p.sources), 0)

    def test_particle_filter_weights_normalize(self) -> None:
        pf = ParticleFilter(10, 0.95, seed=42)
        pf.update(
            {
                "fact": ("s", "p", "o"),
                "source": None,
                "positive": True,
                "weight": None,
            }
        )
        total: float = sum(pf.weights)
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_resample_equal_weights(self) -> None:
        pf = ParticleFilter(10, 0.95, seed=42)
        # Collapse all weight onto one particle
        pf.weights = [0.0] * 9 + [1.0]
        pf.resample()
        for w in pf.weights:
            self.assertAlmostEqual(w, 0.1, places=10)

    def test_ess_triggers_resample(self) -> None:
        pf = ParticleFilter(5, 0.95, seed=42)
        # Collapse weights so ESS is very low
        pf.weights = [0.0, 0.0, 0.0, 0.0, 1.0]
        ess_before: float = pf._ess()
        self.assertLess(ess_before, pf.num_particles / 2.0)
        # Update should trigger resampling
        pf.update(
            {
                "fact": ("s", "p", "o"),
                "source": None,
                "positive": True,
                "weight": None,
            }
        )
        for w in pf.weights:
            self.assertAlmostEqual(w, 0.2, places=10)

    # ─── LivingKnowledgeGraph ───────────────────────────────────────────

    def test_add_fact_increases_confidence(self) -> None:
        kg = LivingKnowledgeGraph(
            num_particles=50, forgetting_factor=0.99, seed=42
        )
        before: float = kg.get_confidence("sky", "is", "blue")["mean"]
        kg.add_fact("sky", "is", "blue")
        after: float = kg.get_confidence("sky", "is", "blue")["mean"]
        self.assertGreater(after, before)

    def test_add_fact_negative_evidence_decreases_confidence(self) -> None:
        kg = LivingKnowledgeGraph(
            num_particles=50, forgetting_factor=0.99, seed=42
        )
        kg.add_fact("sky", "is", "green")
        before: float = kg.get_confidence("sky", "is", "green")["mean"]
        kg.add_fact("sky", "is", "green", positive=False)
        after: float = kg.get_confidence("sky", "is", "green")["mean"]
        self.assertLess(after, before)

    def test_low_reliability_source_moves_less(self) -> None:
        # High-reliability source: many prior positive observations
        kg_high = LivingKnowledgeGraph(
            num_particles=100, forgetting_factor=0.99, seed=42
        )
        for p in kg_high._pf.particles:
            p.sources["trusted"] = BetaDistribution(100.0, 1.0)
        kg_high.add_fact("sky", "is", "blue", source="trusted")
        high_conf: float = kg_high.get_confidence("sky", "is", "blue")["mean"]

        # Low-reliability source: many prior negative observations
        kg_low = LivingKnowledgeGraph(
            num_particles=100, forgetting_factor=0.99, seed=42
        )
        for p in kg_low._pf.particles:
            p.sources["distrusted"] = BetaDistribution(1.0, 100.0)
        kg_low.add_fact("sky", "is", "blue", source="distrusted")
        low_conf: float = kg_low.get_confidence("sky", "is", "blue")["mean"]

        self.assertGreater(high_conf, low_conf)

    def test_manual_weight_combines_with_source_reliability(self) -> None:
        kg_plain = LivingKnowledgeGraph(
            num_particles=30, forgetting_factor=0.99, seed=7
        )
        kg_plain.add_fact("water", "is", "wet", source="sensor", weight=1.0)
        plain_conf: float = kg_plain.get_confidence("water", "is", "wet")["mean"]

        kg_boosted = LivingKnowledgeGraph(
            num_particles=30, forgetting_factor=0.99, seed=7
        )
        kg_boosted.add_fact("water", "is", "wet", source="sensor", weight=5.0)
        boosted_conf: float = kg_boosted.get_confidence("water", "is", "wet")["mean"]

        self.assertGreater(boosted_conf, plain_conf)

    def test_predict_entity_state(self) -> None:
        kg = LivingKnowledgeGraph(
            num_particles=1, forgetting_factor=0.99, seed=42
        )
        kg.define_entity_states("light", ["on", "off"])
        # 3 observations of on→off with prior_strength=1.0:
        #   transition_prob("on","on")  = (0 + 1) / (3 + 2) = 0.2
        #   transition_prob("on","off") = (3 + 1) / (3 + 2) = 0.8
        for particle in kg._pf.particles:
            mc = particle.entities["light"][1]
            for _ in range(3):
                mc.observe_transition("on", "off")
        dist: list[float] = kg.predict_entity_state("light", 1)
        self.assertAlmostEqual(dist[0], 0.2, places=4)
        self.assertAlmostEqual(dist[1], 0.8, places=4)

    def test_get_current_state_distribution_initial(self) -> None:
        kg = LivingKnowledgeGraph(
            num_particles=10, forgetting_factor=0.99, seed=42
        )
        kg.define_entity_states("light", ["on", "off"])
        dist: list[float] = kg.get_current_state_distribution("light")
        self.assertAlmostEqual(sum(dist), 1.0, places=10)
        self.assertAlmostEqual(dist[0], 1.0, places=10)
        self.assertAlmostEqual(dist[1], 0.0, places=10)

    def test_get_current_state_distribution_unknown_entity_raises(self) -> None:
        kg = LivingKnowledgeGraph()
        with self.assertRaises(KeyError):
            kg.get_current_state_distribution("nonexistent")

    def test_unknown_entity_raises(self) -> None:
        kg = LivingKnowledgeGraph()
        with self.assertRaises(KeyError):
            kg.predict_entity_state("nonexistent", 1)

    def test_forget_stale_decays_confidence(self) -> None:
        kg = LivingKnowledgeGraph(
            num_particles=50, forgetting_factor=0.5, seed=42
        )
        kg.add_fact("sky", "is", "blue")
        before: float = kg.get_confidence("sky", "is", "blue")["mean"]
        kg.forget_stale()
        after: float = kg.get_confidence("sky", "is", "blue")["mean"]
        # After decay, confidence should move toward 0.5
        self.assertLess(abs(after - 0.5), abs(before - 0.5))

    # ─── AdvancedLKG ─────────────────────────────────────────────────────

    def test_compute_free_energy_returns_float(self) -> None:
        kg = AdvancedLKG(num_particles=5, forgetting_factor=0.99, seed=42)
        kg.add_fact("sky", "is", "blue")
        free_energy = kg.compute_free_energy()
        self.assertIsInstance(free_energy, float)

    def test_variational_step_moves_toward_prior(self) -> None:
        kg = AdvancedLKG(num_particles=5, forgetting_factor=0.99, seed=42)
        for _ in range(10):
            kg.add_fact("sky", "is", "blue")
        before: float = kg.get_confidence("sky", "is", "blue")["mean"]
        for _ in range(50):
            kg.variational_step(learning_rate=0.05)
        after: float = kg.get_confidence("sky", "is", "blue")["mean"]
        self.assertLess(abs(after - 0.5), abs(before - 0.5))

    def test_compute_laplacian_shape(self) -> None:
        kg = AdvancedLKG(num_particles=3, forgetting_factor=0.99, seed=42)
        kg.define_entity_states("Alice", ["Paris", "London"])
        kg.define_entity_states("Bob", ["NYC", "Boston"])
        kg.add_fact("Alice", "knows", "Bob")
        laplacian: list[list[float]] = kg.compute_laplacian()
        self.assertEqual(len(laplacian), 2)
        self.assertEqual(len(laplacian[0]), 2)

    # ─── End-to-end ────────────────────────────────────────────────────

    def test_end_to_end_stream_tracking(self) -> None:
        kg = LivingKnowledgeGraph(
            num_particles=50, forgetting_factor=0.99, seed=42
        )
        # Repeatedly reinforce the true fact
        for _ in range(20):
            kg.add_fact("water", "is", "wet")
        # Add a false fact only once
        kg.add_fact("water", "is", "dry")

        true_conf: float = kg.get_confidence("water", "is", "wet")["mean"]
        false_conf: float = kg.get_confidence("water", "is", "dry")["mean"]
        self.assertGreater(true_conf, false_conf)
        self.assertGreater(true_conf, 0.5)


# ─── from_config (added for the engine.py integration) ─────────────────────

def test_from_config_reads_knowledge_graph_section() -> None:
    kg = LivingKnowledgeGraph.from_config(
        {"knowledge_graph": {"num_particles": 7, "forgetting_factor": 0.8, "seed": 3}}
    )
    assert kg._pf.num_particles == 7
    assert kg._pf.forgetting_factor == 0.8


def test_from_config_defaults_when_section_missing() -> None:
    kg = LivingKnowledgeGraph.from_config({})
    assert kg._pf.num_particles == 20
    assert kg._pf.forgetting_factor == DEFAULT_FORGETTING_FACTOR


# ─── observe_entity_transition (added for the engine.py integration) ───────
#
# Regression coverage for the gap this method fixes: before it existed, the
# public LivingKnowledgeGraph API had no way to feed an actually-observed
# state into an entity's Markov chain -- DiscreteMarkovChain.observe_transition
# was only ever called from this file's own tests above, never from
# ParticleFilter/LivingKnowledgeGraph. Confirmed empirically (see CHANGELOG.md)
# that repeated step() calls alone leave predict_entity_state locked at a
# uniform distribution forever. test_step_alone_never_learns_a_pattern below
# pins down that this is genuinely a property of step()/predict() and not
# something observe_entity_transition happens to duplicate.

def test_step_alone_never_learns_a_pattern() -> None:
    kg = LivingKnowledgeGraph(num_particles=20, forgetting_factor=0.99, seed=1)
    kg.define_entity_states("session_intent", ["math_symbolic", "coding"])
    for _ in range(25):
        kg.step()
    predicted = kg.predict_entity_state("session_intent", horizon=1)
    assert all(abs(p - 0.5) < 1e-9 for p in predicted)


def test_observe_entity_transition_learns_a_pattern() -> None:
    kg = LivingKnowledgeGraph(num_particles=30, forgetting_factor=0.99, seed=1)
    kg.define_entity_states("session_intent", ["math_symbolic", "coding", "general"])
    for intent in ["math_symbolic"] * 8 + ["coding"] + ["math_symbolic"] * 6:
        kg.observe_entity_transition("session_intent", intent)

    dist = kg.get_current_state_distribution("session_intent")
    predicted = kg.predict_entity_state("session_intent", horizon=1)
    assert dist[0] == 1.0  # last observed state was math_symbolic
    # Predicted next state should favor the dominant observed pattern,
    # strictly more than a uniform 1/3 each.
    assert predicted[0] > 1.0 / 3.0


def test_observe_entity_transition_rejects_unknown_entity() -> None:
    kg = LivingKnowledgeGraph(num_particles=3, seed=0)
    try:
        kg.observe_entity_transition("nope", "x")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_observe_entity_transition_rejects_unknown_state() -> None:
    kg = LivingKnowledgeGraph(num_particles=3, seed=0)
    kg.define_entity_states("session_intent", ["math_symbolic", "coding"])
    try:
        kg.observe_entity_transition("session_intent", "not_a_real_state")
        assert False, "expected ValueError"
    except ValueError:
        pass

