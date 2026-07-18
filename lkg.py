#!/usr/bin/env python3
"""
Living Knowledge Graph — A dynamic, probabilistic knowledge graph.

Stores facts as triples (subject, predicate, object) with Beta-distributed
confidence, tracks entity state evolution with discrete-time Markov chains
and Dirichlet posterior counts, supports online inference with a particle
filter, models per-source reliability with Beta distributions, and applies
forgetting through exponential decay back to a neutral prior. The optional
AdvancedLKG subclass adds free-energy and graph-Laplacian analysis.

Exported classes:
    BetaDistribution
    DiscreteMarkovChain
    Particle
    ParticleFilter
    LivingKnowledgeGraph
    AdvancedLKG

Example usage::

    kg = LivingKnowledgeGraph(num_particles=20, forgetting_factor=0.95, seed=42)
    kg.define_entity_states("weather", ["sunny", "cloudy", "rainy"])
    kg.add_fact("sky", "is", "blue", source="sensor_a")
    kg.step()
    conf = kg.get_confidence("sky", "is", "blue")
    dist = kg.predict_entity_state("weather", horizon=2)

Integration
-----------
``OpenClosedLoopEngine`` (``engine.py``) owns one instance of this graph and
uses it as a structured, cross-turn memory layer alongside ``VectorMemoryStore``:
a ``"session_intent"`` entity tracks the sequence of task intents a session
asks about (see ``decomposer.KNOWN_INTENTS``), and each retrieved document's
match to the current intent is a fact reinforced or penalized by
``close_loop()`` once real feedback is known, attributed to the document's
origin (bootstrap data, a prior successful interaction, or teacher feedback
-- see ``engine._document_source``), so per-source reliability emerges from
real outcomes rather than being fixed in config or inflated by retrieval
alone. See ``docs/UNIFIED_PIPELINE.md`` for the full data flow and
CHANGELOG.md for why ``from_config`` and ``observe_entity_transition`` were
added on top of the module below (neither existed in the original
standalone version).
"""

# ─── Design Decisions ───────────────────────────────────────────────────────
#
# Graph state:           Flat dict keyed by (subject, predicate, object) ->
#                        BetaDistribution. O(1) lookup with minimal overhead,
#                        versus a nested entity->predicate mapping or a full
#                        adjacency-list graph object.
# Confidence updates:    Exact Beta-Bernoulli conjugate updates. Closed-form
#                        and O(1) per update; variational or MCMC methods add
#                        cost and convergence risk with no benefit here.
# Temporal evolution:    Discrete-time Markov chain with Dirichlet posterior
#                        counts. Matches the step()-based API and is simpler
#                        to estimate than a continuous-time or semi-Markov
#                        alternative.
# Whole-graph inference: Particle filter (sequential Monte Carlo). Handles the
#                        mix of discrete Markov states and continuous Beta
#                        beliefs that a Kalman filter or loopy belief
#                        propagation cannot represent cleanly; resampling
#                        combats particle degeneracy.
# Forgetting:            Exponential decay toward a neutral Beta(1, 1) prior.
#                        O(1) per step, smooth, and tunable via one factor;
#                        no per-fact observation history required.
# Source reliability:    Independent Beta distribution per source, used both
#                        to weight incoming evidence and inside a noisy-report
#                        likelihood model. Lets the system learn which sources
#                        are trustworthy without hierarchical-model overhead.
#
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
import random
import warnings
from typing import Any

# ─── Constants ───────────────────────────────────────────────────────────────

MIN_BETA: float = 1e-6
DEFAULT_PRIOR_ALPHA: float = 1.0
DEFAULT_PRIOR_BETA: float = 1.0
DEFAULT_FORGETTING_FACTOR: float = 0.95


# ─── BetaDistribution ───────────────────────────────────────────────────────

class BetaDistribution:
    """Beta(alpha, beta) distribution for binary fact confidence."""

    def __init__(
        self,
        alpha: float = DEFAULT_PRIOR_ALPHA,
        beta: float = DEFAULT_PRIOR_BETA,
    ) -> None:
        if alpha < 0 or beta < 0:
            raise RuntimeError(
                f"Beta parameters must be non-negative, got alpha={alpha}, beta={beta}"
            )
        self.alpha: float = alpha
        self.beta: float = beta

    def update(self, positive: bool, weight: float) -> None:
        """Update the distribution with a weighted observation."""
        w: float = max(weight, MIN_BETA)
        new_alpha: float = self.alpha + (w if positive else 0.0)
        new_beta: float = self.beta + (0.0 if positive else w)
        if new_alpha < 0 or new_beta < 0:
            raise RuntimeError(
                f"Beta parameters became negative: alpha={new_alpha}, beta={new_beta}"
            )
        self.alpha = new_alpha
        self.beta = new_beta

    def mean(self) -> float:
        """Posterior mean of the Beta distribution."""
        total: float = self.alpha + self.beta
        if total < MIN_BETA:
            return 0.5
        return self.alpha / total

    def variance(self) -> float:
        """Posterior variance of the Beta distribution."""
        total: float = self.alpha + self.beta
        if total < MIN_BETA:
            # Degenerate case: no evidence. Return the variance of
            # Uniform(0, 1) to be consistent with mean() returning 0.5.
            return 1.0 / 12.0
        return (self.alpha * self.beta) / ((total ** 2) * (total + 1.0))

    def decay(
        self,
        factor: float,
        prior_alpha: float = DEFAULT_PRIOR_ALPHA,
        prior_beta: float = DEFAULT_PRIOR_BETA,
    ) -> None:
        """Exponential decay toward a neutral prior."""
        new_alpha: float = factor * self.alpha + (1.0 - factor) * prior_alpha
        new_beta: float = factor * self.beta + (1.0 - factor) * prior_beta
        if new_alpha < 0 or new_beta < 0:
            raise RuntimeError(
                f"Beta parameters became negative after decay: "
                f"alpha={new_alpha}, beta={new_beta}"
            )
        self.alpha = new_alpha
        self.beta = new_beta

    def copy(self) -> BetaDistribution:
        """Return an independent copy."""
        return BetaDistribution(self.alpha, self.beta)

    def __repr__(self) -> str:
        return (
            f"BetaDistribution(alpha={self.alpha:.4f}, beta={self.beta:.4f}, "
            f"mean={self.mean():.4f})"
        )


# ─── DiscreteMarkovChain ────────────────────────────────────────────────────

class DiscreteMarkovChain:
    """Finite-state discrete-time Markov chain with Dirichlet posterior counts."""

    def __init__(
        self, states: list[str], prior_strength: float = 1.0, seed: int | None = None
    ) -> None:
        if len(states) == 0:
            raise ValueError("State list must not be empty")
        if len(set(states)) != len(states):
            raise ValueError(f"Duplicate states detected: {states}")
        self.states: list[str] = list(states)
        self._state_set: set[str] = set(states)
        self.prior_strength: float = prior_strength
        self.counts: dict[str, dict[str, float]] = {
            s1: {s2: 0.0 for s2 in self.states} for s1 in self.states
        }
        self._rng: random.Random = random.Random(seed)

    def _validate_state(self, state: str) -> None:
        if state not in self._state_set:
            raise ValueError(
                f"Unknown state: '{state}'. Valid states: {self.states}"
            )

    def observe_transition(self, from_state: str, to_state: str) -> None:
        """Record one observed transition."""
        self._validate_state(from_state)
        self._validate_state(to_state)
        self.counts[from_state][to_state] += 1.0

    def transition_prob(self, from_state: str, to_state: str) -> float:
        """Posterior mean transition probability (Dirichlet-Multinomial)."""
        self._validate_state(from_state)
        self._validate_state(to_state)
        row: dict[str, float] = self.counts[from_state]
        total: float = sum(row.values()) + self.prior_strength * len(self.states)
        return (row[to_state] + self.prior_strength) / total

    def sample_next(self, from_state: str) -> str:
        """Sample the next state from the posterior predictive."""
        self._validate_state(from_state)
        probs: list[float] = [
            self.transition_prob(from_state, s) for s in self.states
        ]
        r: float = self._rng.random()
        cumulative: float = 0.0
        for s, p in zip(self.states, probs):
            cumulative += p
            if r < cumulative:
                return s
        return self.states[-1]

    def row_total(self, from_state: str) -> float:
        """Total observation count for a row (excluding prior)."""
        self._validate_state(from_state)
        return sum(self.counts[from_state].values())

    def get_transition_matrix(self) -> list[list[float]]:
        """Return the full posterior mean transition matrix (n x n)."""
        return [
            [self.transition_prob(s1, s2) for s2 in self.states]
            for s1 in self.states
        ]

    def copy(self) -> DiscreteMarkovChain:
        """Return an independent copy."""
        mc = DiscreteMarkovChain(self.states, self.prior_strength)
        mc.counts = {
            s1: {s2: v for s2, v in row.items()}
            for s1, row in self.counts.items()
        }
        mc._rng = self._rng
        return mc

    def __repr__(self) -> str:
        return (
            f"DiscreteMarkovChain(states={self.states}, "
            f"prior_strength={self.prior_strength})"
        )


# ─── Particle ───────────────────────────────────────────────────────────────

class Particle:
    """Lightweight snapshot of graph belief state."""

    def __init__(self) -> None:
        self.facts: dict[tuple[str, str, str], BetaDistribution] = {}
        self.entities: dict[str, tuple[str, DiscreteMarkovChain]] = {}
        self.sources: dict[str, BetaDistribution] = {}

    def copy(self) -> Particle:
        """Return an independent deep copy."""
        p = Particle()
        p.facts = {k: v.copy() for k, v in self.facts.items()}
        p.entities = {k: (v[0], v[1].copy()) for k, v in self.entities.items()}
        p.sources = {k: v.copy() for k, v in self.sources.items()}
        return p


# ─── ParticleFilter ─────────────────────────────────────────────────────────

class ParticleFilter:
    """Particle filter for online inference over the knowledge graph."""

    def __init__(
        self,
        num_particles: int,
        forgetting_factor: float,
        seed: int | None = 0,
    ) -> None:
        if num_particles < 1:
            raise ValueError(
                f"num_particles must be >= 1, got {num_particles}"
            )
        self.num_particles: int = num_particles
        if forgetting_factor < 0.0 or forgetting_factor > 1.0:
            raise ValueError(
                f"forgetting_factor must be in [0, 1], got {forgetting_factor}"
            )
        self.forgetting_factor: float = forgetting_factor
        self.rng: random.Random = random.Random(seed)
        self.particles: list[Particle] = [
            Particle() for _ in range(num_particles)
        ]
        self.weights: list[float] = [1.0 / num_particles] * num_particles
        # Shared entity state definitions (set via define_entity_states)
        self.entity_states: dict[str, list[str]] = {}

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_fact(
        particle: Particle, key: tuple[str, str, str]
    ) -> BetaDistribution:
        if key not in particle.facts:
            particle.facts[key] = BetaDistribution()
        return particle.facts[key]

    @staticmethod
    def _ensure_source(
        particle: Particle, source_id: str
    ) -> BetaDistribution:
        if source_id not in particle.sources:
            particle.sources[source_id] = BetaDistribution()
        return particle.sources[source_id]

    # ── predict ──────────────────────────────────────────────────────────

    def predict(self) -> None:
        """Propagate each particle forward one time step."""
        for particle in self.particles:
            # Sample next entity states from each particle's Markov chains
            for entity_id in list(particle.entities.keys()):
                current_state, mc = particle.entities[entity_id]
                if mc.row_total(current_state) < MIN_BETA:
                    # No useful transition data: strong self-loop bias
                    new_state = current_state
                else:
                    new_state = mc.sample_next(current_state)
                particle.entities[entity_id] = (new_state, mc)

            # Apply Beta decay to all facts
            for beta_dist in particle.facts.values():
                beta_dist.decay(
                    self.forgetting_factor,
                    DEFAULT_PRIOR_ALPHA,
                    DEFAULT_PRIOR_BETA,
                )

    # ── update ───────────────────────────────────────────────────────────

    def update(self, evidence: dict[str, Any]) -> None:
        """Incorporate new evidence and reweight / resample particles."""
        s, p, o = evidence["fact"]
        source: str | None = evidence.get("source")
        positive: bool | None = evidence.get("positive")
        supplied_weight: float | None = evidence.get("weight")
        base_weight: float = (
            supplied_weight if supplied_weight is not None else 1.0
        )

        for i, particle in enumerate(self.particles):
            # ── source reliability and effective update weight ────────
            if source is not None:
                src_beta = self._ensure_source(particle, source)
                reliability: float = src_beta.mean()
            else:
                reliability = 1.0
            effective_weight: float = max(reliability * base_weight, MIN_BETA)

            # ── compute likelihood BEFORE updating (noisy-source model) ─
            fact_beta = self._ensure_fact(particle, (s, p, o))
            belief: float = fact_beta.mean()
            if positive is True:
                likelihood: float = (
                    reliability * belief + (1.0 - reliability) * (1.0 - belief)
                )
            elif positive is False:
                likelihood = (
                    reliability * (1.0 - belief) + (1.0 - reliability) * belief
                )
            else:
                likelihood = 1.0  # neutral — no weight adjustment
            likelihood = max(likelihood, MIN_BETA)

            # ── update fact confidence ───────────────────────────────
            if positive is not None:
                fact_beta.update(positive, effective_weight)

            # ── update source reliability ────────────────────────────
            if source is not None and positive is not None:
                src_beta = self._ensure_source(particle, source)
                src_beta.update(positive, 1.0)

            # ── multiply particle weight by likelihood ───────────────
            self.weights[i] *= likelihood

        # ── renormalize weights ──────────────────────────────────────
        total: float = sum(self.weights)
        if total < MIN_BETA:
            warnings.warn(
                "All particle weights collapsed to zero; reinitializing uniformly.",
                RuntimeWarning,
            )
            self.weights = [1.0 / self.num_particles] * self.num_particles
        else:
            self.weights = [w / total for w in self.weights]

        # ── resample if effective sample size is too low ─────────────
        if self._ess() < self.num_particles / 2.0:
            self.resample()

    # ── resample ─────────────────────────────────────────────────────────

    def _ess(self) -> float:
        """Effective sample size: 1 / Σ(w_i²)."""
        total_sq: float = sum(w * w for w in self.weights)
        if total_sq < MIN_BETA:
            return 0.0
        return 1.0 / total_sq

    def resample(self) -> None:
        """Systematic resampling."""
        N: int = self.num_particles
        positions: list[float] = [
            (self.rng.random() + i) / N for i in range(N)
        ]
        cumsum: list[float] = []
        s: float = 0.0
        for w in self.weights:
            s += w
            cumsum.append(s)

        new_particles: list[Particle] = []
        idx: int = 0
        for pos in positions:
            while idx < N - 1 and pos > cumsum[idx]:
                idx += 1
            new_particles.append(self.particles[idx].copy())

        self.particles = new_particles
        self.weights = [1.0 / N] * N

    # ── posterior queries ────────────────────────────────────────────────

    def posterior_confidence(
        self, s: str, p: str, o: str
    ) -> tuple[float, float]:
        """Weighted (mean, variance) across particles for a given fact."""
        key: tuple[str, str, str] = (s, p, o)
        means: list[float] = []
        variances: list[float] = []
        for particle in self.particles:
            if key in particle.facts:
                means.append(particle.facts[key].mean())
                variances.append(particle.facts[key].variance())
            else:
                means.append(0.5)
                variances.append(BetaDistribution().variance())
        weighted_mean: float = sum(
            w * m for w, m in zip(self.weights, means)
        )
        weighted_var: float = sum(
            w * v for w, v in zip(self.weights, variances)
        )
        return (weighted_mean, weighted_var)

    # ── standalone decay (for forget_stale) ──────────────────────────────

    def apply_decay(self) -> None:
        """Apply forgetting decay to all facts and sources across all particles."""
        for particle in self.particles:
            for beta_dist in particle.facts.values():
                beta_dist.decay(
                    self.forgetting_factor,
                    DEFAULT_PRIOR_ALPHA,
                    DEFAULT_PRIOR_BETA,
                )
            for beta_dist in particle.sources.values():
                beta_dist.decay(
                    self.forgetting_factor,
                    DEFAULT_PRIOR_ALPHA,
                    DEFAULT_PRIOR_BETA,
                )


# ─── LivingKnowledgeGraph ───────────────────────────────────────────────────

class LivingKnowledgeGraph:
    """Public API over the particle filter for a living knowledge graph."""

    def __init__(
        self,
        num_particles: int = 20,
        forgetting_factor: float = DEFAULT_FORGETTING_FACTOR,
        seed: int | None = 0,
    ) -> None:
        self._pf: ParticleFilter = ParticleFilter(
            num_particles, forgetting_factor, seed
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LivingKnowledgeGraph":
        """Build from the ``knowledge_graph:`` section of config.yaml, the
        same ``from_config(config)`` idiom every other engine.py component
        (``TaskDecomposer``, ``SharedMoEBackbone``, ``TextEmbedder``, ...)
        uses. Added for the OpenClosedLoopEngine integration -- the original
        standalone module only had the constructor above."""
        kg_cfg = config.get("knowledge_graph", {})
        seed = kg_cfg.get("seed", 0)
        return cls(
            num_particles=int(kg_cfg.get("num_particles", 20)),
            forgetting_factor=float(
                kg_cfg.get("forgetting_factor", DEFAULT_FORGETTING_FACTOR)
            ),
            seed=None if seed is None else int(seed),
        )

    def define_entity_states(
        self, entity_id: str, states: list[str]
    ) -> None:
        """Define a closed, MECE state set for an entity."""
        if len(states) == 0:
            raise ValueError("State list must not be empty")
        if len(set(states)) != len(states):
            raise ValueError(
                f"Duplicate states detected for entity '{entity_id}': {states}"
            )
        self._pf.entity_states[entity_id] = list(states)
        initial_state: str = states[0]
        for particle in self._pf.particles:
            mc = DiscreteMarkovChain(states)
            mc._rng = self._pf.rng
            particle.entities[entity_id] = (initial_state, mc)

    def add_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        source: str | None = None,
        weight: float | None = None,
        positive: bool = True,
    ) -> None:
        """Add evidence for (or against, via positive=False) a fact, updating
        confidence and source reliability."""
        evidence: dict[str, Any] = {
            "fact": (subject, predicate, object),
            "source": source,
            "positive": positive,
            "weight": weight,
        }
        self._pf.update(evidence)

    def get_confidence(
        self, subject: str, predicate: str, object: str
    ) -> dict[str, float]:
        """Return mean and variance for a fact's confidence."""
        mean, variance = self._pf.posterior_confidence(subject, predicate, object)
        return {"mean": mean, "variance": variance}

    def predict_entity_state(
        self, entity_id: str, horizon: int
    ) -> list[float]:
        """Return the averaged state distribution after *horizon* steps."""
        if entity_id not in self._pf.entity_states:
            raise KeyError(f"Unknown entity: '{entity_id}'")
        if horizon < 0:
            raise ValueError(f"Horizon must be non-negative, got {horizon}")

        states: list[str] = self._pf.entity_states[entity_id]
        n: int = len(states)
        state_idx: dict[str, int] = {s: i for i, s in enumerate(states)}
        avg_dist: list[float] = [0.0] * n

        for i, particle in enumerate(self._pf.particles):
            if entity_id not in particle.entities:
                continue
            current_state, mc = particle.entities[entity_id]

            # Start with one-hot at the current state
            dist: list[float] = [0.0] * n
            dist[state_idx[current_state]] = 1.0

            # Build transition matrix once per particle
            tm: list[list[float]] = mc.get_transition_matrix()

            # Multiply by transition matrix `horizon` times
            for _ in range(horizon):
                new_dist: list[float] = [0.0] * n
                for j in range(n):
                    for k in range(n):
                        new_dist[j] += dist[k] * tm[k][j]
                dist = new_dist

            w: float = self._pf.weights[i]
            for j in range(n):
                avg_dist[j] += w * dist[j]

        # Normalise: if no particle had this entity, fall back to uniform
        total_weight: float = sum(avg_dist)
        if total_weight < MIN_BETA:
            return [1.0 / n] * n
        return avg_dist

    def get_current_state_distribution(self, entity_id: str) -> list[float]:
        """Return the current (horizon-0) weighted distribution over entity states."""
        if entity_id not in self._pf.entity_states:
            raise KeyError(f"Unknown entity: '{entity_id}'")
        states: list[str] = self._pf.entity_states[entity_id]
        n: int = len(states)
        state_idx: dict[str, int] = {s: i for i, s in enumerate(states)}
        dist: list[float] = [0.0] * n
        total_weight: float = 0.0

        for i, particle in enumerate(self._pf.particles):
            if entity_id not in particle.entities:
                continue
            current_state, _ = particle.entities[entity_id]
            w: float = self._pf.weights[i]
            dist[state_idx[current_state]] += w
            total_weight += w

        if total_weight < MIN_BETA:
            return [1.0 / n] * n
        return [d / total_weight for d in dist]

    def observe_entity_transition(self, entity_id: str, to_state: str) -> None:
        """Record that ``entity_id`` was actually observed to be in
        ``to_state`` this step, updating every particle's Markov chain
        transition counts (Dirichlet posterior) from wherever that particle
        currently believes the entity was, and snapping its stored state to
        the observed value.

        Without a call like this, ``define_entity_states`` +
        ``predict_entity_state``/``get_current_state_distribution`` have no
        way to learn anything: ``DiscreteMarkovChain.observe_transition``
        exists on the low-level chain, but nothing in ``ParticleFilter`` or
        ``LivingKnowledgeGraph`` ever called it, so ``predict()``'s
        self-loop-biased sampling was the only thing moving entity state,
        and predictions stayed at a uniform prior indefinitely (verified:
        20+ calls to ``step()`` alone never changes
        ``predict_entity_state``'s output). This is the observation-side
        counterpart to ``add_fact`` -- ``add_fact`` feeds evidence about a
        fact's truth into the Beta distributions; this feeds a directly
        observed state into the entity's transition model. See CHANGELOG.md.
        """
        if entity_id not in self._pf.entity_states:
            raise KeyError(f"Unknown entity: '{entity_id}'")
        if to_state not in self._pf.entity_states[entity_id]:
            raise ValueError(
                f"Unknown state '{to_state}' for entity '{entity_id}'. "
                f"Valid states: {self._pf.entity_states[entity_id]}"
            )
        for particle in self._pf.particles:
            current_state, mc = particle.entities[entity_id]
            mc.observe_transition(current_state, to_state)
            particle.entities[entity_id] = (to_state, mc)

    def step(self) -> None:
        """Advance the model by one time step (predict only)."""
        self._pf.predict()

    def forget_stale(self) -> None:
        """Apply decay to all facts and sources without new evidence."""
        self._pf.apply_decay()

    def set_forgetting_factor(self, factor: float) -> None:
        """Update the forgetting factor."""
        if factor < 0.0 or factor > 1.0:
            raise ValueError(
                f"Forgetting factor must be in [0, 1], got {factor}"
            )
        self._pf.forgetting_factor = factor


# ─── AdvancedLKG ─────────────────────────────────────────────────────────────

class AdvancedLKG(LivingKnowledgeGraph):
    """LivingKnowledgeGraph extended with free-energy and graph-Laplacian analysis.

    Experimental and intentionally lightweight relative to the core inference
    engine; useful for inspecting how concentrated beliefs are and how entities
    relate to one another through shared facts.
    """

    def compute_free_energy(self) -> float:
        """Approximate variational free energy of the current belief state.

        Combines a quadratic approximation of each fact's KL divergence from
        the neutral Beta(1, 1) prior with the KL divergence of each entity's
        learned transition matrix from a uniform one. Lower values indicate
        beliefs closer to the neutral prior; higher values indicate more
        concentrated, evidence-driven beliefs.
        """
        free_energy: float = 0.0
        for w, particle in zip(self._pf.weights, self._pf.particles):
            for beta_dist in particle.facts.values():
                free_energy += w * (
                    (beta_dist.alpha - 1.0) ** 2 + (beta_dist.beta - 1.0) ** 2
                )
            for _, mc in particle.entities.values():
                n = len(mc.states)
                for row in mc.get_transition_matrix():
                    for prob in row:
                        if prob > 0.0:
                            free_energy += w * prob * math.log(prob * n)
        return free_energy

    def variational_step(self, learning_rate: float = 0.01) -> None:
        """Nudge every fact's Beta parameters toward the prior via one step
        of gradient descent on the quadratic free-energy approximation."""
        for particle in self._pf.particles:
            for beta_dist in particle.facts.values():
                grad_alpha = 2.0 * (beta_dist.alpha - 1.0)
                grad_beta = 2.0 * (beta_dist.beta - 1.0)
                beta_dist.alpha = max(
                    beta_dist.alpha - learning_rate * grad_alpha, MIN_BETA
                )
                beta_dist.beta = max(
                    beta_dist.beta - learning_rate * grad_beta, MIN_BETA
                )

    def compute_laplacian(self) -> list[list[float]]:
        """Build a graph Laplacian over defined entities, weighted by the mean
        confidence of facts linking them as subject/object."""
        entities: list[str] = list(self._pf.entity_states.keys())
        n: int = len(entities)
        index: dict[str, int] = {e: i for i, e in enumerate(entities)}
        adjacency: list[list[float]] = [[0.0] * n for _ in range(n)]

        for particle in self._pf.particles:
            for (subject, _, obj), beta_dist in particle.facts.items():
                if subject in index and obj in index:
                    i, j = index[subject], index[obj]
                    w = beta_dist.mean()
                    adjacency[i][j] += w
                    adjacency[j][i] += w

        degree: list[float] = [sum(row) for row in adjacency]
        laplacian: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                laplacian[i][j] = degree[i] if i == j else -adjacency[i][j]
        return laplacian

