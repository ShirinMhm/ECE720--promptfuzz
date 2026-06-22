"""
budget_scheduler.py
-------------------
Budget-Aware Multi-Fidelity Scheduler for PromptFuzz.

Implements a Successive-Halving / Hyperband-style staged evaluator over
defense subsets, as described in Section 4.1 of the project proposal.

Instead of evaluating every mutant against all |D| defenses (cost = O(|mutants| * |D|)),
we evaluate in stages:

  Stage 1  →  evaluate on D1 ⊂ D  (small diverse subset, ~stage_fractions[0]*|D| defenses)
  Stage 2  →  promote top-k mutants to D2  (~stage_fractions[1]*|D| defenses, cumulative)
  Stage 3  →  full evaluation only for survivors

Promotion uses a Lower Confidence Bound (LCB) criterion from the proposal:

    lcb_i = ASR_hat_i  -  beta * sqrt(log(T) / n_i)

where:
  ASR_hat_i  = empirical ASR of mutant i so far  (successes / n_i)
  n_i        = number of defenses evaluated for mutant i
  T          = current global fuzzing iteration
  beta       = exploration coefficient (higher = more conservative promotion)

A mutant is promoted to the next stage only if its LCB is in the top-k
AND exceeds promotion_threshold.  This ensures we never promote a mutant
whose pessimistic ASR estimate is too low to be useful.
"""

import math
import logging
import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from gptfuzzer.fuzzer.core import PromptNode, GPTFuzzer
    from gptfuzzer.fuzzer.coverage_tracker import DefenseCoverageTracker

# Sentinel string used to mark budget-eliminated nodes (mirrors 'early termination')
BUDGET_ELIMINATED = "budget_eliminated"


class MultiFidelityScheduler:
    """
    Successive-Halving style multi-fidelity scheduler over defense subsets.

    Parameters
    ----------
    fuzzer : GPTFuzzer
        The parent fuzzer instance (set after construction via fuzzer.scheduler = self).
    stage_fractions : list[float]
        Cumulative fraction of defenses used at each stage.
        E.g. [0.2, 0.5, 1.0] means Stage 1 uses 20% of defenses,
        Stage 2 uses 50% (adding 30% more), Stage 3 uses all 100%.
        Must be strictly increasing and end with 1.0.
    beta : float
        Coefficient for the LCB exploration penalty.
        Higher beta = more conservative (fewer promotions).
    top_k_fractions : list[float]
        Fraction of active mutants to promote at each stage transition.
        Length must be len(stage_fractions) - 1.
        E.g. [0.5, 0.5] keeps top-50% at each transition.
    promotion_threshold : float
        Minimum score required for promotion.  Mutants with score below
        this are eliminated even if they are in the top-k.
        Set to 0.0 to always promote top-k regardless of absolute score.
    coverage_tracker : DefenseCoverageTracker or None
        If provided, enables Part B:
          - D1 ordering becomes coverage-aware (novel clusters first)
          - Promotion scoring uses combined  λ·LCB + (1-λ)·Novelty
    seed : int or None
        Random seed for the defense shuffle (used only when coverage_tracker
        is None, i.e. in pure budget-aware mode).
    """

    def __init__(
        self,
        fuzzer: "GPTFuzzer" = None,
        stage_fractions: List[float] = (0.2, 0.5, 1.0),
        beta: float = 1.0,
        top_k_fractions: List[float] = (0.5, 0.5),
        promotion_threshold: float = 0.0,
        coverage_tracker: "Optional[DefenseCoverageTracker]" = None,
        seed: int = 42,
    ):
        assert len(top_k_fractions) == len(stage_fractions) - 1, (
            "top_k_fractions must have length len(stage_fractions) - 1"
        )
        assert stage_fractions[-1] == 1.0, "Final stage fraction must be 1.0"
        assert all(
            stage_fractions[i] < stage_fractions[i + 1]
            for i in range(len(stage_fractions) - 1)
        ), "stage_fractions must be strictly increasing"

        self.fuzzer = fuzzer
        self.stage_fractions = list(stage_fractions)
        self.beta = beta
        self.top_k_fractions = list(top_k_fractions)
        self.promotion_threshold = promotion_threshold
        self.coverage_tracker = coverage_tracker
        self.seed = seed

        # Defense stage index lists – built lazily once fuzzer is set
        self._defense_stages: List[List[int]] = []

        # Per-iteration record of which defense indices were evaluated for each node
        # Used by coverage_tracker.update_coverage() and novelty()
        self._last_eval_def_indices: dict = {}   # node_id → list[int]

        # Tracking / diagnostics
        self.total_queries_saved: int = 0
        self.stage_promotion_counts: List[int] = [0] * len(stage_fractions)
        self.stage_elimination_counts: List[int] = [0] * len(stage_fractions)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def setup(self):
        """
        Build defense stage index lists.  Called after self.fuzzer is set.

        When a coverage_tracker is available, D1 uses coverage-aware ordering
        (novel-cluster defenses first) so Stage 1 maximises diagnostic value.
        Without a tracker, a fixed random shuffle is used.
        """
        assert self.fuzzer is not None, "fuzzer must be set before calling setup()"
        n = len(self.fuzzer.defenses)

        if self.coverage_tracker is not None:
            # Coverage-aware ordering: novel clusters first
            if self.coverage_tracker.defense_labels is None:
                self.coverage_tracker.setup(self.fuzzer.defenses)
            ordered = self.coverage_tracker.coverage_aware_ordering()
        else:
            # Pure random shuffle (Part A only)
            rng = np.random.default_rng(self.seed)
            ordered = rng.permutation(n).tolist()

        self._defense_stages = []
        for frac in self.stage_fractions:
            size = max(1, int(round(frac * n)))
            self._defense_stages.append(ordered[:size])

        stage_sizes = [len(s) for s in self._defense_stages]
        mode = "coverage-aware" if self.coverage_tracker else "random"
        logging.info(
            f"[MultiFidelity] Defense stages built ({mode}): "
            f"{stage_sizes} out of {n} total defenses"
        )

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, prompt_nodes: "list[PromptNode]"):
        """
        Staged multi-fidelity evaluation.  Replaces GPTFuzzer.evaluate().

        Each prompt_node ends up with:
          - node.response : list of responses (one per evaluated defense)
          - node.results  : list of 0/1 predictions (one per evaluated defense)
          - node.prompt   : set to BUDGET_ELIMINATED if the node was cut early

        The length of node.results equals the number of defenses *actually*
        evaluated for that node (not the full |D|), so current_query correctly
        reflects real API calls made.

        Also records which defense indices were evaluated for each node in
        self._last_eval_def_indices, used by the coverage tracker.
        """
        if not self._defense_stages:
            self.setup()

        T = max(self.fuzzer.current_iteration + 1, 2)  # avoid log(0)

        # Initialise
        for pn in prompt_nodes:
            pn.response = []
            pn.results = []

        messages = [pn.prompt for pn in prompt_nodes]
        active_set = list(range(len(prompt_nodes)))
        evaluated_def_indices: set = set()
        # Track ordered list of def indices per node (for coverage tracker)
        node_def_indices: dict = {i: [] for i in range(len(prompt_nodes))}

        for stage_idx, cumulative_def_indices in enumerate(self._defense_stages):
            if not active_set:
                break

            # Defenses new at this stage
            new_def_indices = [
                i for i in cumulative_def_indices if i not in evaluated_def_indices
            ]
            if not new_def_indices:
                continue

            # ----- Evaluate new defenses for active mutants -----
            for def_i in new_def_indices:
                defense = self.fuzzer.defenses[def_i]
                active_msgs = [messages[i] for i in active_set]
                active_nodes = [prompt_nodes[i] for i in active_set]

                responses = self.fuzzer.target.generate_batch(active_msgs, target=defense)
                predictions = self.fuzzer.predictor.predict(
                    responses, defense["access_code"]
                )

                for node_pos, (pn, resp, pred) in enumerate(
                    zip(active_nodes, responses, predictions)
                ):
                    pn.response.append(resp)
                    pn.results.append(pred)
                    node_def_indices[active_set[node_pos]].append(def_i)

            evaluated_def_indices.update(new_def_indices)
            n_evaluated = len(evaluated_def_indices)
            self.stage_promotion_counts[stage_idx] += len(active_set)

            # ----- Promotion decision (skip for the last stage) -----
            if stage_idx < len(self._defense_stages) - 1:
                top_k_frac = self.top_k_fractions[stage_idx]
                k = max(1, int(round(len(active_set) * top_k_frac)))

                # Score: combined λ·LCB + (1-λ)·Novelty if tracker available,
                # else pure LCB
                scored = []
                for i in active_set:
                    pn = prompt_nodes[i]
                    if self.coverage_tracker is not None:
                        score = self.coverage_tracker.combined_score(
                            pn,
                            def_indices_evaluated=node_def_indices[i],
                            n_evaluated=n_evaluated,
                            T=T,
                            beta=self.beta,
                        )
                    else:
                        score = self._lcb(pn, n_evaluated, T)
                    scored.append((i, score))

                scored.sort(key=lambda x: x[1], reverse=True)

                promoted = [
                    idx for idx, sc in scored[:k]
                    if sc >= self.promotion_threshold
                ]
                eliminated = [
                    idx for idx, _ in scored if idx not in set(promoted)
                ]

                n_remaining_defs = len(self.fuzzer.defenses) - n_evaluated
                self.total_queries_saved += len(eliminated) * n_remaining_defs
                self.stage_elimination_counts[stage_idx] += len(eliminated)

                mode = "coverage+LCB" if self.coverage_tracker else "LCB"
                logging.info(
                    f"[MultiFidelity] Stage {stage_idx + 1}/{len(self._defense_stages)}: "
                    f"{len(active_set)} mutants | {n_evaluated} defenses | "
                    f"scoring={mode} → {len(promoted)} promoted, "
                    f"{len(eliminated)} eliminated "
                    f"(~{len(eliminated) * n_remaining_defs} queries saved)"
                )

                for idx in eliminated:
                    prompt_nodes[idx].prompt = BUDGET_ELIMINATED

                active_set = promoted

        # Store per-node defense index lists so update() can call coverage_tracker
        self._last_eval_def_indices = {
            id(prompt_nodes[i]): node_def_indices[i]
            for i in range(len(prompt_nodes))
        }

        for stage_idx, cumulative_def_indices in enumerate(self._defense_stages):
            if not active_set:
                break

            # Defenses that are new at this stage (not yet evaluated)
            new_def_indices = [
                i for i in cumulative_def_indices if i not in evaluated_def_indices
            ]
            if not new_def_indices:
                continue

            # ----- Evaluate new defenses for active mutants -----
    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of budget savings and coverage."""
        total_possible = (
            len(self.fuzzer.defenses)
            * sum(self.stage_promotion_counts[0:1])
        )
        pct_saved = (
            100.0 * self.total_queries_saved / total_possible
            if total_possible > 0
            else 0.0
        )
        base = (
            f"[MultiFidelity] "
            f"Queries saved: {self.total_queries_saved} / {total_possible} "
            f"({pct_saved:.1f}%) | "
            f"Stage eliminations: {self.stage_elimination_counts}"
        )
        if self.coverage_tracker is not None:
            base += " | " + self.coverage_tracker.summary()
        return base

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lcb(self, node: "PromptNode", n_evaluated: int, T: int) -> float:
        """
        Lower Confidence Bound for the mutant's true ASR.

            lcb = ASR_hat  -  beta * sqrt(log(T) / n_i)

        The subtracted term is the exploration penalty: a mutant that has
        only been tested on a few defenses has a wider confidence interval,
        so we penalise it more.  This is deliberately conservative — we only
        promote mutants that look good even in the pessimistic case.
        """
        n_results = len(node.results)
        if n_results == 0:
            return 0.0
        asr_hat = sum(node.results) / n_results
        penalty = self.beta * math.sqrt(math.log(T) / n_results)
        return max(0.0, asr_hat - penalty)
