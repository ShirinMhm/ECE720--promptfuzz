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
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from gptfuzzer.fuzzer.core import PromptNode, GPTFuzzer

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
        Minimum LCB score required for promotion.  Mutants with LCB below
        this are eliminated even if they are in the top-k.
        Set to 0.0 to always promote top-k regardless of absolute LCB.
    seed : int or None
        Random seed for the defense shuffle (for reproducibility).
    """

    def __init__(
        self,
        fuzzer: "GPTFuzzer" = None,
        stage_fractions: List[float] = (0.2, 0.5, 1.0),
        beta: float = 1.0,
        top_k_fractions: List[float] = (0.5, 0.5),
        promotion_threshold: float = 0.0,
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
        self.seed = seed

        # Defense stage index lists – built lazily once fuzzer is set
        self._defense_stages: List[List[int]] = []

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
        Defense order is shuffled once with a fixed seed so that D1 is a
        reproducible random sample of the full defense set.
        (Coverage-guided defense selection, i.e. embedding-based clustering,
        can replace this shuffle in the Part B extension.)
        """
        assert self.fuzzer is not None, "fuzzer must be set before calling setup()"
        n = len(self.fuzzer.defenses)
        rng = np.random.default_rng(self.seed)
        shuffled = rng.permutation(n).tolist()

        self._defense_stages = []
        for frac in self.stage_fractions:
            size = max(1, int(round(frac * n)))
            # Each stage stores the *cumulative* defense indices up to that point
            self._defense_stages.append(shuffled[:size])

        stage_sizes = [len(s) for s in self._defense_stages]
        logging.info(
            f"[MultiFidelity] Defense stages built: {stage_sizes} out of {n} total defenses"
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
        """
        if not self._defense_stages:
            self.setup()

        T = max(self.fuzzer.current_iteration + 1, 2)  # avoid log(0)

        # Initialise
        for pn in prompt_nodes:
            pn.response = []
            pn.results = []

        messages = [pn.prompt for pn in prompt_nodes]
        # active_set: indices into prompt_nodes that are still in the running
        active_set = list(range(len(prompt_nodes)))
        evaluated_def_indices: set = set()

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
            for def_i in new_def_indices:
                defense = self.fuzzer.defenses[def_i]
                active_msgs = [messages[i] for i in active_set]
                active_nodes = [prompt_nodes[i] for i in active_set]

                responses = self.fuzzer.target.generate_batch(active_msgs, target=defense)
                predictions = self.fuzzer.predictor.predict(
                    responses, defense["access_code"]
                )

                for pn, resp, pred in zip(active_nodes, responses, predictions):
                    pn.response.append(resp)
                    pn.results.append(pred)

            evaluated_def_indices.update(new_def_indices)
            n_evaluated = len(evaluated_def_indices)
            self.stage_promotion_counts[stage_idx] += len(active_set)

            # ----- Promotion decision (skip for the last stage) -----
            if stage_idx < len(self._defense_stages) - 1:
                top_k_frac = self.top_k_fractions[stage_idx]
                k = max(1, int(round(len(active_set) * top_k_frac)))

                # Score each active mutant by LCB
                scored = [
                    (i, self._lcb(prompt_nodes[i], n_evaluated, T))
                    for i in active_set
                ]
                scored.sort(key=lambda x: x[1], reverse=True)

                # Promote top-k that exceed the threshold
                promoted = [
                    idx
                    for idx, lcb in scored[:k]
                    if lcb >= self.promotion_threshold
                ]
                eliminated = [
                    idx for idx, _ in scored if idx not in set(promoted)
                ]

                n_remaining_defs = len(self.fuzzer.defenses) - n_evaluated
                self.total_queries_saved += len(eliminated) * n_remaining_defs
                self.stage_elimination_counts[stage_idx] += len(eliminated)

                logging.info(
                    f"[MultiFidelity] Stage {stage_idx + 1}/{len(self._defense_stages)}: "
                    f"{len(active_set)} mutants evaluated on {n_evaluated} defenses → "
                    f"{len(promoted)} promoted, {len(eliminated)} eliminated "
                    f"(~{len(eliminated) * n_remaining_defs} queries saved)"
                )

                # Mark eliminated nodes
                for idx in eliminated:
                    prompt_nodes[idx].prompt = BUDGET_ELIMINATED

                active_set = promoted

        # Any mutant still active after all stages has been fully evaluated.
        # Any mutant marked BUDGET_ELIMINATED has partial results only —
        # update() will handle them by checking the prompt string.

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of budget savings."""
        total_possible = (
            len(self.fuzzer.defenses)
            * sum(self.stage_promotion_counts[0:1])  # all mutants that entered stage 1
        )
        pct_saved = (
            100.0 * self.total_queries_saved / total_possible
            if total_possible > 0
            else 0.0
        )
        return (
            f"[MultiFidelity Summary] "
            f"Queries saved: {self.total_queries_saved} / {total_possible} "
            f"({pct_saved:.1f}%) | "
            f"Stage eliminations: {self.stage_elimination_counts}"
        )

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
