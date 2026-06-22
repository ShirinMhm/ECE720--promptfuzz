import random
import numpy as np

from gptfuzzer.fuzzer import GPTFuzzer, PromptNode


class SelectPolicy:
    def __init__(self, fuzzer: GPTFuzzer):
        self.fuzzer = fuzzer

    def select(self) -> PromptNode:
        raise NotImplementedError(
            "SelectPolicy must implement select method.")

    def update(self, prompt_nodes: 'list[PromptNode]'):
        pass


class RoundRobinSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: GPTFuzzer = None):
        super().__init__(fuzzer)
        self.index: int = 0

    def select(self) -> PromptNode:
        seed = self.fuzzer.prompt_nodes[self.index]
        seed.visited_num += 1
        return seed

    def update(self, prompt_nodes: 'list[PromptNode]'):
        self.index = (self.index - 1 + len(self.fuzzer.prompt_nodes)
                      ) % len(self.fuzzer.prompt_nodes)


class RandomSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: GPTFuzzer = None):
        super().__init__(fuzzer)

    def select(self) -> PromptNode:
        seed = random.choice(self.fuzzer.prompt_nodes)
        seed.visited_num += 1
        return seed


class UCBSelectPolicy(SelectPolicy):
    def __init__(self,
                 explore_coeff: float = 1.0,
                 fuzzer: GPTFuzzer = None):
        super().__init__(fuzzer)

        self.step = 0
        self.last_choice_index = None
        self.explore_coeff = explore_coeff
        self.rewards = [0 for _ in range(len(self.fuzzer.prompt_nodes))]

    def select(self) -> PromptNode:
        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend(
                [0 for _ in range(len(self.fuzzer.prompt_nodes) - len(self.rewards))])

        self.step += 1
        scores = np.zeros(len(self.fuzzer.prompt_nodes))
        for i, prompt_node in enumerate(self.fuzzer.prompt_nodes):
            smooth_visited_num = prompt_node.visited_num + 1
            scores[i] = self.rewards[i] / smooth_visited_num + \
                self.explore_coeff * \
                np.sqrt(2 * np.log(self.step) / smooth_visited_num)

        self.last_choice_index = np.argmax(scores)
        self.fuzzer.prompt_nodes[self.last_choice_index].visited_num += 1
        return self.fuzzer.prompt_nodes[self.last_choice_index]

    def update(self, prompt_nodes: 'list[PromptNode]'):
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes])
        self.rewards[self.last_choice_index] += \
            succ_num / len(self.fuzzer.defenses)


class MCTSExploreSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: GPTFuzzer = None, ratio=0.5, alpha=0.1, beta=0.2):
        super().__init__(fuzzer)

        self.step = 0
        self.mctc_select_path: 'list[PromptNode]' = []
        self.last_choice_index = None
        self.rewards = []
        self.ratio = ratio  # balance between exploration and exploitation
        self.alpha = alpha  # penalty for level
        self.beta = beta   # minimal reward after penalty

    def select(self) -> PromptNode:
        self.step += 1
        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend(
                [0 for _ in range(len(self.fuzzer.prompt_nodes) - len(self.rewards))])

        self.mctc_select_path.clear()
        cur = max(
            self.fuzzer.initial_prompts_nodes,
            key=lambda pn:
            self.rewards[pn.index] / (pn.visited_num + 1) +
            self.ratio * np.sqrt(2 * np.log(self.step) /
                                 (pn.visited_num + 0.01))
        )
        self.mctc_select_path.append(cur)

        while len(cur.child) > 0:
            if np.random.rand() < self.alpha:
                break
            cur = max(
                cur.child,
                key=lambda pn:
                self.rewards[pn.index] / (pn.visited_num + 1) +
                self.ratio * np.sqrt(2 * np.log(self.step) /
                                     (pn.visited_num + 0.01))
            )
            self.mctc_select_path.append(cur)

        for pn in self.mctc_select_path:
            pn.visited_num += 1

        self.last_choice_index = cur.index
        return cur

    def update(self, prompt_nodes: 'list[PromptNode]'):
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes])

        last_choice_node = self.fuzzer.prompt_nodes[self.last_choice_index]
        for prompt_node in reversed(self.mctc_select_path):
            reward = succ_num / (len(self.fuzzer.defenses)
                                 * len(prompt_nodes))
            self.rewards[prompt_node.index] += reward * \
                max(self.beta, (1 - 0.1 * last_choice_node.level))


class EXP3SelectPolicy(SelectPolicy):
    def __init__(self,
                 gamma: float = 0.05,
                 alpha: float = 25,
                 fuzzer: GPTFuzzer = None):
        super().__init__(fuzzer)

        self.energy = self.fuzzer.energy
        self.gamma = gamma
        self.alpha = alpha
        self.last_choice_index = None
        self.weights = [1. for _ in range(len(self.fuzzer.prompt_nodes))]
        self.probs = [0. for _ in range(len(self.fuzzer.prompt_nodes))]

    def select(self) -> PromptNode:
        if len(self.fuzzer.prompt_nodes) > len(self.weights):
            self.weights.extend(
                [1. for _ in range(len(self.fuzzer.prompt_nodes) - len(self.weights))])

        np_weights = np.array(self.weights)
        probs = (1 - self.gamma) * np_weights / np_weights.sum() + \
            self.gamma / len(self.fuzzer.prompt_nodes)

        self.last_choice_index = np.random.choice(
            len(self.fuzzer.prompt_nodes), p=probs)

        self.fuzzer.prompt_nodes[self.last_choice_index].visited_num += 1
        self.probs[self.last_choice_index] = probs[self.last_choice_index]

        return self.fuzzer.prompt_nodes[self.last_choice_index]

    def update(self, prompt_nodes: 'list[PromptNode]'):
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes])

        r = 1 - succ_num / len(prompt_nodes)
        x = -1 * r / self.probs[self.last_choice_index]
        self.weights[self.last_choice_index] *= np.exp(
            self.alpha * x / len(self.fuzzer.prompt_nodes))


class CoverageGuidedSelectPolicy(SelectPolicy):
    """
    Seed selection policy that combines ASR reward with defense-family novelty.

    Score = λ · (ASR reward) + (1 − λ) · Novelty

    Where:
      - ASR reward  = MCTS-style UCB reward normalised by visits (same as
                      MCTSExploreSelectPolicy)
      - Novelty     = fraction of uncovered defense clusters that this seed
                      (and its best descendant) has successfully attacked

    This ensures that seeds which open up new defense families are preferred
    over seeds with slightly higher raw ASR but no new coverage.

    Parameters
    ----------
    coverage_tracker : DefenseCoverageTracker
        Must already be set up (setup() called).
    lam : float
        λ weight. Overrides coverage_tracker.lam if provided.
    ratio : float
        UCB exploration ratio (same role as in MCTSExploreSelectPolicy).
    alpha : float
        MCTS level-penalty coefficient.
    beta : float
        UCB lower bound coefficient.
    """

    def __init__(
        self,
        coverage_tracker,
        lam: float = None,
        ratio: float = 0.5,
        alpha: float = 0.1,
        beta: float = 0.2,
        fuzzer: GPTFuzzer = None,
    ):
        super().__init__(fuzzer)
        self.coverage_tracker = coverage_tracker
        # lam: use tracker's value by default so there's one source of truth
        self.lam = lam if lam is not None else coverage_tracker.lam
        self.ratio = ratio
        self.alpha = alpha
        self.beta = beta

        self.step = 0
        self.mctc_select_path = []
        self.last_choice_index = None
        self.rewards = []

    def select(self) -> PromptNode:
        self.step += 1
        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend(
                [0.0] * (len(self.fuzzer.prompt_nodes) - len(self.rewards))
            )

        self.mctc_select_path.clear()

        def _score(pn: PromptNode) -> float:
            ucb_term = (
                self.rewards[pn.index] / (pn.visited_num + 1)
                + self.ratio * np.sqrt(2 * np.log(self.step) / (pn.visited_num + 0.01))
            )
            # Novelty: use the last evaluated defense indices stored by the scheduler,
            # or fall back to 0 if scheduler hasn't set them yet
            scheduler = getattr(self.fuzzer, "scheduler", None)
            if scheduler is not None and hasattr(scheduler, "_last_eval_def_indices"):
                def_indices = scheduler._last_eval_def_indices.get(id(pn), [])
            else:
                def_indices = []
            nov = self.coverage_tracker.novelty(pn, def_indices)
            return self.lam * ucb_term + (1.0 - self.lam) * nov

        cur = max(self.fuzzer.initial_prompts_nodes, key=_score)
        self.mctc_select_path.append(cur)

        while len(cur.child) > 0:
            if np.random.rand() < self.alpha:
                break
            cur = max(cur.child, key=_score)
            self.mctc_select_path.append(cur)

        for pn in self.mctc_select_path:
            pn.visited_num += 1

        self.last_choice_index = cur.index
        return cur

    def update(self, prompt_nodes):
        succ_num = sum(pn.num_jailbreak for pn in prompt_nodes)
        last_choice_node = self.fuzzer.prompt_nodes[self.last_choice_index]
        for pn in reversed(self.mctc_select_path):
            reward = succ_num / (len(self.fuzzer.defenses) * len(prompt_nodes))
            self.rewards[pn.index] += reward * max(
                self.beta, (1 - 0.1 * last_choice_node.level)
            )
