"""
coverage_tracker.py
-------------------
Coverage-Guided Exploration via Defense Embeddings  (Section 4.2)

This module encodes each defense (pre_prompt + post_prompt) into embedding
space, clusters defenses into K latent "families", and tracks which clusters
have been successfully attacked across the entire fuzzing run.

Two things it provides:

  1.  Novelty(node)
      Fraction of *new* defense families broken by this mutant —
      i.e. clusters that were uncovered before this mutant was evaluated.
      Used in the combined promotion / selection score:

          Score = λ · ASR  +  (1 − λ) · Novelty

  2.  Coverage-aware defense ordering
      Returns defense indices sorted so that Stage 1 (D1) of the
      multi-fidelity scheduler samples from as many distinct families
      as possible, maximising the diagnostic value of early evaluation.

Design notes:
  - Embeddings are computed once at setup() and cached; no repeated API calls.
  - Cluster labels are also computed once (KMeans, k = cluster_k).
  - Global coverage is a set of cluster IDs updated after each iteration
    via update_coverage(node).
  - The tracker is intentionally stateless except for `covered_clusters`;
    everything else is precomputed.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from gptfuzzer.fuzzer.core import PromptNode, GPTFuzzer
    from gptfuzzer.llm import OpenAIEmbeddingLLM


class DefenseCoverageTracker:
    """
    Tracks defense-family coverage and computes novelty scores.

    Parameters
    ----------
    embedding_model : OpenAIEmbeddingLLM
        Model used to embed defense prompts (pre_prompt + post_prompt).
    cluster_k : int
        Number of defense families (clusters).  Rule of thumb: √|D|.
    lam : float
        λ in  Score = λ·ASR + (1−λ)·Novelty.
        lam=1.0 → pure ASR (disables novelty); lam=0.0 → pure novelty.
    embedding_cache_path : str or None
        If given, load/save embeddings from this .npy file to avoid
        recomputing on repeated runs.
    """

    def __init__(
        self,
        embedding_model: "OpenAIEmbeddingLLM",
        cluster_k: int = 8,
        lam: float = 0.7,
        embedding_cache_path: Optional[str] = None,
    ):
        self.embedding_model = embedding_model
        self.cluster_k = cluster_k
        self.lam = lam
        self.embedding_cache_path = embedding_cache_path

        # Populated by setup()
        self.defense_embeddings: Optional[np.ndarray] = None  # (|D|, d)
        self.defense_labels: Optional[np.ndarray] = None      # (|D|,) int cluster id
        self.cluster_centers: Optional[np.ndarray] = None     # (K, d)
        self.label_to_indices: Dict[int, List[int]] = {}      # cluster → defense indices

        # Mutable state: updated by update_coverage() after each iteration
        self.covered_clusters: set = set()

        # Diagnostics
        self.total_clusters: int = 0
        self.coverage_history: List[float] = []   # fraction covered over time

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, defenses: List[dict]):
        """
        Embed all defenses and cluster them.
        Called once before fuzzing starts.

        Parameters
        ----------
        defenses : list[dict]
            Full defense list from fuzzer.defenses.
            Each dict has keys: pre_prompt, post_prompt, access_code, ...
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError(
                "scikit-learn is required for coverage-guided exploration. "
                "Install with: pip install scikit-learn"
            )

        n = len(defenses)
        k = min(self.cluster_k, n)  # can't have more clusters than defenses
        self.total_clusters = k

        # ── 1. Compute / load embeddings ──────────────────────────────
        if self.embedding_cache_path:
            try:
                self.defense_embeddings = np.load(self.embedding_cache_path)
                logging.info(
                    f"[Coverage] Loaded cached embeddings from {self.embedding_cache_path}"
                )
            except FileNotFoundError:
                self.defense_embeddings = None

        if self.defense_embeddings is None or len(self.defense_embeddings) != n:
            logging.info(
                f"[Coverage] Computing embeddings for {n} defenses "
                f"(this makes {n} API calls once) ..."
            )
            texts = [
                (d.get("pre_prompt", "") + " " + d.get("post_prompt", "")).strip()
                for d in defenses
            ]
            vecs = [self.embedding_model.get_embedding(t) for t in texts]
            self.defense_embeddings = np.array(vecs, dtype=np.float32)

            if self.embedding_cache_path:
                np.save(self.embedding_cache_path, self.defense_embeddings)
                logging.info(f"[Coverage] Embeddings saved to {self.embedding_cache_path}")

        # ── 2. KMeans clustering ──────────────────────────────────────
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(self.defense_embeddings)
        self.defense_labels = kmeans.labels_          # shape (n,)
        self.cluster_centers = kmeans.cluster_centers_

        self.label_to_indices = {c: [] for c in range(k)}
        for idx, label in enumerate(self.defense_labels):
            self.label_to_indices[int(label)].append(idx)

        sizes = [len(v) for v in self.label_to_indices.values()]
        logging.info(
            f"[Coverage] {n} defenses clustered into {k} families. "
            f"Sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}"
        )

    # ------------------------------------------------------------------
    # Coverage-aware defense ordering
    # ------------------------------------------------------------------

    def coverage_aware_ordering(self) -> List[int]:
        """
        Return defense indices sorted so that the *first* slice covers as
        many distinct clusters as possible (round-robin over clusters),
        with remaining intra-cluster defenses appended at the end.

        This ensures D1 in the multi-fidelity scheduler samples at least one
        defense from every cluster, maximising diagnostic value of Stage 1.

        Returns
        -------
        list[int]
            Permutation of range(|D|).
        """
        # Sort clusters: uncovered clusters first, then by size descending
        def cluster_priority(cid):
            is_uncovered = 1 if cid not in self.covered_clusters else 0
            return (is_uncovered, len(self.label_to_indices[cid]))

        sorted_clusters = sorted(
            self.label_to_indices.keys(),
            key=cluster_priority,
            reverse=True,
        )

        # Round-robin: on each pass take the next unused defense from each cluster
        cluster_queues = [list(self.label_to_indices[c]) for c in sorted_clusters]
        ordered: List[int] = []

        while any(cluster_queues):
            for q in cluster_queues:
                if q:
                    ordered.append(q.pop(0))

        return ordered

    # ------------------------------------------------------------------
    # Novelty scoring
    # ------------------------------------------------------------------

    def novelty(
        self,
        node: "PromptNode",
        def_indices_evaluated: List[int],
    ) -> float:
        """
        Novelty of a mutant = fraction of *new* (uncovered) clusters
        that this mutant successfully attacked.

        Parameters
        ----------
        node : PromptNode
            Must have node.results populated (partial or full).
        def_indices_evaluated : list[int]
            Which defense indices were actually evaluated for this node
            (in order, matching node.results).

        Returns
        -------
        float in [0, 1]
        """
        if self.defense_labels is None:
            return 0.0

        newly_broken_clusters = set()
        for result, def_idx in zip(node.results, def_indices_evaluated):
            if result == 1:
                cluster = int(self.defense_labels[def_idx])
                if cluster not in self.covered_clusters:
                    newly_broken_clusters.add(cluster)

        if self.total_clusters == 0:
            return 0.0
        return len(newly_broken_clusters) / self.total_clusters

    def combined_score(
        self,
        node: "PromptNode",
        def_indices_evaluated: List[int],
        n_evaluated: int,
        T: int,
        beta: float = 1.0,
    ) -> float:
        """
        Combined promotion / selection score:

            Score = λ · LCB  +  (1 − λ) · Novelty

        LCB is the lower-confidence-bound ASR from the multi-fidelity
        scheduler.  Novelty is computed by this tracker.

        Parameters
        ----------
        node : PromptNode
        def_indices_evaluated : list[int]
            Defense indices evaluated for this node (for novelty calc).
        n_evaluated : int
            Number of defenses evaluated so far (for LCB penalty).
        T : int
            Current fuzzing iteration (for LCB penalty).
        beta : float
            LCB exploration coefficient.
        """
        import math
        n_results = len(node.results)
        if n_results == 0:
            return 0.0
        asr_hat = sum(node.results) / n_results
        penalty = beta * math.sqrt(math.log(max(T, 2)) / n_results)
        lcb = max(0.0, asr_hat - penalty)
        nov = self.novelty(node, def_indices_evaluated)
        return self.lam * lcb + (1.0 - self.lam) * nov

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update_coverage(
        self,
        node: "PromptNode",
        def_indices_evaluated: List[int],
    ):
        """
        Update global covered_clusters after a mutant is evaluated.
        Should be called in GPTFuzzer.update() for every non-eliminated node.

        Parameters
        ----------
        node : PromptNode
        def_indices_evaluated : list[int]
            Defense indices that were actually evaluated for this node.
        """
        if self.defense_labels is None:
            return
        for result, def_idx in zip(node.results, def_indices_evaluated):
            if result == 1:
                self.covered_clusters.add(int(self.defense_labels[def_idx]))

        frac = len(self.covered_clusters) / max(self.total_clusters, 1)
        self.coverage_history.append(frac)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def coverage_fraction(self) -> float:
        """Current fraction of defense families that have been broken."""
        return len(self.covered_clusters) / max(self.total_clusters, 1)

    def per_cluster_summary(self) -> str:
        lines = [
            f"[Coverage] {len(self.covered_clusters)}/{self.total_clusters} "
            f"defense families covered ({100*self.coverage_fraction():.1f}%)"
        ]
        for cid in range(self.total_clusters):
            status = "✓" if cid in self.covered_clusters else "✗"
            n_defs = len(self.label_to_indices.get(cid, []))
            lines.append(f"  Cluster {cid:2d}: {status}  ({n_defs} defenses)")
        return "\n".join(lines)

    def summary(self) -> str:
        return (
            f"[Coverage] {len(self.covered_clusters)}/{self.total_clusters} families covered "
            f"({100*self.coverage_fraction():.1f}%) | "
            f"λ={self.lam}"
        )
