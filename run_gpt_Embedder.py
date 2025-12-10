import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class SectionEmbedder:
    """
    Wraps a sentence-transformers model and precomputes embeddings for TOC sections.
    """

    def __init__(
        self,
        sections: List[Dict[str, Any]],
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.sections = sections
        self.model = SentenceTransformer(model_name)

        # We'll embed the "full" path string for each section
        texts = [s["full"] for s in sections]
        self.section_embs = self.model.encode(
            texts,
            normalize_embeddings=True
        )  # shape: (num_sections, dim)

    def match_by_embedding(
        self,
        topic: str,
        top_k: int = 5,
        min_sim: float = 0.40,
    ) -> List[Dict[str, Any]]:
        """
        Return best-matching sections by cosine similarity.

        Behaviour:
        - If some results exceed min_sim → return only those (up to top_k).
        - If none exceed min_sim → return the single best (highest-similarity)
          section anyway.

        Each returned dict is a copy of the original section with an extra
        "similarity" field.
        """
        topic_norm = topic.strip().replace("_", " ")
        if not topic_norm:
            # Degenerate case: no topic → just return the top 1 section
            idx = 0
            sec = self.sections[idx].copy()
            sec["similarity"] = 0.0
            return [sec]

        topic_emb = self.model.encode(
            [topic_norm],
            normalize_embeddings=True
        )[0]

        sims = np.dot(self.section_embs, topic_emb)  # cosine sim via normalization
        idxs = np.argsort(-sims)  # descending

        best_matches: List[Dict[str, Any]] = []

        # collect those above threshold
        for idx in idxs[:top_k]:
            score = float(sims[idx])
            if score >= min_sim:
                sec = self.sections[idx].copy()
                sec["similarity"] = score
                best_matches.append(sec)

        if best_matches:
            return best_matches

        # No one passed threshold: return the single closest
        best_idx = idxs[0]
        sec = self.sections[best_idx].copy()
        sec["similarity"] = float(sims[best_idx])
        return [sec]

