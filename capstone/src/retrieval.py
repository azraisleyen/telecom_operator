from __future__ import annotations

import math
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .preprocessing import normalize_tr_text, save_table

CATEGORY_MAP = {
    "billing": ["faturalandirma", "fatura", "ucret", "odeme", "iade"],
    "data": ["mobil internet", "mobil veri", "internet"],
    "coverage": ["kapsama", "sinyal", "cekim"],
    "voice": ["ses", "arama"],
    "connectivity": ["baglanti"],
    "speed": ["hiz", "yavas internet"],
    "sim_device": ["sim", "cihaz"],
    "equipment": ["donanim", "modem", "cihaz"],
    "roaming": ["roaming", "yurtdisi"],
    "installation": ["kurulum"],
}


def clean_category(value):
    return normalize_tr_text(str(value).replace("**", "").replace("#", "").strip())


def category_terms(category):
    c = clean_category(category)
    return [c] + CATEGORY_MAP.get(c, [])


def parse_reference_sections(df):
    o = df.copy()
    c = o.content.fillna("")
    o["title"] = c.str.extract(r"^#\s*(.+)$", flags=re.M)[0].fillna(o.document_id.astype(str)).str.replace("**", "", regex=False).str.strip()
    o["category"] = c.str.extract(r"Kategori\s*[:：]\s*([^\n#]+)", flags=re.I)[0].fillna("").map(clean_category)
    o["problem"] = c.str.extract(r"Sorun\s*[:：]\s*([^\n#]+)", flags=re.I)[0].fillna("")
    o["solution"] = c.str.extract(r"(?:Cozum|Çözüm) Adimlari\s*[:：]?\s*(.*?)(?:\n#+\s*Notlar|$)", flags=re.I | re.S)[0].fillna("")
    o["content_norm"] = (o.title + " " + o.category + " " + o.problem + " " + o.content).map(normalize_tr_text)
    return o


class ReferenceIndex:
    def __init__(self, ref):
        self.ref = parse_reference_sections(ref)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.mat = self.vec.fit_transform(self.ref.content_norm)
        self.bm25 = None
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([t.split() for t in self.ref.content_norm])
        except Exception:
            self.bm25 = None

    def retrieve(self, query, k=3, method="tfidf"):
        qnorm = normalize_tr_text(query)
        q = self.vec.transform([qnorm])
        sims = cosine_similarity(q, self.mat).ravel()
        if method == "bm25" and self.bm25 is not None:
            bm = self.bm25.get_scores(qnorm.split())
            max_bm = max(bm) if len(bm) else 0
            if max_bm > 0:
                sims = 0.5 * sims + 0.5 * (bm / max_bm)
        idx = sims.argsort()[::-1][:k]
        return [
            dict(
                document_id=str(self.ref.iloc[i].document_id),
                title=self.ref.iloc[i].title,
                category=self.ref.iloc[i].category,
                score=float(sims[i]),
                content_snippet=str(self.ref.iloc[i].content)[:450],
                content=str(self.ref.iloc[i].content),
            )
            for i in idx
        ]


def build_reference_index(reference_df):
    return ReferenceIndex(reference_df)


def build_complaint_query(row):
    base = " ".join(str(row.get(c, "")) for c in ["problem_category", "problem_subcategory", "first_customer_message", "customer_text"])
    expanded = " ".join(category_terms(row.get("problem_category", "")))
    return (expanded + " " + base)[:2500]


def retrieve_top_k(index, query, k=3):
    return index.retrieve(query, k)


def _is_relevant(doc, row):
    cat_terms = category_terms(row.get("problem_category", ""))
    sub = normalize_tr_text(row.get("problem_subcategory", ""))
    hay = normalize_tr_text(" ".join([doc.get("category", ""), doc.get("title", ""), doc.get("content_snippet", "")]))
    return any(t and t in hay for t in cat_terms) or (bool(sub) and sub in hay)


def evaluate_retrieval(index, complaints, outdir, sample_size=5000, k=5):
    smp = complaints.dropna(subset=["problem_category"]).sample(min(sample_size, len(complaints)), random_state=42) if len(complaints) > sample_size else complaints
    rows = []
    for _, r in smp.iterrows():
        docs = index.retrieve(build_complaint_query(r), k)
        rel = [_is_relevant(d, r) for d in docs]
        rr = next((1 / (i + 1) for i, ok in enumerate(rel[:5]) if ok), 0.0)
        dcg = sum((1 if ok else 0) / math.log2(i + 2) for i, ok in enumerate(rel[:5]))
        idcg = 1.0
        rows.append({
            "customer_id": r.customer_id,
            "problem_category": r.problem_category,
            "problem_subcategory": r.get("problem_subcategory", ""),
            "top1_document_id": docs[0]["document_id"] if docs else "",
            "top1_score": docs[0]["score"] if docs else 0,
            "hit_at_1": int(any(rel[:1])),
            "hit_at_3": int(any(rel[:3])),
            "hit_at_5": int(any(rel[:5])),
            "precision_at_1": float(sum(rel[:1]) / 1) if docs else 0.0,
            "precision_at_3": float(sum(rel[:3]) / min(3, len(docs))) if docs else 0.0,
            "mrr_at_5": rr,
            "ndcg_at_5": dcg / idcg,
        })
    res = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "hit_at_1": res.hit_at_1.mean() if len(res) else 0,
        "hit_at_3": res.hit_at_3.mean() if len(res) else 0,
        "hit_at_5": res.hit_at_5.mean() if len(res) else 0,
        "precision_at_1": res.precision_at_1.mean() if len(res) else 0,
        "precision_at_3": res.precision_at_3.mean() if len(res) else 0,
        "mrr_at_5": res.mrr_at_5.mean() if len(res) else 0,
        "ndcg_at_5": res.ndcg_at_5.mean() if len(res) else 0,
        "avg_top1_score": res.top1_score.mean() if len(res) else 0,
        "median_top1_score": res.top1_score.median() if len(res) else 0,
        "source_coverage_rate": (res.top1_document_id != "").mean() if len(res) else 0,
    }])
    save_table(summary, f"{outdir}/tables/rag_retrieval_readiness_summary.csv")
    save_table(summary, f"{outdir}/tables/rag_retrieval_metrics.csv")
    save_table(res.head(200), f"{outdir}/tables/rag_sample_results.csv")
    return summary
