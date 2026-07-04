from __future__ import annotations

import os
from typing import Any

from .preprocessing import normalize_tr_text
from .retrieval import build_complaint_query


def llm_configured() -> bool:
    return os.getenv("ENABLE_LLM", "false").lower() == "true" and bool(os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")) and bool(os.getenv("CHAT_MODEL"))


def _template_reply(docs, confidence: float):
    if not docs or confidence < 0.05:
        return {
            "generated_agent_reply": "Bilgi tabanında bu talep için yeterli kaynak bulunamadı. Yanlış yönlendirme yapmamak için uzman ekibe aktarım önerilir.",
            "used_document_ids": [],
            "fallback_used": True,
            "retrieval_confidence": confidence,
            "source_groundedness_flag": False,
            "llm_status": "template_fallback",
        }
    ids = [d["document_id"] for d in docs[:3]]
    return {
        "generated_agent_reply": f"Yaşadığınız durumu anladım. Bilgi tabanındaki {', '.join(ids)} kaynaklarına göre talebinizi kontrol edip uygun işlem adımlarını kaynaklı şekilde paylaşacağım. Gerekirse ilgili uzman kuyruğa kayıt açacağım.",
        "used_document_ids": ids,
        "fallback_used": False,
        "retrieval_confidence": confidence,
        "source_groundedness_flag": True,
        "llm_status": "template_fallback",
    }


def _prompt(complaint_text: str, docs: list[dict[str, Any]]) -> list[dict[str, str]]:
    sources = "\n\n".join(f"document_id={d['document_id']}\nBaşlık={d.get('title','')}\nİçerik={d.get('content_snippet','')}" for d in docs)
    system = (
        "Sen bir telekom müşteri destek yardımcı copilotusun. Yalnızca verilen KB kaynaklarını kullan. "
        "Kısa, profesyonel Türkçe yanıt yaz. document_id kaynaklarını açıkça belirt. "
        "Kaynak yetersizse bunu söyle ve uzman yönlendirme öner. Öncelik, hukuki durum, churn veya routing kararı verme. "
        "Desteklenmeyen iade, tazminat veya teknik neden uydurma."
    )
    user = f"Müşteri talebi:\n{complaint_text}\n\nKB kaynakları:\n{sources}\n\nYanıt:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_rag_response(index, complaint: str | dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    query = build_complaint_query(complaint) if isinstance(complaint, dict) else complaint
    docs = index.retrieve(query, top_k)
    confidence = float(docs[0]["score"]) if docs else 0.0
    if not docs or confidence < 0.05 or not llm_configured():
        out = _template_reply(docs, confidence)
        if not llm_configured():
            out["llm_status"] = "llm_disabled_template_fallback"
        out["retrieved_documents"] = [{k: v for k, v in d.items() if k != "content"} for d in docs]
        return out
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY"), base_url=os.getenv("OPENAI_BASE_URL") or None)
        resp = client.chat.completions.create(
            model=os.getenv("CHAT_MODEL"),
            messages=_prompt(query, docs),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        )
        text = resp.choices[0].message.content or ""
        ids = [d["document_id"] for d in docs if d["document_id"] in text]
        if not ids:
            return {**_template_reply(docs, confidence), "retrieved_documents": [{k: v for k, v in d.items() if k != "content"} for d in docs], "llm_status": "llm_missing_citation_fallback"}
        return {
            "generated_agent_reply": text.strip(),
            "used_document_ids": ids,
            "fallback_used": False,
            "retrieval_confidence": confidence,
            "source_groundedness_flag": True,
            "llm_status": "llm_generated",
            "retrieved_documents": [{k: v for k, v in d.items() if k != "content"} for d in docs],
        }
    except Exception as exc:
        out = _template_reply(docs, confidence)
        out["llm_status"] = f"llm_error_template_fallback:{type(exc).__name__}"
        out["retrieved_documents"] = [{k: v for k, v in d.items() if k != "content"} for d in docs]
        return out


def evaluate_generation(index, complaints, sample_size=200):
    smp = complaints.head(sample_size)
    rows = []
    for _, r in smp.iterrows():
        out = generate_rag_response(index, r.to_dict(), top_k=3)
        ans = out["generated_agent_reply"]
        rows.append({
            "customer_id": r.get("customer_id"),
            "problem_category": r.get("problem_category"),
            "generated_agent_reply": ans,
            "used_document_ids": ",".join(out["used_document_ids"]),
            "fallback_used": out["fallback_used"],
            "retrieval_confidence": out["retrieval_confidence"],
            "source_groundedness_flag": out["source_groundedness_flag"],
            "answer_length": len(normalize_tr_text(ans).split()),
            "low_confidence_routing": out["fallback_used"] and out["retrieval_confidence"] < 0.05,
        })
    return rows
