from __future__ import annotations

import re
import pandas as pd

from .preprocessing import normalize_tr_text, save_table

TECH_CATEGORIES = {"data", "coverage", "connectivity", "speed", "voice", "equipment", "installation", "sim_device"}
BILLING_CATEGORIES = {"billing", "roaming"}

KW = {
    "legal": ["dava", "avukat", "yasal islem", "btk", "tuketici hakem", "cimer", "mahkeme", "icra"],
    "churn": ["iptal", "kapatacagim", "hattimi tasiyacagim", "baska operator", "sozlesmeyi sonlandir", "aboneligimi iptal", "numarami tasiyacagim"],
    "billing_strong": ["fazla ucret", "beklenmeyen ucret", "haksiz ucret", "iade", "fazla faturalandirma", "mukerrer", "paket uyusmazligi", "taahhut bedeli"],
    "billing_context": ["fatura", "ucret", "taahhut", "paket"],
    "technical_strong": ["internet yok", "sinyal yok", "baglanti yok", "sebeke yok", "servis yok", "kopuyor", "modem ariz", "teknik ariza"],
    "technical_context": ["cekmiyor", "yavas internet", "mobil veri", "dns", "modem", "baglanti", "sebeke", "kapsama", "arama yapamiyorum"],
    "repetition": ["daha once denedim", "bunu daha once denedim", "ucuncu kez", "ikinci kez", "tekrar", "defalarca", "ayni sorun"],
    "unresolved": ["hala devam ediyor", "cozulmedi", "bir sey degismedi", "sonuc alamadim"],
    "frustration": ["vaktimi bosa harciyorsunuz", "kabul edilemez", "magdur", "yeter artik", "biktim", "rezalet", "sikayetciyim", "anlamiyor musunuz", "kac kere soyledim"],
    "clarification": ["anlayamadim", "tekrar eder misiniz", "biraz daha aciklar misiniz", "hangi konuda", "tam olarak", "detay verir misiniz", "kontrol edebilmem icin"],
    "billing_action": ["fatura", "ucret", "iade", "hesabinizda gerekli duzeltme", "bir sonraki faturaniza"],
    "tech_action": ["modem", "sinyal", "sebeke", "teknisyen", "baglanti", "yeniden baslat", "teknik ekip"],
}


def _pattern(term: str) -> re.Pattern:
    escaped = re.escape(normalize_tr_text(term))
    return re.compile(rf"(?<!\w){escaped}(?!\w)")


def _hits(text: str, keys: list[str]) -> list[str]:
    return [k for k in keys if _pattern(k).search(text)]


def _count_kw(text: str, keys: list[str]) -> int:
    return sum(len(_pattern(k).findall(text)) for k in keys)


def _reason(hits: list[str], prefix: str = "keyword") -> str:
    return "; ".join(f"{prefix}:{h}" for h in hits[:5])


def add_complaint_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    full = out.get("full_text", pd.Series("", index=out.index)).map(normalize_tr_text)
    cust = out.get("customer_text", full).map(normalize_tr_text)
    agent = out.get("agent_text", pd.Series("", index=out.index)).map(normalize_tr_text)
    cat = out.get("problem_category", pd.Series("", index=out.index)).astype(str).str.lower()

    legal_hits = full.map(lambda s: _hits(s, KW["legal"]))
    churn_hits = full.map(lambda s: _hits(s, KW["churn"]))
    billing_strong_hits = full.map(lambda s: _hits(s, KW["billing_strong"]))
    billing_context_hits = full.map(lambda s: _hits(s, KW["billing_context"]))
    technical_strong_hits = full.map(lambda s: _hits(s, KW["technical_strong"]))
    technical_context_hits = full.map(lambda s: _hits(s, KW["technical_context"]))

    out["has_legal_risk"] = legal_hits.map(lambda x: int(bool(x)))
    out["has_churn_risk"] = churn_hits.map(lambda x: int(bool(x)))
    out["has_billing_dispute"] = [int(bool(s) or (c in BILLING_CATEGORIES and bool(ctx))) for s, ctx, c in zip(billing_strong_hits, billing_context_hits, cat)]
    out["has_technical_outage"] = [int(bool(s) or (c in TECH_CATEGORIES and bool(ctx))) for s, ctx, c in zip(technical_strong_hits, technical_context_hits, cat)]

    out["legal_signal_reason"] = legal_hits.map(lambda x: _reason(x) if x else "")
    out["churn_signal_reason"] = churn_hits.map(lambda x: _reason(x, "retention_keyword") if x else "")
    out["billing_signal_reason"] = [_reason(s or ctx, "billing_keyword") + ("; category_guard" if c in BILLING_CATEGORIES and ctx else "") for s, ctx, c in zip(billing_strong_hits, billing_context_hits, cat)]
    out["technical_signal_reason"] = [_reason(s or ctx, "technical_keyword") + ("; category_guard" if c in TECH_CATEGORIES and ctx else "") for s, ctx, c in zip(technical_strong_hits, technical_context_hits, cat)]

    out["customer_repetition_count_v3"] = cust.map(lambda s: _count_kw(s, KW["repetition"]))
    out["unresolved_problem_count_v3"] = cust.map(lambda s: _count_kw(s, KW["unresolved"]))
    out["customer_frustration_count_v3"] = cust.map(lambda s: _count_kw(s, KW["frustration"]))
    out["agent_clarification_count_v3"] = agent.map(lambda s: _count_kw(s, KW["clarification"]))
    out["high_frustration_flag"] = (out["customer_frustration_count_v3"] > 0).astype(int)

    has_bill_action = agent.map(lambda s: bool(_hits(s, KW["billing_action"])))
    has_tech_action = agent.map(lambda s: bool(_hits(s, KW["tech_action"])))
    out["potential_agent_mismatch"] = (((cat.isin(BILLING_CATEGORIES)) & has_tech_action & ~has_bill_action) | ((cat.isin(TECH_CATEGORIES)) & has_bill_action & ~has_tech_action)).astype(int)
    return out


def export_signal_tables(df: pd.DataFrame, outdir: str) -> None:
    signals = ["has_legal_risk", "has_churn_risk", "has_billing_dispute", "has_technical_outage", "potential_agent_mismatch", "high_frustration_flag"]
    summary = pd.DataFrame([{"signal": s, "row_count": int(df[s].sum()), "rate": float(df[s].mean()) if len(df) else 0.0} for s in signals if s in df])
    by_cat = df.groupby("problem_category", dropna=False)[signals].mean().reset_index()
    save_table(summary, f"{outdir}/tables/signal_distribution.csv")
    save_table(by_cat, f"{outdir}/tables/signal_by_category.csv")
