from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

MSG = "Run python capstone/scripts/run_final_pipeline.py --data-dir capstone/data --output-dir capstone/outputs --rebuild false first."

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data-dir', default='capstone/data'); ap.add_argument('--output-dir', default='capstone/outputs'); ap.add_argument('--sample-size', type=int, default=200); args = ap.parse_args()
    data = Path(args.data_dir) / 'processed'; out = Path(args.output_dir)
    cp = data/'complaints_final_features.parquet'; rp = data/'reference_clean.parquet'
    if not cp.exists() or not rp.exists():
        raise SystemExit(MSG)
    (out/'tables').mkdir(parents=True, exist_ok=True)
    import pandas as pd
    from capstone.src.retrieval import build_reference_index, evaluate_retrieval
    from capstone.src.llm_rag import evaluate_generation
    from capstone.src.preprocessing import save_table
    complaints = pd.read_parquet(cp); reference = pd.read_parquet(rp); index = build_reference_index(reference)
    retrieval = evaluate_retrieval(index, complaints, args.output_dir, sample_size=min(5000, len(complaints)), k=5)
    gen_rows = pd.DataFrame(evaluate_generation(index, complaints, args.sample_size))
    gen_summary = pd.DataFrame([{
        'answer_has_source_id_rate': float((gen_rows.used_document_ids.astype(str) != '').mean()) if len(gen_rows) else 0,
        'fallback_rate': float(gen_rows.fallback_used.mean()) if len(gen_rows) else 0,
        'avg_answer_length': float(gen_rows.answer_length.mean()) if len(gen_rows) else 0,
        'grounded_response_rate': float(gen_rows.source_groundedness_flag.mean()) if len(gen_rows) else 0,
        'low_confidence_routing_rate': float(gen_rows.low_confidence_routing.mean()) if len(gen_rows) else 0,
    }])
    save_table(gen_summary, out/'tables/rag_generation_metrics.csv')
    save_table(gen_rows.head(200), out/'tables/rag_eval_sample.csv')
    save_table(gen_rows[(gen_rows.fallback_used) | (~gen_rows.source_groundedness_flag)].head(100), out/'tables/rag_failure_cases.csv')
    print('RAG evaluation complete. Outputs:', out/'tables')
if __name__ == '__main__': main()
