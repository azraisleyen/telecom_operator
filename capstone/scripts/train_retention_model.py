from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

MSG = "Run python capstone/scripts/run_final_pipeline.py --data-dir capstone/data --output-dir capstone/outputs --rebuild false first."


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data-dir', default='capstone/data'); ap.add_argument('--output-dir', default='capstone/outputs'); args = ap.parse_args()
    data = Path(args.data_dir); out = Path(args.output_dir)
    src = data/'processed/complaints_final_features.parquet'
    if not src.exists(): raise SystemExit(MSG)
    (out/'tables').mkdir(parents=True, exist_ok=True); (data/'processed').mkdir(parents=True, exist_ok=True); Path('capstone/models').mkdir(parents=True, exist_ok=True)
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    from capstone.src.modeling import FEATURES, TARGET, build_customer_month_table, build_models, calibration_table, decile_table, evaluate_scores, save_model, score_model, valid_time_split
    from capstone.src.preprocessing import save_table
    complaints = pd.read_parquet(src)
    table = build_customer_month_table(complaints)
    save_table(table, data/'processed/customer_month_retention_modeling.parquet')
    train, val, test = valid_time_split(table)
    if train.empty or val.empty or test.empty: raise SystemExit('Not enough valid customer-month rows for train/validation/test split.')
    models = build_models(); metrics=[]; fitted={}; scores={}
    for split_name, split_df in [('validation', val), ('test', test)]:
        base_score = score_model('business_risk_baseline', split_df)
        metrics.append({'model':'business_risk_baseline', **evaluate_scores(split_df[TARGET], base_score, split_name, split_df.total_revenue)})
        scores[('business_risk_baseline', split_name)] = base_score
    for name, model in models.items():
        model.fit(train[FEATURES], train[TARGET])
        fitted[name]=model
        for split_name, split_df in [('validation', val), ('test', test)]:
            sc = score_model(model, split_df[FEATURES])
            metrics.append({'model':name, **evaluate_scores(split_df[TARGET], sc, split_name, split_df.total_revenue)})
            scores[(name, split_name)] = sc
    mdf = pd.DataFrame(metrics)
    valm = mdf[mdf.split=='validation'].copy()
    valm['selection_score'] = valm['average_precision_pr_auc'].fillna(0) + valm['precision_top_5pct'].fillna(0) + valm['lift_top_5pct'].fillna(0)/10
    best = valm.sort_values('selection_score', ascending=False).iloc[0]['model']
    best_model = fitted.get(best)
    if best_model is None:
        # fall back to best trainable model if rule baseline wins validation
        best = valm[valm.model!='business_risk_baseline'].sort_values('selection_score', ascending=False).iloc[0]['model']
        best_model = fitted[best]
    save_model(best_model, 'capstone/models/retention_risk_model.joblib')
    save_table(mdf, out/'tables/retention_model_metrics.csv')
    pk_rows=[]; rev_rows=[]; deciles=[]; calibs=[]
    for split_name, split_df in [('validation', val), ('test', test)]:
        sc = score_model(best_model, split_df[FEATURES])
        em = evaluate_scores(split_df[TARGET], sc, split_name, split_df.total_revenue)
        for k in [1,5,10]:
            pk_rows.append({'split':split_name,'top_k_percent':k,'precision':em[f'precision_top_{k}pct'],'recall':em.get(f'recall_top_{k}pct', np.nan),'lift':em[f'lift_top_{k}pct']})
            rev_rows.append({'split':split_name,'top_k_percent':k,'revenue_capture':em[f'revenue_capture_top_{k}pct']})
        order=np.argsort(-sc); n=max(1, int(np.ceil(len(sc)*0.05))); pred=np.zeros(len(sc), dtype=int); pred[order[:n]]=1
        cm=confusion_matrix(split_df[TARGET], pred, labels=[0,1])
        save_table(pd.DataFrame(cm, index=['actual_0','actual_1'], columns=['pred_0','pred_1']).reset_index(names='actual'), out/f'tables/retention_confusion_matrix_top5_{split_name}.csv')
        if split_name=='test': save_table(pd.DataFrame(cm, index=['actual_0','actual_1'], columns=['pred_0','pred_1']).reset_index(names='actual'), out/'tables/retention_confusion_matrix_top5.csv')
        deciles.append(decile_table(split_df, sc, split_name)); calibs.append(calibration_table(split_df, sc, split_name))
    save_table(pd.DataFrame(pk_rows), out/'tables/retention_precision_at_k.csv')
    save_table(pd.DataFrame(rev_rows), out/'tables/retention_revenue_capture.csv')
    save_table(pd.concat(deciles, ignore_index=True), out/'tables/retention_decile_lift.csv')
    save_table(pd.concat(calibs, ignore_index=True), out/'tables/retention_calibration_table.csv')
    top = test.copy(); top['risk_probability']=score_model(best_model, test[FEATURES]); top['risk_percentile']=top.risk_probability.rank(pct=True); save_table(top.sort_values('risk_probability', ascending=False).head(1000), out/'tables/retention_top_risk_customers.csv')
    fi = pd.DataFrame({'feature': FEATURES, 'importance': np.nan})
    try:
        est = best_model.named_steps['model']
        if hasattr(est, 'feature_importances_'):
            names = best_model.named_steps['prep'].get_feature_names_out(); fi = pd.DataFrame({'feature': names, 'importance': est.feature_importances_}).sort_values('importance', ascending=False)
        elif hasattr(est, 'coef_'):
            names = best_model.named_steps['prep'].get_feature_names_out(); fi = pd.DataFrame({'feature': names, 'importance': est.coef_[0]}).sort_values('importance', key=lambda s: s.abs(), ascending=False)
    except Exception: pass
    save_table(fi, out/'tables/retention_feature_importance.csv')
    print(f'Retention risk model trained. Selected={best}. Outputs: {out}/tables and capstone/models/retention_risk_model.joblib')
if __name__ == '__main__': main()
