import numpy as np, pandas as pd
from .preprocessing import ym_to_month_index
def build_lifecycle_v2(spending:pd.DataFrame)->pd.DataFrame:
    s=spending.copy(); s['month_index']=s.billing_year_month.map(ym_to_month_index); global_last=s.month_index.max()
    rows=[]
    for cid,g in s.sort_values('month_index').groupby('customer_id'):
        months=np.array(sorted(g.month_index.unique())); gaps=np.diff(months)-1
        active=len(months); window=months[-1]-months[0]+1; ratio=active/window if window else 0; since=global_last-months[-1]
        react=int((gaps>0).sum()); maxgap=int(gaps.max()) if len(gaps) else 0; inner=int(gaps[gaps>0].sum()) if len(gaps) else 0
        if active<3: status='short_observed_history'
        elif since>=6: status='extended_no_spending'
        elif since>=2: status='recent_no_spending'
        elif react>0 and since<=1: status='reactivated_after_gap'
        elif ratio>=.95: status='continuous_observed'
        elif since<=1 and ratio>=.70: status='active_recent_regular'
        elif since<=1: status='active_recent_irregular'
        else: status='other_irregular'
        rows.append(dict(customer_id=cid,first_active_month=g.billing_year_month.iloc[0],last_active_month=g.billing_year_month.iloc[-1],first_month_index=months[0],last_month_index=months[-1],months_active=active,observed_window_months=window,active_month_ratio=ratio,months_since_last_spending=since,inactive_gap_count=react,max_inactive_gap=maxgap,total_inner_gap_months=inner,reactivation_count=react,total_revenue=g.bill_amount.sum(),avg_bill=g.bill_amount.mean(),avg_data_usage=g.data_usage.mean(),lifecycle_status_v2=status))
    return pd.DataFrame(rows)
