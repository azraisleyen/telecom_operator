import pandas as pd
from .preprocessing import save_table
def add_friction_v3(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy(); q90=out['turn_count'].quantile(.90); q95=out['turn_count'].quantile(.95)
    out['long_conversation_flag_v3']=(out['turn_count']>=q90).astype(int); out['very_long_conversation_flag_v3']=(out['turn_count']>=q95).astype(int)
    out['friction_score_v3']=(2*(out.customer_repetition_count_v3>0)+2*(out.unresolved_problem_count_v3>0)+3*(out.customer_frustration_count_v3>0)+2*(out.agent_clarification_count_v3>0)+3*(out.potential_agent_mismatch==1)+out.long_conversation_flag_v3+2*out.very_long_conversation_flag_v3).astype(int)
    severe=(out.customer_frustration_count_v3>0)&((out.customer_repetition_count_v3>0)|(out.unresolved_problem_count_v3>0)|(out.potential_agent_mismatch==1))
    out['friction_level_v3']='medium'; out.loc[(out.friction_score_v3<=1)&~severe,'friction_level_v3']='low'; out.loc[(out.friction_score_v3>=5)|severe,'friction_level_v3']='high'
    return out
def export_friction_tables(df,outdir):
    save_table(df['friction_level_v3'].value_counts().rename_axis('friction_level_v3').reset_index(name='complaint_count'),f'{outdir}/tables/friction_v3_distribution.csv')
    save_table(pd.crosstab(df.problem_category,df.friction_level_v3,normalize='index').reset_index(),f'{outdir}/tables/friction_v3_by_category.csv')
