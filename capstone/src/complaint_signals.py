from __future__ import annotations
import pandas as pd
from .preprocessing import normalize_tr_text
KW={
'legal':['dava','avukat','yasal islem','btk','tuketici hakem','cimer','mahkeme'],
'churn':['iptal','kapatacagim','hattimi tasiyacagim','baska operator','sozlesmeyi sonlandir','aboneligimi iptal'],
'billing':['fatura','fazla ucret','beklenmeyen ucret','iade','haksiz','taahhut','paket yanlis','paket uyusmazligi','fazla faturalandirma'],
'technical':['internet yok','cekmiyor','sinyal yok','baglanti yok','kopuyor','yavas','mobil veri','dns','modem','hiz'],
'repetition':['daha once denedim','bunu daha once denedim','ucuncu kez','ikinci kez','tekrar','defalarca','ayni sorun'],
'unresolved':['hala devam ediyor','cozulmedi','bir sey degismedi'],
'frustration':['vaktimi bosa harciyorsunuz','kabul edilemez','magdur','yeter artik','biktim','rezalet','sikayetciyim','anlamiyor musunuz','kac kere soyledim'],
'clarification':['anlayamadim','tekrar eder misiniz','biraz daha aciklar misiniz','hangi konuda','tam olarak','detay verir misiniz','kontrol edebilmem icin'],
'billing_action':['fatura','ucret','iade','hesabinizda gerekli duzeltme','bir sonraki faturaniza'],
'tech_action':['modem','sinyal','sebeke','teknisyen','baglanti','yeniden baslat','teknik ekip']}
def count_kw(s, keys): return sum(s.count(k) for k in keys)
def add_complaint_signals(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy(); full=out.get('full_text',pd.Series('',index=out.index)).map(normalize_tr_text); cust=out.get('customer_text',full).map(normalize_tr_text); agent=out.get('agent_text',pd.Series('',index=out.index)).map(normalize_tr_text)
    for name,keys,col in [('legal',KW['legal'],'has_legal_risk'),('churn',KW['churn'],'has_churn_risk'),('billing',KW['billing'],'has_billing_dispute'),('technical',KW['technical'],'has_technical_outage')]: out[col]=full.map(lambda s:int(any(k in s for k in keys)))
    out['customer_repetition_count_v3']=cust.map(lambda s:count_kw(s,KW['repetition']))
    out['unresolved_problem_count_v3']=cust.map(lambda s:count_kw(s,KW['unresolved']))
    out['customer_frustration_count_v3']=cust.map(lambda s:count_kw(s,KW['frustration']))
    out['agent_clarification_count_v3']=agent.map(lambda s:count_kw(s,KW['clarification']))
    out['high_frustration_flag']=(out['customer_frustration_count_v3']>0).astype(int)
    has_bill_action=agent.map(lambda s:any(k in s for k in KW['billing_action'])); has_tech_action=agent.map(lambda s:any(k in s for k in KW['tech_action']))
    cat=out['problem_category'].astype(str).str.lower()
    out['potential_agent_mismatch']=(((cat.isin(['billing','roaming'])) & has_tech_action & ~has_bill_action) | ((cat.isin(['data','coverage','connectivity','speed','voice','equipment'])) & has_bill_action & ~has_tech_action)).astype(int)
    return out
