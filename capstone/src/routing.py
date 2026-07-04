def recommended_queue(row):
    if row.has_legal_risk: return 'Legal / Escalation / Senior Support'
    if row.has_churn_risk and row.high_value_customer: return 'Retention + Senior Support'
    if row.has_churn_risk and row.has_technical_outage: return 'Technical Support + Retention'
    if row.has_billing_dispute: return 'Billing Review'
    if row.has_technical_outage: return 'Technical Support'
    if row.friction_level_v3=='high': return 'Senior Agent / Escalation'
    if row.potential_agent_mismatch: return 'KB-grounded Response Review'
    return 'Standard Support + RAG'
def recommended_business_action(row):
    acts=[]
    if row.has_billing_dispute: acts+=['proactive billing review','last 3-month bill explanation']
    if row.has_churn_risk or (row.has_churn_risk and row.high_value_customer): acts.append('retention callback')
    if row.has_technical_outage: acts.append('technical ticket')
    if row.friction_level_v3=='high': acts+=['senior agent handoff','no-repeat handoff with conversation summary']
    acts.append('source-grounded KB response')
    if row.business_risk_level=='low': acts.append('self-service guidance if source confidence is high')
    return '; '.join(dict.fromkeys(acts))
def add_routing(df):
    o=df.copy(); o['recommended_queue']=o.apply(recommended_queue,axis=1); o['recommended_business_action']=o.apply(recommended_business_action,axis=1); return o
