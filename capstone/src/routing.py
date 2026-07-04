def recommended_queue(row):
    if row.has_legal_risk:
        return "Legal / Escalation / Senior Support"
    if row.has_technical_outage and row.has_churn_risk:
        return "Technical Support + Retention"
    if row.has_billing_dispute and (row.has_churn_risk or row.high_value_customer):
        return "Billing Review + Retention"
    if row.has_churn_risk and row.high_value_customer:
        return "Retention + Senior Support"
    if row.has_billing_dispute:
        return "Billing Review"
    if row.has_technical_outage:
        return "Technical Support"
    if row.friction_level_v3 == "high":
        return "Senior Agent / No-repeat Handoff"
    if row.potential_agent_mismatch:
        return "KB-grounded Response Review"
    return "Standard Support + RAG"


def recommended_business_action(row):
    acts = []
    if row.has_legal_risk:
        acts += ["senior escalation", "legal risk review", "documented callback plan"]
    elif row.has_technical_outage and row.has_churn_risk:
        acts += ["technical outage diagnosis", "retention callback after technical update"]
    elif row.has_billing_dispute and (row.has_churn_risk or row.high_value_customer):
        acts += ["billing review", "retention-safe explanation", "proactive callback"]
    elif row.has_churn_risk and row.high_value_customer:
        acts += ["retention callback", "high-value customer protection review"]
    elif row.has_billing_dispute:
        acts += ["billing review", "explain relevant bill/plan items"]
    elif row.has_technical_outage:
        acts += ["technical ticket", "service quality follow-up"]
    elif row.friction_level_v3 == "high":
        acts += ["conversation summary", "senior agent no-repeat handoff"]
    elif row.potential_agent_mismatch:
        acts += ["KB-grounded response review", "confirm customer intent before next action"]
    else:
        acts += ["source-grounded standard support", "self-service guidance if source confidence is high"]
    acts.append("cite KB document_id or use expert fallback")
    return "; ".join(dict.fromkeys(acts))


def add_routing(df):
    o = df.copy()
    o["recommended_queue"] = o.apply(recommended_queue, axis=1)
    o["recommended_business_action"] = o.apply(recommended_business_action, axis=1)
    return o
