import os

def confirmation_prompt(row): return f"Müşterinin ana talebini '{row.get('problem_category','')}/{row.get('problem_subcategory','')}' olarak anladım. Yanlış anlamamak için: Öncelik çözülmeyen konu ve beklenen aksiyon bu mu?"
def conversation_summary(row): return f"Müşteri {row.get('billing_year_month','')} döneminde {row.get('problem_category','')} konusunda başvurdu. Friction={row.get('friction_level_v3')} risk={row.get('business_risk_level')} öncelik={row.get('priority_level')}."
def suggested_agent_reply(row, docs):
    if not docs or docs[0].get('score',0)<0.05: return 'Bilgi tabanında yeterli kaynak bulunamadı; uzman ekibe yönlendirme önerilir.'
    ids=', '.join(d['document_id'] for d in docs)
    return f"Yaşadığınız durumu anladım. Bilgi tabanındaki {ids} dokümanlarına göre işlemi kaynaklı şekilde kontrol edeceğim; gerekli ise ilgili uzman kuyruğa kayıt açacağım."
def build_template_response(row, docs):
    return {'confirmation_prompt':confirmation_prompt(row),'conversation_summary':conversation_summary(row),'suggested_agent_reply':suggested_agent_reply(row,docs),'recommended_business_action':row.get('recommended_business_action','source-grounded KB response')}
def llm_enabled(): return os.getenv('ENABLE_LLM','').lower()=='true' and (os.getenv('OPENAI_API_KEY') or os.getenv('API_KEY')) and os.getenv('CHAT_MODEL')
