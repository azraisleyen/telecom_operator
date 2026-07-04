from pathlib import Path
import json
p=Path('capstone/outputs/demo_cases/final_demo_cases.json')
if not p.exists(): raise SystemExit('Demo cases not found. Run: python capstone/scripts/run_final_pipeline.py --data-dir capstone/data --output-dir capstone/outputs')
cases=json.loads(p.read_text(encoding='utf-8'))
for i,c in enumerate(cases,1):
    print(f"\n=== Demo {i}: {c['case_name']} ===")
    print(f"Müşteri: {c['customer_id']} | Ay: {c['billing_year_month']} | Yaşam döngüsü: {c['lifecycle_status_v2']}")
    print(f"Öncelik: {c['priority_level']} | Risk: {c['business_risk_score']} ({c['business_risk_level']}) | Friction: {c['friction_score_v3']} ({c['friction_level_v3']})")
    print(f"Kuyruk: {c['recommended_queue']}")
    print(f"Kaynaklar: {', '.join(d['document_id'] for d in c['retrieved_documents'])}")
    print(f"Yanıt: {c['suggested_agent_reply']}")
print('\nÇıktılar: capstone/outputs/demo_cases/final_demo_cases.json ve capstone/outputs/final/director_demo_cases.md')
