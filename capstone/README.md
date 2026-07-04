# Bundlekom Operator AI Console

Bundlekom Customer Risk & Resolution Copilot, Türk Telekom AI Bootcamp capstone projesi için geliştirilen modüler bir telekom müşteri deneyimi ve retention risk platformudur. Sistem tekil beş müşteri kaydını çözmek için değil; bir operatörün tekrar kullanılabilir iş kullanım senaryolarını desteklemek için tasarlanmıştır.

> **Önemli sınırlılık:** Veri setinde resmi churn etiketi yoktur. Bu nedenle proje kesin churn tahmini yaptığını iddia etmez. Dil ve metrikler `retention risk`, `lifecycle risk`, `no observed billable activity`, `extended no-spending`, `churn candidate` ve `proactive retention prioritization` çerçevesinde yorumlanmalıdır.

## İş problemi

Telekom operatörü her müşteriye aynı anda ulaşamaz. Operasyon ekipleri aşağıdaki sorulara cevap veren, açıklanabilir ve kaynaklı bir karar destek platformuna ihtiyaç duyar:

- Hangi müşteri-ayları yakın gelecekte gözlenen billable activity kaybı açısından risklidir?
- Hangi şikâyetler hukuki, retention, billing veya teknik eskalasyon gerektirir?
- Hangi konuşmalarda müşteri kendini tekrar ediyor, yanlış anlaşılıyor veya yüksek friction yaşıyor?
- Hangi high-value müşteriler proaktif korunmalıdır?
- Agent’lar bilgi tabanı kaynaklarına dayalı tutarlı Türkçe cevapları nasıl üretebilir?

## Çoklu iş kullanım senaryosu mimarisi

```text
Raw JSON/JSONL/.json.gz
  -> data_loader + preprocessing
  -> lifecycle_status_v2 + bill_shock + complaint_signals
  -> repeat_complaint_next_1m/2m + friction_score_v3
  -> business_risk_score + P0/P1/P2/P3 + routing
  -> Retention Risk ML customer-month model
  -> TF-IDF/BM25-ready RAG + optional OpenAI-compatible LLM
  -> Streamlit Bundlekom Operator AI Console
```

## İş kullanım senaryoları

### 1. Retention Risk ML Model

Model `no_billable_activity_next_2m` hedefini müşteri-ay seviyesinde tahmin eder. Bu hedef kesin churn değildir; sonraki iki ayda gözlenen billable activity olmaması için lifecycle/retention risk sinyalidir.

Modelleme müşteri-şikâyet satırında değil, `customer_id + billing_year_month` seviyesinde yapılır. Complaint count, dominant category, max friction/risk, lifecycle, high-value, bill shock ve geçmiş sinyaller customer-month tablosunda birleştirilir.

Değerlendirme accuracy ile değil operasyonel top-K metrikleriyle yapılır:

- **Precision@Top 1/5/10%:** Operasyonun ulaşabileceği en riskli müşteri segmentinde gerçek target oranını gösterir.
- **Lift@TopK:** Top segmentin genel popülasyona göre kaç kat daha yoğun risk taşıdığını gösterir.
- **PR-AUC:** Dengesiz sınıf yapısında ranking kalitesini ölçer.
- **Revenue Capture@TopK:** Riskli müşteri gelirinin ne kadarının aksiyon listesine girdiğini ölçer.
- **Brier Score:** Olasılık kalibrasyonunu kontrol eder.

### 2. LLM-Powered RAG Resolution Copilot

TF-IDF retrieval korunur; `rank-bm25` varsa BM25 ile genişletilebilir. RAG, bilgi tabanı dokümanlarını getirir ve LLM açıksa OpenAI-compatible chat client ile kısa, profesyonel, Türkçe ve kaynaklı agent yanıtı üretir.

LLM opsiyoneldir. Ana pipeline LLM anahtarı olmadan çalışır. Kaynak güveni düşükse veya LLM kapalıysa template fallback kullanılır. Yanıtlar `document_id` içermeli veya fallback vermelidir. LLM öncelik, routing, hukuki durum veya churn kararı vermez.

### 3. Risk-Aware Complaint Triage

`business_risk_score`, `business_risk_level`, `priority_level`, `recommended_queue` ve `recommended_business_action` ile şikâyetler P0/P1/P2/P3 olarak önceliklendirilir. Risk bileşenleri ayrı kolonlar halinde tutulur; böylece skor denetlenebilir.

### 4. Friction-Aware Agent Copilot

`friction_score_v3`, `friction_level_v3`, `potential_agent_mismatch`, confirmation prompt ve conversation summary agent’ın müşteriye aynı soruları tekrar sormasını azaltmayı hedefler. Final kararlar eski `friction_level` yerine `friction_level_v3` ile yapılır.

### 5. High-Value Customer Protection

High-value müşteriler tarihsel gelir segmentine göre belirlenir ve lifecycle/risk/friction/retention sinyalleriyle birleştirilerek proaktif koruma listeleri üretilir.

### 6. Unified Operator Console

Streamlit tabanlı `Bundlekom Operator AI Console`, ML, RAG, risk triage, friction assistance, high-value protection, evaluation çıktıları ve temsilî demo örneklerini tek üründe birleştirir.

## Kurulum ve çalıştırma

```bash
pip install -r capstone/requirements.txt
python capstone/scripts/run_final_pipeline.py --data-dir capstone/data --output-dir capstone/outputs --rebuild false
python capstone/scripts/train_retention_model.py --data-dir capstone/data --output-dir capstone/outputs
python capstone/scripts/evaluate_rag.py --data-dir capstone/data --output-dir capstone/outputs
streamlit run capstone/app/streamlit_app.py
```

Ham veri yoksa pipeline doğal olarak veri bulunamadığını bildirir. ML ve RAG evaluation script’leri önce final pipeline çıktılarının üretilmesini bekler.

## Beklenen ham veri

`capstone/data/raw` altında aşağıdaki dosyalar beklenir:

- `customer.json(.gz)`: `customer_id`, `city`, `birth_year`
- `customer_spending.json(.gz)`: `customer_id`, `billing_year_month`, `bill_amount`, `data_usage`
- `customer_complaints.json(.gz)`: kategori, alt kategori ve `chat_history`
- `reference_material.json(.gz)`: ardışık pretty-printed JSON objeleri

## Ana çıktılar

- `capstone/data/processed/complaints_final_features.parquet`
- `capstone/data/processed/customer_month_retention_modeling.parquet`
- `capstone/models/retention_risk_model.joblib`
- `capstone/outputs/tables/retention_model_metrics.csv`
- `capstone/outputs/tables/rag_retrieval_metrics.csv`
- `capstone/outputs/tables/rag_generation_metrics.csv`
- `capstone/outputs/tables/risk_component_summary.csv`
- `capstone/outputs/tables/repeat_complaint_summary.csv`
- `capstone/outputs/demo_cases/final_demo_cases.json`
- `capstone/outputs/final/final_executive_summary.md`

## Demo hikâyesi

Demo örnekleri bireysel müşteri kayıtlarıdır; “case” kelimesi burada business use case anlamına gelir. Demo örnekleri şu kullanım senaryolarını temsil eder:

1. Billing escalation / plan mismatch or refund / billing risk
2. Technical outage + retention risk
3. Misunderstood customer / high friction / potential mismatch
4. High-value at-risk customer
5. KB-grounded standard support / self-service

## LLM ayarları

`.env.example` dosyasını `.env` olarak kopyalayabilirsiniz:

```env
ENABLE_LLM=false
OPENAI_API_KEY=
OPENAI_BASE_URL=
CHAT_MODEL=
LLM_TEMPERATURE=0.1
```

Herhangi bir OpenAI-compatible instruct model kullanılabilir. Kod belirli bir provider veya model dayatmaz.

## Sınırlılıklar

- Resmi churn etiketi yoktur; model kesin churn tahmini değildir.
- Bill shock nedensel şikâyet tetikleyicisi olarak yorumlanmaz.
- Rule-based risk engine açıklanabilir ancak iş ağırlıkları saha geri bildirimiyle kalibre edilmelidir.
- LLM opsiyoneldir ve kaynak yoksa fallback kullanmalıdır.
- Demo örnekleri ürünün tamamı değil, iş kullanım senaryolarının temsilî kayıtlarıdır.
