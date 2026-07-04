# Bundlekom Customer Risk & Resolution Copilot

Bundlekom Customer Risk & Resolution Copilot, Türk Telekom AI Bootcamp capstone projesi için hazırlanmış açıklanabilir bir müşteri operasyon zekâsı akışıdır. Sistem yalnızca şikâyeti sınıflandırmaz; müşteri yaşam döngüsü, tarihsel gelir değeri, fatura şoku, konuşma sürtünmesi, iş riski, doğru operasyon kuyruğu ve bilgi tabanı kaynaklı yanıtı tek bir karar destek çıktısında birleştirir.

> Önemli: Veri setinde resmi churn etiketi yoktur. Bu nedenle proje “kesin churn” iddiasında bulunmaz; “lifecycle risk”, “no observed billable activity”, “extended no-spending”, “retention risk” ve “churn candidate” dilini kullanır.

## Neden sadece chatbot değil?

Kontrolsüz bir sohbet botu yerine kaynakları görülebilen, iş kuralları denetlenebilir ve operasyon aksiyonu üreten bir copilot tasarlanmıştır:

```text
Raw JSON/JSONL
  -> data_loader + preprocessing
  -> lifecycle_status_v2 + bill_shock + complaint_signals
  -> friction_score_v3 + business_risk_score
  -> routing + recommended_business_action
  -> TF-IDF RAG over reference_material
  -> 5 director-ready demo cases + executive summaries
```

## Veri özeti

Beklenen ham dosyalar `capstone/data/raw` altında bulunur:

- `customer.json(.gz)`: `customer_id`, `city`, `birth_year`
- `customer_spending.json(.gz)`: `customer_id`, `billing_year_month`, `bill_amount`, `data_usage`
- `customer_complaints.json(.gz)`: kategori, alt kategori ve `chat_history`
- `reference_material.json(.gz)`: ardışık pretty-printed JSON objeleri; JSONL değildir

Mevcut analiz bulgularına göre tam ölçekte 200K müşteri, 4.32M harcama satırı, 2.16M şikâyet ve 1,000 bilgi tabanı dokümanı vardır. Billing en büyük şikâyet alanıdır; bill shock bağlamsal bir risk sinyalidir fakat aynı ay billing şikâyetini anlamlı şekilde artırdığı iddia edilmez.

## Modüller

- `capstone/src/data_loader.py`: JSONL, gzip ve concatenated JSON okuma; complaint flattening; parquet üretimi
- `capstone/src/preprocessing.py`: Türkçe normalizasyon, ay indeksi, güvenli bölme, tablo kaydetme
- `capstone/src/lifecycle.py`: `lifecycle_status_v2` ve no-billable-activity odaklı yaşam döngüsü metrikleri
- `capstone/src/bill_shock.py`: önceki fatura, 3 aylık rolling ortalama, fatura değişimi ve next-2-month no billable activity
- `capstone/src/complaint_signals.py`: hukuki risk, iptal/retention, billing dispute, teknik kesinti ve v3 konuşma sinyalleri
- `capstone/src/friction_scoring.py`: eski `friction_level` kullanılmadan `friction_score_v3` ve `friction_level_v3`
- `capstone/src/risk_scoring.py`: açıklanabilir iş riski ve P0/P1/P2/P3 önceliklendirme
- `capstone/src/routing.py`: operasyon kuyruğu ve iş aksiyonu önerisi
- `capstone/src/retrieval.py`: TF-IDF tabanlı, inspectable V1 RAG ve retrieval değerlendirmesi
- `capstone/src/response_generation.py`: her zaman çalışan template yanıt; opsiyonel, düşük sıcaklıklı ve kaynak kısıtlı LLM modu
- `capstone/src/demo_cases.py`: tam olarak 5 director-ready demo case üretimi
- `capstone/src/evaluation.py`: final yönetici özetleri ve tablolar

## Çalıştırma

```bash
python capstone/scripts/run_final_pipeline.py --data-dir capstone/data --output-dir capstone/outputs --rebuild false
python capstone/scripts/run_demo_cases.py
```

`--rebuild true` ham veriden parquet dosyalarını yeniden üretir. Pipeline idempotent tasarlanmıştır; işlenmiş parquet dosyaları varsa tekrar kullanılır.

## Üretilen ana çıktılar

- `capstone/data/processed/customer_clean.parquet`
- `capstone/data/processed/spending_clean.parquet`
- `capstone/data/processed/complaints_flat.parquet`
- `capstone/data/processed/reference_clean.parquet`
- `capstone/data/processed/complaints_final_features.parquet`
- `capstone/outputs/tables/business_risk_level_summary.csv`
- `capstone/outputs/tables/friction_v3_distribution.csv`
- `capstone/outputs/tables/rag_retrieval_readiness_summary.csv`
- `capstone/outputs/demo_cases/final_demo_cases.json`
- `capstone/outputs/final/director_demo_cases.md`
- `capstone/outputs/final/final_executive_summary.md`

## 5 demo case

Pipeline gerçek veri satırlarından şu beş sunum vakasını seçer:

1. Billing escalation / legal or highest billing risk
2. Technical outage + churn risk
3. Misunderstood customer / high friction
4. High-value at-risk customer
5. KB-grounded standard support / self-service

Hukuki anahtar kelime yoksa sistem hukuki vaka uydurmaz; en yüksek riskli gerçek billing escalation satırını seçer.

## Temel bulgular ve dürüst yorum

- Billing hacim önceliğidir; billing tekrarları operasyonel iyileştirme alanıdır.
- Friction v3 tek başına churn modeli değildir; konuşma kalitesi, escalation ve senior-agent routing sinyalidir.
- Business risk score, retention ve müşteri deneyimi aksiyonlarını görünür kılar.
- High-value at-risk segmenti müşteri yaşam boyu değerini koruma açısından önceliklidir.
- TF-IDF RAG, V1 için güçlü, hızlı ve açıklanabilir bir kaynak bulma yaklaşımıdır.
- Kaynak doküman ID’lerinin yanıta eklenmesi yanıt tutarsızlığını azaltır.

## Sınırlılıklar

- Resmi churn etiketi yoktur; model “kesin churn” tahmini yapmaz.
- Bill shock nedensel şikâyet tetikleyicisi olarak yorumlanmaz.
- Age/city bireysel risk ayrımcılığı için değil, yalnızca agregat deneyim analizi için kullanılmalıdır.
- Opsiyonel LLM modu baseline için gerekli değildir ve sadece retrieved KB kaynaklarını özetlemek için kullanılmalıdır.

## Gelecek yol haritası

- BM25/embedding retrieval karşılaştırması
- pgvector veya benzeri vektör indeksleri
- FastAPI/Streamlit demo arayüzü
- Zaman penceresi dikkatle tasarlanmış lifecycle risk modeli
- Agentic RAG yalnızca güçlü guardrail ve kaynak denetimiyle
