from __future__ import annotations
import gzip,json
from pathlib import Path
from typing import Iterator
import pandas as pd
from .preprocessing import save_table
REQ={'customer':['customer_id','city','birth_year'],'spending':['customer_id','billing_year_month','bill_amount','data_usage'],'complaints':['customer_id','billing_year_month','problem_category','problem_subcategory','chat_history'],'reference':['document_id','content']}
def _open(path): return gzip.open(path,'rt',encoding='utf-8') if str(path).endswith('.gz') else open(path,encoding='utf-8')
def read_jsonl_chunks(path: str|Path, chunksize:int=100000)->Iterator[pd.DataFrame]:
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(f'Raw JSONL not found: {path}')
    buf=[]
    with _open(path) as f:
        for line in f:
            if line.strip(): buf.append(json.loads(line))
            if len(buf)>=chunksize: yield pd.DataFrame(buf); buf=[]
    if buf: yield pd.DataFrame(buf)
def read_jsonl_file(path: str|Path)->pd.DataFrame: return pd.concat(read_jsonl_chunks(path),ignore_index=True)
def read_concatenated_json(path: str|Path)->pd.DataFrame:
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(f'Concatenated JSON not found: {path}')
    txt=_open(path).read(); dec=json.JSONDecoder(); i=0; out=[]
    while i<len(txt):
        while i<len(txt) and txt[i].isspace(): i+=1
        if i>=len(txt): break
        obj,j=dec.raw_decode(txt,i); out.append(obj); i=j
    return pd.DataFrame(out)
def validate_columns(df,name):
    miss=[c for c in REQ[name] if c not in df.columns]
    if miss: raise ValueError(f'{name} missing required columns: {miss}')
def flatten_complaints(df:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for r in df.to_dict('records'):
        turns=sorted(r.get('chat_history') or [], key=lambda x:x.get('history_id',0))
        cust=[t.get('text','') for t in turns if str(t.get('site','')).lower() in ('customer','musteri','user')]
        agent=[t.get('text','') for t in turns if str(t.get('site','')).lower() not in ('customer','musteri','user')]
        rr={k:v for k,v in r.items() if k!='chat_history'}
        rr.update(turn_count=len(turns),customer_turn_count=len(cust),agent_turn_count=len(agent),customer_text=' '.join(cust),agent_text=' '.join(agent),full_text=' '.join([t.get('text','') for t in turns]),first_customer_message=cust[0] if cust else '',last_customer_message=cust[-1] if cust else '')
        rows.append(rr)
    return pd.DataFrame(rows)
def _find(data_dir,name):
    for base in [Path(data_dir)/'raw',Path(data_dir)]:
        for ext in ['.json','.json.gz']:
            p=base/f'{name}{ext}'
            if p.exists(): return p
    raise FileNotFoundError(f'Missing {name}.json(.gz) under {data_dir}/raw or {data_dir}')
def load_or_build_processed_data(data_dir='capstone/data', rebuild=False, chunksize=100000):
    data_dir=Path(data_dir); proc=data_dir/'processed'; proc.mkdir(parents=True,exist_ok=True)
    files={'customer':proc/'customer_clean.parquet','spending':proc/'spending_clean.parquet','complaints':proc/'complaints_flat.parquet','reference':proc/'reference_clean.parquet'}
    if not rebuild and all(p.exists() for p in files.values()): return {k:pd.read_parquet(v) for k,v in files.items()}
    customer=read_jsonl_file(_find(data_dir,'customer')); validate_columns(customer,'customer')
    spending=read_jsonl_file(_find(data_dir,'customer_spending')); validate_columns(spending,'spending')
    complaints=flatten_complaints(read_jsonl_file(_find(data_dir,'customer_complaints'))); validate_columns(complaints.assign(chat_history=[]),'complaints')
    reference=read_concatenated_json(_find(data_dir,'reference_material')); validate_columns(reference,'reference')
    for k,df in [('customer',customer),('spending',spending),('complaints',complaints),('reference',reference)]: save_table(df,files[k])
    return {'customer':customer,'spending':spending,'complaints':complaints,'reference':reference}
