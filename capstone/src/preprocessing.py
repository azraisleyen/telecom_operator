from __future__ import annotations
import re, unicodedata
from pathlib import Path
import pandas as pd
TR_MAP=str.maketrans({'ç':'c','Ç':'c','ğ':'g','Ğ':'g','ı':'i','I':'i','İ':'i','ö':'o','Ö':'o','ş':'s','Ş':'s','ü':'u','Ü':'u'})
def normalize_tr_text(text: object)->str:
    if text is None or (isinstance(text,float) and pd.isna(text)): return ''
    s=str(text).translate(TR_MAP).lower(); s=unicodedata.normalize('NFKC',s)
    return re.sub(r'\s+',' ',s).strip()
def ym_to_month_index(ym: object)->int:
    s=str(ym); return int(s[:4])*12+int(s[-2:])
def safe_divide(num, den, default=0.0):
    return default if den is None or den==0 or pd.isna(den) else num/den
def save_table(df: pd.DataFrame, path: str|Path, index: bool=False):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix=='.parquet': df.to_parquet(p,index=index)
    else: df.to_csv(p,index=index)
    return p
