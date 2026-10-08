from __future__ import annotations
import argparse,json,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Inches,Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/"scripts"/"baseline_cases.json"
OUT=ROOT/"scripts"/"results"

def call(base,path,payload=None,timeout=1800):
    t=time.perf_counter(); data=None; headers={"Accept":"application/json"}; method="GET"
    if payload is not None:
        data=json.dumps(payload,ensure_ascii=False).encode(); headers["Content-Type"]="application/json"; method="POST"
    try:
        with urlopen(Request(base.rstrip("/") + path,data=data,headers=headers,method=method),timeout=timeout) as r:
            raw=r.read().decode("utf-8","replace")
            try: obj=json.loads(raw)
            except: obj=None
            return {"status":r.status,"ms":(time.perf_counter()-t)*1000,"json":obj,"raw":raw,"error":None}
    except HTTPError as e:
        raw=e.read().decode("utf-8","replace")
        try: obj=json.loads(raw)
        except: obj=None
        return {"status":e.code,"ms":(time.perf_counter()-t)*1000,"json":obj,"raw":raw,"error":str(e)}
    except Exception as e:
        return {"status":None,"ms":(time.perf_counter()-t)*1000,"json":None,"raw":"","error":f"{type(e).__name__}: {e}"}

def ok(r):
    return not r["error"] and isinstance(r["json"],dict) and r["json"].get("success") is True

def guided_check(p):
    if not isinstance(p,dict) or p.get("success") is not True: return "backend failure"
    steps=p.get("steps")
    if not isinstance(steps,list) or len(steps)<2: return "fewer than two guided steps"
    for i,s in enumerate(steps,1):
        if not isinstance(s,dict): return f"step {i} invalid"
        opts=s.get("options")
        if not isinstance(opts,list) or len(opts)!=4: return f"step {i}: expected four options"
        ci=s.get("correct_index")
        if not isinstance(ci,int) or isinstance(ci,bool) or not 0<=ci<4: return f"step {i}: invalid correct_index={ci!r}"
    return ""

def compact(x,n=14000):
    if x is None:return ""
    s=x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2)
    return s if len(s)<=n else s[:n]+"\n… [truncated; raw JSON retains full response]"

def shade(c):
    tc=c._tc.get_or_add_tcPr(); sh=OxmlElement("w:shd"); sh.set(qn("w:fill"),"E8EDF3"); tc.append(sh)
def put(c,x,b=False,size=7):
    c.text=""; p=c.paragraphs[0]; r=p.add_run(compact(x)); r.bold=b; r.font.size=Pt(size)
def table(doc,headers):
    t=doc.add_table(rows=1,cols=len(headers)); t.style="Table Grid"; t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for i,h in enumerate(headers): put(t.rows[0].cells[i],h,True,6.5); shade(t.rows[0].cells[i])
    tr=t.rows[0]._tr.get_or_add_trPr(); tr.append(OxmlElement("w:tblHeader"))
    return t

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--base-url",default="http://127.0.0.1:8000"); ap.add_argument("--timeout",type=float,default=1800); a=ap.parse_args()
    cases=json.loads(CASES.read_text(encoding="utf-8")); OUT.mkdir(parents=True,exist_ok=True); stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    report={"started_at":datetime.now(timezone.utc).isoformat(),"base_url":a.base_url,"cases":cases,"results":[]}
    def save():
        jp=OUT/f"baseline_{stamp}.json"; dp=OUT/f"baseline_{stamp}.docx"; jp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        d=Document(); sec=d.sections[0]; sec.orientation=WD_ORIENT.LANDSCAPE; sec.page_width,sec.page_height=sec.page_height,sec.page_width
        for x in ("top_margin","bottom_margin","left_margin","right_margin"): setattr(sec,x,Inches(.4))
        d.add_heading("Nova2 — Full Pre-Change STEM Baseline",0); d.add_paragraph(f"Started: {report['started_at']} | Backend: {a.base_url}")
        d.add_paragraph("Standard: Off / Concise / Detailed. Guided: MCQs + hints + separate full solution. Accuracy Remarks intentionally blank.")
        t=table(d,["Case","Diff","Domain","Mode","Question","Actual Output","Elapsed","Error","Accuracy Remarks"])
        for r in report["results"]:
            if r["kind"]!="standard":continue
            row=t.add_row().cells
            vals=[r["id"],r["difficulty"],r["domain"],r["mode"],r["question"],r.get("response") or r.get("raw",""),f"{r['ms']/1000:.2f}s",r.get("error",""),""]
            for i,v in enumerate(vals): put(row[i],v,size=6.5)
        d.add_page_break(); d.add_heading("Guided Attempts",1)
        t=table(d,["Case","Diff","Domain","Question","Actual MCQs + Hints","MCQ time","Full Solution","Solution time","Error","Accuracy Remarks"])
        for r in report["results"]:
            if r["kind"]!="guided":continue
            row=t.add_row().cells; p=r.get("response") or {}
            vals=[r["id"],r["difficulty"],r["domain"],r["question"],p,f"{r['ms']/1000:.2f}s",r.get("solution") or r.get("solution_raw",""),f"{r.get('solution_ms',0)/1000:.2f}s",r.get("error",""),""]
            for i,v in enumerate(vals): put(row[i],v,size=6.2)
        d.save(dp); return jp,dp
    h=call(a.base_url,"/api/health",timeout=120)
    if h["error"] or not isinstance(h["json"],dict): raise SystemExit(f"Backend unavailable: {h['error'] or h['raw']}")
    total=0
    for c in cases:
        for mode in ("off","concise","detailed"):
            total+=1; print(f"[{total}] STANDARD {c['id']} {mode}",flush=True); r=call(a.base_url,"/api/reason",{"query":c["question"],"explanation":mode},a.timeout)
            report["results"].append({"kind":"standard","id":c["id"],"difficulty":c["difficulty"],"domain":c["domain"],"mode":mode,"question":c["question"],"ms":r["ms"],"response":r["json"],"raw":r["raw"],"error":r["error"] or (None if ok(r) else str((r["json"] or {}).get("error","success=false")))})
            save()
        total+=1; print(f"[{total}] GUIDED {c['id']}",flush=True)
        r=call(a.base_url,"/api/guided/start",{"query":c["question"],"include_hints":True,"include_full_solution":False},a.timeout)
        err=r["error"] or guided_check(r["json"])
        sol=None
        if not err and isinstance(r["json"],dict):
            s=call(a.base_url,"/api/guided/solution",{"query":c["question"],"steps":r["json"].get("steps",[])},a.timeout); sol=s
            if not ok(s): err=err or s["error"] or str((s["json"] or {}).get("error","success=false"))
        report["results"].append({"kind":"guided","id":c["id"],"difficulty":c["difficulty"],"domain":c["domain"],"question":c["question"],"ms":r["ms"],"response":r["json"],"raw":r["raw"],"solution":sol["json"] if sol else None,"solution_raw":sol["raw"] if sol else "","solution_ms":sol["ms"] if sol else 0,"error":err})
        save()
    report["finished_at"]=datetime.now(timezone.utc).isoformat(); report["status"]="completed"; jp,dp=save()
    print(f"JSON: {jp}\nDOCX: {dp}")

if __name__=="__main__": main()
