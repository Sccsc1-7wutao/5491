"""Batch-1 descriptive attribute tables for VisEvent and DepthTrack.

Inputs (read only): VisEvent annotations/results and batch1_supp/depth_ci.json.
Outputs: batch1_supp/{visevent_attributes.csv,depthtrack_attributes.csv,attributes.md}.
Run: D:\Conda\envs\sdstrack\python.exe D:\cvresearch\scripts\batch1_attributes.py
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, r"D:\cvresearch\scripts")
from visevent_eval import sequence_scores

VIPT=Path(r"D:\cvresearch\ViPT"); OUT=VIPT/"output"/"eval_results"/"batch1_supp"
TK=Path(r"D:\cvresearch\tools\VisEvent_SOT_Benchmark")
V=("fedavg","modality","modality_a05","ours","fedprox")
P=(("modality","fedavg"),("ours","modality"),("modality_a05","fedavg"),("fedprox","fedavg"))

def paired(a,b):
    d=np.asarray(a)-np.asarray(b); n=len(d); mean=float(d.mean())
    se=float(d.std(ddof=1)/np.sqrt(n)) if n>1 else float('nan')
    t=mean/se if se else float('nan')
    return mean,t,n,int((d>0).sum()),int((d<0).sum())

def make_rows(labels, values):
    rows=[]
    for label, vals in labels:
        row={"attribute":label,"n":len(vals)}
        for v in V: row[f"{v}_mean"] = float(np.mean([values[v][i] for i in vals]))
        for left,right in P:
            m,t,n,w,l=paired([values[left][i] for i in vals],[values[right][i] for i in vals])
            key=f"{left}_minus_{right}"; row.update({f"{key}_diff":m,f"{key}_t":t,f"{key}_n":n,f"{key}_wins":w,f"{key}_losses":l})
        rows.append(row)
    return rows

def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    seqs=[x.strip() for x in (TK/'sequence_evaluation_config'/'VisEvent_testing_subset.txt').read_text().splitlines() if x.strip()]
    scores={}
    for v in V:
        rd=VIPT/'RGBE_workspace'/'results'/'VisEvent'/f'batch1_{v}_event'
        s=sequence_scores(TK/'annos',rd,seqs)
        scores[v]={"SR":np.array([s[x][1] for x in seqs]),"PR":np.array([s[x][0] for x in seqs])}
    names=['CM','ROT','DEF','FOC','LI','OV','POC','VC','SV','BC','MB','ARC','FM','NMO','IV','OE','BOM']
    att=np.array([np.loadtxt(TK/'annos'/'att'/f'{s}.txt',ndmin=1) for s in seqs])
    labels=[(name,np.flatnonzero(att[:,i]>0)) for i,name in enumerate(names)]
    outdoor=np.array(['outdoor' in s.lower() for s in seqs])
    labels += [('name_outdoor',np.flatnonzero(outdoor)),('name_without_outdoor',np.flatnonzero(~outdoor))]
    ve_rows=make_rows(labels,{v:scores[v]['SR'] for v in V})
    for r in ve_rows: r['metric']='SR'; r['note']='sequence-level; descriptive only'
    write_csv(OUT/'visevent_attributes.csv',ve_rows)
    overall={v:float(scores[v]['SR'].mean()) for v in V}
    expected={'fedavg':.5304,'modality':.5607,'modality_a05':.5460,'ours':.5647,'fedprox':.5307}
    if max(abs(overall[v]-expected[v]) for v in V)>5e-5: raise RuntimeError(f'VisEvent SR acceptance failed: {overall}')
    depth=json.loads((OUT/'depth_ci.json').read_text(encoding='utf-8'))
    dn=depth['sequence_names']; ix={n:i for i,n in enumerate(dn)}
    dlabels=[('indoor',[i for i,n in enumerate(dn) if n.endswith('_indoor')]),('wild',[i for i,n in enumerate(dn) if n.endswith('_wild')])]
    # A tag excludes sequences without tag=1 frames; scores use fixed variant-global optimal threshold.
    tags=depth['attributes']['fedavg']['tags']
    dvalues={v:np.array([depth['per_sequence_f_at_variant_global_optimum'][v][n] for n in dn]) for v in V}
    drows=make_rows(dlabels,dvalues)
    for r in drows: r.update({'metric':'F','frames':'','note':'sequence F at each variant global optimum'})
    for tag in tags:
        present=sorted(tag['per_sequence_f'])
        vals={v:np.array([depth['attributes'][v]['tags'][[x['attribute'] for x in depth['attributes'][v]['tags']].index(tag['attribute'])]['per_sequence_f'][n] for n in present]) for v in V}
        tr=make_rows([(tag['attribute'],list(range(len(present))))],vals)[0]
        tr.update({'metric':'F','frames':tag['frames'],'note':'tag=1 frames only; fixed variant-global optimum'})
        drows.append(tr)
    write_csv(OUT/'depthtrack_attributes.csv',drows)
    lines=['# Batch-1 属性分析（探索性）','', '全部结果为单个训练 seed、未做多重比较校正，仅作描述性分析；不据此作显著性结论。','', '## VisEvent','', f"整体 SR 验收通过："+', '.join(f'{v} {100*overall[v]:.2f}' for v in V)+'。`visevent_attributes.csv` 含 17 个官方属性的 SR，以及 4 个主比较的配对均差、t、胜/负和 n。', '', '其中昼夜对应 LI；遮挡对应 FOC / POC。序列名中只有 59 条含 `outdoor`，其余只是未标注，不能将其作为可靠室内标签。', '', '## DepthTrack','', '`depthtrack_attributes.csv` 给出室内(41)/野外(9)，以及所有逐帧 tag 的覆盖序列数、帧数、各变体 F 与配对描述统计。昼夜对应 `dark-scene`；遮挡对应 `full-occlusion` / `partial-occlusion`。tag 的 F 只使用 tag=1 的帧、各变体自己的全局最优阈值。', '', '## LasHeR','', '已在 `D:\\cvresearch\\data\\lasher` 与 `D:\\cvresearch\\tools` 搜索属性标注（attr/att/AttriSeqs/xlsx/csv）。只找到 LasHeR toolkit 的 `annos/att` 边界框文件及无关文件名匹配，未找到官方属性标注；按任务约束停止，未下载任何外部数据，因此没有 LasHeR 属性 CSV。']
    (OUT/'attributes.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__': main()
