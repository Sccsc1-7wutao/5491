import json
from pathlib import Path
import numpy as np

ROOT = Path(r'D:\cvresearch\data\depthtrack\test')
RUNS = {
    'fedavg': Path(r'D:\cvresearch\ViPT\Depthtrack_workspace_legacy_fedavg\results\legacy_fedavg\rgbd-unsupervised'),
    'modality': Path(r'D:\cvresearch\ViPT\Depthtrack_workspace_legacy_modality\results\legacy_modality\rgbd-unsupervised'),
}

def overlap(a, b):
    x = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    y = max(0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    inter = x*y
    return inter/(a[2]*a[3]+b[2]*b[3]-inter+1e-12)

def read_boxes(p):
    return np.array([[float(x) for x in s.split(',')] for s in p.read_text().splitlines() if ',' in s])

def load(run):
    out = {}
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir(): continue
        gt = np.loadtxt(d/'groundtruth.txt', delimiter=',', ndmin=2)
        bp = next(RUNS[run].glob(d.name+'/*_001.txt'))
        cp = next(RUNS[run].glob(d.name+'/*_confidence.value'))
        boxes = read_boxes(bp)
        conf = np.r_[0.0, np.loadtxt(cp, ndmin=1)]
        n = min(len(gt), len(boxes), len(conf))
        valid = np.isfinite(conf[:n]) & np.isfinite(boxes[:n]).all(axis=1) & np.isfinite(gt[:n]).all(axis=1)
        gt = gt[:n][valid]; boxes = boxes[:n][valid]; conf = conf[:n][valid]
        n = len(conf)
        ovs = np.array([overlap(boxes[i], gt[i]) for i in range(n)])
        out[d.name] = (ovs, conf[:n])
    return out

def thresholds(data, resolution=100):
    scores = sorted(np.concatenate([v[1] for v in data.values()]).tolist(), reverse=True)
    if len(scores) > resolution-2:
        delta = len(scores)//(resolution-2)
        idxs = np.round(np.linspace(delta, len(scores)-delta, num=resolution-2)).astype(int)
        scores = [scores[i] for i in idxs]
    return np.array([np.inf] + scores + [-np.inf])

def curve(data, names, th):
    pr=[]; re=[]
    for t in th:
        ps=[]; rs=[]
        for name in names:
            ov, sc = data[name]; mask=sc >= t
            if mask.sum()==0: ps.append(1.0); rs.append(0.0)
            else: ps.append(float(ov[mask].mean())); rs.append(float(ov[mask].sum()/len(ov)))
        pr.append(float(np.mean(ps))); re.append(float(np.mean(rs)))
    f=2*np.array(pr)*np.array(re)/(np.array(pr)+np.array(re)+1e-12)
    return float(np.max(f)), float(pr[np.argmax(f)]), float(re[np.argmax(f)])

def main():
    data={k:load(k) for k in RUNS}; names=sorted(set(data['fedavg']) & set(data['modality']))
    th={k:thresholds(data[k]) for k in data}
    point={k:curve(data[k], names, th[k]) for k in data}
    per=[(curve(data['modality'], [n], th['modality']), curve(data['fedavg'], [n], th['fedavg'])) for n in names]
    seqdiff=np.array([x[0][0]-x[1][0] for x in per])
    prdiff=np.array([x[0][1]-x[1][1] for x in per]); rediff=np.array([x[0][2]-x[1][2] for x in per])
    rng=np.random.default_rng(20260927)
    draws=rng.choice(seqdiff, size=(20000,len(seqdiff)), replace=True)
    diffs=np.nanmean(draws, axis=1)
    result={'n_sequences':len(names),'point_fedavg':point['fedavg'],'point_modality':point['modality'],'diff_modality_minus_fedavg':{'F':float(point['modality'][0]-point['fedavg'][0]),'Pr':float(point['modality'][1]-point['fedavg'][1]),'Re':float(point['modality'][2]-point['fedavg'][2])},'bootstrap_seed':20260927,'bootstrap_draws':20000,'bootstrap_ci95':{'F':[float(x) for x in np.quantile(diffs,[.025,.975])],'Pr':[float(x) for x in np.quantile(rng.choice(prdiff,size=(20000,len(prdiff)),replace=True).mean(1),[.025,.975])],'Re':[float(x) for x in np.quantile(rng.choice(rediff,size=(20000,len(rediff)),replace=True).mean(1),[.025,.975])]}}
    print(json.dumps(result, indent=2))
    Path(r'D:\cvresearch\ViPT\output\eval_results\overnight\depthtrack_vot_bootstrap.json').write_text(json.dumps(result,indent=2), encoding='utf-8')
if __name__=='__main__': main()
