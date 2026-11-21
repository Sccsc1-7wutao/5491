"""Paired VisEvent comparison; scoring is delegated to visevent_eval."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import t
from visevent_eval import sequence_scores
ap=argparse.ArgumentParser(); ap.add_argument('--toolkit',required=True); ap.add_argument('--a',required=True); ap.add_argument('--b',required=True); a=ap.parse_args()
root=Path(a.toolkit); seq=[x.strip() for x in (root/'sequence_evaluation_config/VisEvent_testing_subset.txt').read_text().splitlines() if x.strip()]
aa=sequence_scores(root/'annos',Path(a.a),seq); bb=sequence_scores(root/'annos',Path(a.b),seq)
for name,ix in [('PR',0),('SR',1)]:
 d=np.array([bb[s][ix]-aa[s][ix] for s in seq]); n=len(d); mean=d.mean(); se=d.std(ddof=1)/np.sqrt(n); crit=t.ppf(.975,n-1); ci=(mean-crit*se,mean+crit*se); tv=mean/se; p=2*t.sf(abs(tv),n-1); exact=np.sum(d==0); top=np.argsort(-np.abs(d))[:5]
 print(f'{name}: mean_diff={mean*100:.6f} pp CI95=[{ci[0]*100:.6f},{ci[1]*100:.6f}] pp t={tv:.6f} p={p:.6g} exact={exact}/{n}'); print('TOP5',[(seq[i],d[i]*100) for i in top])
