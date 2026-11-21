"""Recompute metric summaries, paired uncertainty, and budget accounting."""
from pathlib import Path
import json, hashlib, math
import numpy as np
from scipy import stats
ROOT=Path('D:/cvresearch/ViPT')
DIR=ROOT/'output/eval_results'
OUT=DIR/'independent_audit_20260921'
out={'metrics':{},'paired':{},'prediction_integrity':{},'log_integrity':{}}
curves={}
for p in sorted(DIR.glob('*_curves.npz')):
    a=np.load(p,allow_pickle=True)
    curves[p.stem]={'AUC':a['curves_ovl'].mean(axis=1)*100,'PR':a['curves_err'][:,20]*100,'names':a['names'].tolist()}
    out['metrics'][p.stem]={k:float(v.mean()) for k,v in curves[p.stem].items() if k!='names'}
    out['metrics'][p.stem]['n']=len(a['names'])
    out['metrics'][p.stem]['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
for a,b in [('long_fedavg','long_modality'),('long_fedavg','long_ours'),('long_modality','long_ours'),('pilot_fedavg','pilot_modality'),('pilot_modality','pilot_ours')]:
    ca,cb=curves[a+'_curves'],curves[b+'_curves']
    assert ca['names']==cb['names']
    result={}
    for metric in ('AUC','PR'):
        d=cb[metric]-ca[metric]
        se=d.std(ddof=1)/np.sqrt(len(d))
        t=d.mean()/se
        rng=np.random.default_rng(20260921)
        boot=d[rng.integers(0,len(d),(20000,len(d)))].mean(axis=1)
        result[metric]={'diff_pp':float(d.mean()),'se_pp':float(se),'t':float(t),'p_two_sided':float(2*stats.t.sf(abs(t),len(d)-1)),'ci95_t':(d.mean()+np.array([-1,1])*stats.t.ppf(.975,len(d)-1)*se).tolist(),'ci95_bootstrap_sequence':np.percentile(boot,[2.5,97.5]).tolist(),'wins_losses_ties':[int((d>0).sum()),int((d<0).sum()),int((d==0).sum())]}
    out['paired'][b+' minus '+a]=result
for run in ['pilot_fedavg','pilot_modality','pilot_ours','long_fedavg','long_modality','long_ours','traj_fedavg_r40','traj_fedavg_r80','traj_modality_r40','traj_modality_r80']:
    pred=ROOT/'RGBT_workspace/results/LasHeR'/run
    files=list(pred.glob('*.txt'))
    bad=[];frames=0
    for p in files:
        x=np.loadtxt(p)
        if x.ndim!=2 or x.shape[1]!=4 or not np.isfinite(x).all(): bad.append(p.name)
        frames+=len(x)
    out['prediction_integrity'][run]={'files':len(files),'frames':frames,'bad':bad}
    log=DIR/(run+'_eval.log')
    if not log.exists(): log=ROOT/'output'/('eval_'+run+'.log')
    if log.exists():
        lines=log.read_text(encoding='utf-8',errors='replace').splitlines()
        out['log_integrity'][run]={'checkpoint_paths':sorted(set(l for l in lines if '[params] checkpoint ->' in l)),'skips':sum(l.startswith('-1 ') for l in lines),'errors':sum('Traceback' in l or 'RuntimeError' in l for l in lines),'finished':sum('Totally cost' in l for l in lines)}
out['budgets']={'central_samples':60000*60,'central_optimizer_steps_tail_flush':60*math.ceil((60000//32)/2),'old_all_clients_samples':120*12*15*32,'new_all_clients_samples':120*12*50*32,'old_each_client_updates':120*math.ceil(15/2),'new_each_client_updates':120*50,'sample_multiplier':50/15,'update_multiplier':50/math.ceil(15/2),'old_thermal_present_expected_samples':120*3*15*32*.9,'new_thermal_present_expected_samples':120*3*50*32*.9}
rounds=np.array([20,40,80,120])
traj={}
for metric in ('AUC','PR'):
    d=[]
    for r in rounds:
        names=('pilot_fedavg','pilot_modality') if r==20 else ('long_fedavg','long_modality') if r==120 else (f'traj_fedavg_r{r}',f'traj_modality_r{r}')
        d.append(out['metrics'][names[1]+'_curves'][metric]-out['metrics'][names[0]+'_curves'][metric])
    fit=stats.linregress(rounds,d)
    traj[metric]={'delta_pp':d,'gap_closure_per_round':(np.diff(d)/np.diff(rounds)).tolist(),'slope':fit.slope,'r_squared':fit.rvalue**2,'naive_zero_crossing':-fit.intercept/fit.slope}
out['trajectory']=traj
(OUT/'metrics_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ('log_integrity',)},indent=2))
