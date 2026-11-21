"""Contract, lineage, and split checks for the independent audit."""
from pathlib import Path
import sys, json, pickle, hashlib
import torch
import numpy as np
ROOT=Path('D:/cvresearch/ViPT');sys.path.insert(0,str(ROOT))
OUT=ROOT/'output/eval_results/independent_audit_20260921'
from lib.federated.fed_wrapper import FedViPTWrapper
class Capture(torch.nn.Module):
    def forward(self,template,search,**kwargs):
        self.aux=template[:,3:].clone()
        return {'pred_boxes':torch.zeros(template.shape[0],1,4)}
net=Capture();wrapper=FedViPTWrapper(net)
rgb=torch.ones(2,3,8,8);aux=torch.ones_like(rgb)*7
out=wrapper(rgb,rgb,aux,aux,aux_type='thermal',modality_mask={'rgb':1,'thermal':0,'depth':0,'event':0})
result={'wrapper_explicit_mask_zero':{'forward_aux_mean':net.aux.mean().item(),'reported_aux_present':out['aux_present'],'reported_mask':out['modality_mask']}}
for v in ('fedavg','modality'):
    p=ROOT/f'output/fed/pilot20/{v}/fed_state_r0020.pt'
    q=ROOT/f'output/fed/long120/{v}/fed_state_r0020.pt'
    a=torch.load(p,map_location='cpu',weights_only=False)['state']
    b=torch.load(q,map_location='cpu',weights_only=False)['state']
    result[v+'_pilot_vs_long_r20']={'keys':len(a),'equal':sum(torch.equal(a[k],b[k]) for k in a),'max_diff':max((a[k]-b[k]).abs().max().item() for k in a)}
for v in ('fedavg','modality','ours'):
    a=np.load(ROOT/f'output/eval_results/long_{v}_curves.npz',allow_pickle=True)
    b=np.load(OUT/f'rescored_long_{v}.npz',allow_pickle=True)
    result[v+'_rescore']={k:bool(np.array_equal(a[k],b[k])) for k in a.files}
split=json.loads((ROOT/'configs/fed/client_splits/lasher_depthtrack_visevent_seed42.json').read_text(encoding='utf-8'))
for ds,folder in [('lasher','D:/cvresearch/data/lasher/testingset'),('depthtrack','D:/cvresearch/data/depthtrack/test'),('visevent','D:/cvresearch/data/visevent/test_subset')]:
    train={s for c in split['clients'] if c['dataset']==ds for s in c['sequence_ids']}
    test={p.name for p in Path(folder).iterdir() if p.is_dir()}
    result[ds+'_name_overlap']={'train_sequences':len(train),'test_directories':len(test),'overlap':sorted(train&test)}
val=set((ROOT/'lib/train/data_specs/lasher_val.txt').read_text().splitlines())
train={s for c in split['clients'] if c['dataset']=='lasher' for s in c['sequence_ids']}
result['lasher_val']={'n':len(val),'train_overlap':len(val&train),'test_overlap':len(val&{p.name for p in Path('D:/cvresearch/data/lasher/testingset').iterdir() if p.is_dir()})}
from lib.config.vipt.config import cfg, update_config_from_file
from lib.federated.client import FederatedClient
from lib.federated.server import _Settings
from lib.train.base_functions import update_settings
update_config_from_file(str(ROOT/'experiments/vipt/repro_rgbt.yaml'))
cfg.TRAIN.NUM_WORKER=1
settings=_Settings('cpu');update_settings(settings,cfg)
c=FederatedClient(next(c for c in split['clients'] if c['client_id']=='rgbd_003'),cfg,settings)
loader=c._build_loader(1,42)
try:
    pickle.dumps(loader.worker_init_fn)
    result['windows_worker_init_pickle']='ok'
except Exception as exc:
    result['windows_worker_init_pickle']=str(exc)
result['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'lib/federated').glob('*.py')}
(OUT/'contracts_and_lineage.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
