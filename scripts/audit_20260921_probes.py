"""Minimal diagnostic probes. Does not edit training code or old artifacts."""
from pathlib import Path
import sys, json, contextlib, io, copy
from collections import OrderedDict
import torch

ROOT=Path('D:/cvresearch/ViPT')
sys.path.insert(0,str(ROOT))
OUT=ROOT/'output/eval_results/independent_audit_20260921'
from lib.config.vipt.config import cfg, update_config_from_file
from lib.federated.client import FederatedClient, seed_everything
from lib.federated.server import _Settings
from lib.federated.client_data import load_split
from lib.federated.aggregator import Aggregator, ClientUpdate, make_initial_state
from lib.federated.param_groups import build_param_groups, AUX_SPECIFIC
from lib.federated.warmup import WarmupScheduler
from lib.models.vipt.ostrack_prompt import build_viptrack
from lib.train.base_functions import update_settings
from lib.train.actors import ViPTActor

update_config_from_file(str(ROOT/'experiments/vipt/repro_rgbt.yaml'))
cfg.TRAIN.BATCH_SIZE=2
cfg.TRAIN.NUM_WORKER=0
settings=_Settings('cuda')
update_settings(settings,cfg)
seed_everything(42)
with contextlib.redirect_stdout(io.StringIO()):
    net=build_viptrack(cfg,training=True).cuda()
groups=build_param_groups(net.state_dict().keys())
state=make_initial_state(net.state_dict(),groups,aux_types=('thermal','depth','event'))
_,clients=load_split(str(ROOT/'configs/fed/client_splits/lasher_depthtrack_visevent_seed42.json'))
client=FederatedClient(clients['rgbd_003'],cfg,settings,base_seed=42)
buffers0={k:v.clone() for k,v in net.named_buffers()}
initial={k:v.clone() for k,v in net.state_dict().items()}
bn=[(k,m) for k,m in net.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
plan=WarmupScheduler.no_warmup(120).resolve(25)
_,stats=client.local_train(net,state.view_for('depth'),plan,25,local_steps=1,device='cuda')
changed=[k for k,v in net.named_buffers() if not torch.equal(buffers0[k],v)]
result={'settings_fix_bn':settings.fix_bn,'bn_layers':len(bn),'bn_training_after_local_train':sum(m.training for _,m in bn),'changed_buffers':changed,'changed_buffers_count':len(changed),'changed_buffers_all_frozen':all(groups[k]=='frozen' for k in changed),'loss':stats['mean_Loss/total']}

# The same actual one-batch run, applying the official actor's fix immediately
# after net.train(), via a process-local hook (no source modification).
net.load_state_dict(initial)
original_train=net.train
def official_train(mode=True):
    original_train(mode)
    if mode and settings.fix_bn:
        net.box_head.apply(lambda m: m.eval() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) else None)
    return net
net.train=official_train
_,fixed_stats=client.local_train(net,state.view_for('depth'),plan,25,local_steps=1,device='cuda')
result['fixed_changed_buffers']=[k for k,v in net.named_buffers() if not torch.equal(buffers0[k],v)]
result['fixed_loss_same_seed_same_batch']=fixed_stats['mean_Loss/total']
net.train=original_train

# Does an RGB-only client's supposedly absent auxiliary path actually stay fixed?
net.load_state_dict(initial)
rgb_client=FederatedClient(clients['rgbd_000'],cfg,settings,base_seed=42)
rgb_update,rgb_stats=rgb_client.local_train(net,state.view_for('none'),plan,25,local_steps=2,device='cuda')
rgb_aux_changed=[k for k,v in rgb_update.state.items() if groups[k]==AUX_SPECIFIC and not torch.equal(v,initial[k])]
result['rgb_only']={'aux_changed_count':len(rgb_aux_changed),'aux_changed_keys':rgb_aux_changed,'all_aux_upload_masks_false':all(not rgb_update.param_mask[k] for k in rgb_update.state if groups[k]==AUX_SPECIFIC),'loss':rgb_stats['mean_Loss/total']}

# alpha=0 endpoint: same valid updates, but NO_AUX dispatch differs.
key=next(k for k,g in groups.items() if g==AUX_SPECIFIC)
tiny_groups={key:AUX_SPECIFIC}
tiny=make_initial_state({key:torch.tensor([0.])},tiny_groups,aux_types=('thermal','depth'))
updates=[ClientUpdate('t',1,'thermal',{key:torch.tensor([2.])}),ClientUpdate('d',1,'depth',{key:torch.tensor([4.])})]
pooled,_=Aggregator(tiny_groups,modality_aware=False).aggregate(tiny,updates)
alpha0,_=Aggregator(tiny_groups,modality_aware=True,modality_alpha=0.).aggregate(tiny,updates)
result['alpha_zero_endpoint']={'fedavg_thermal':pooled.view_for('thermal')[key].item(),'alpha0_thermal':alpha0.view_for('thermal')[key].item(),'fedavg_rgb_only':pooled.view_for('none')[key].item(),'alpha0_rgb_only':alpha0.view_for('none')[key].item()}

print(json.dumps(result,indent=2))
(OUT/'runtime_probes.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
