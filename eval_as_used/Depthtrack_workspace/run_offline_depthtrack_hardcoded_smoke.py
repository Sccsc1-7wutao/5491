"""One-off equivalence probe: original hard-coded s50 checkpoint selection."""
import sys, time
import argparse
import torch
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1]))
from lib.test.evaluation import Tracker
from lib.train.dataset.depth_utils import get_rgbd_frame

ROOT = Path(r'D:\cvresearch\data\depthtrack\test')

def run(output_dir, sequence):
    torch.set_num_threads(1)
    yaml_name = 's50_modality_depth'
    tr = Tracker('vipt', yaml_name, 'vot22', None); params = tr.get_parameters()
    params.checkpoint = r'D:\cvresearch\ViPT\output\checkpoints\train\vipt\s50_modality_depth\ViPTrack_ep0120.pth.tar'
    print('[params] checkpoint ->', params.checkpoint, flush=True)
    out_root = Path(output_dir); out_root.mkdir(parents=True, exist_ok=True)
    params.visualization = False; params.debug = 0; tracker = tr.create_tracker(params)
    d = ROOT / sequence
    if not d.is_dir(): raise ValueError(f'unknown sequence {sequence!r}')
    gt=np.loadtxt(d/'groundtruth.txt',delimiter=',',ndmin=2); names=sorted((d/'color').glob('*.jpg')); dep=sorted((d/'depth').glob('*.png')); od=out_root/yaml_name/d.name; od.mkdir(parents=True,exist_ok=True)
    boxes=['1']; scores=[]; times=[]
    first=get_rgbd_frame(str(names[0]),str(dep[0]),dtype='rgbcolormap',depth_clip=True); tracker.initialize(first,{'init_bbox':gt[0].tolist()})
    for color,depth in zip(names[1:],dep[1:]):
        t0=time.time(); image=get_rgbd_frame(str(color),str(depth),dtype='rgbcolormap',depth_clip=True); out=tracker.track(image)
        boxes.append(','.join(str(float(v)) for v in out['target_bbox'])); scores.append(float(out['best_score'])); times.append(time.time()-t0)
    (od/f'{d.name}_001.txt').write_text('\n'.join(boxes)+'\n',encoding='utf-8'); (od/f'{d.name}_001_confidence.value').write_text('\n'.join(str(v) for v in scores)+'\n',encoding='utf-8'); (od/f'{d.name}_001_time.value').write_text('\n'.join(str(v) for v in times)+'\n',encoding='utf-8')
    print(yaml_name,d.name,flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--sequence', required=True)
    args = parser.parse_args()
    run(args.output_dir, args.sequence)
