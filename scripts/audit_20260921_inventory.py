"""Read-only evidence collection for the independent 2026-09-21 audit."""
from pathlib import Path
import json, re, hashlib
from collections import Counter

ROOT = Path('D:/cvresearch')
OUT = ROOT / 'ViPT/output/eval_results/independent_audit_20260921'
OUT.mkdir(exist_ok=True)

def text_blocks(x):
    msg = x.get('message', {})
    content = msg.get('content', []) if isinstance(msg, dict) else []
    if isinstance(content, str):
        return content
    return '\n'.join(b.get('text', '') for b in content if isinstance(b, dict) and b.get('type') == 'text')

summaries = []
pattern = re.compile(r'FIX_BN|BatchNorm|loss_weights|consistency|calib|warm.up|噪声|预算|欠训练|创新|α|alpha|审计|结论|截稿|ICMR|accum|56.?250|158|192|0\.976|调度|local|FedProx|SCAFFOLD', re.I)
for p in sorted(Path('C:/Users/A/.claude/projects/D--Downloads-cv--').glob('*.jsonl')):
    lines = p.read_text(encoding='utf-8').splitlines()
    entries, hist, matches = [], [], []
    for line in lines:
        try:
            x = json.loads(line)
        except json.JSONDecodeError:
            continue
        typ = x.get('type', '')
        t = text_blocks(x)
        if typ in ('user', 'assistant') and t:
            entries.append((x.get('timestamp',''), typ, t))
            hist.append(f"[{x.get('timestamp','')}] {typ}\n{t}\n")
            if pattern.search(t):
                matches.append(f"[{x.get('timestamp','')}] {typ}\n{t}\n")
    (OUT / (p.stem + '_conversation.txt')).write_text('\n'.join(hist), encoding='utf-8')
    (OUT / (p.stem + '_relevant.txt')).write_text('\n'.join(matches), encoding='utf-8')
    s = {'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(), 'messages':len(entries),'first':entries[0][0] if entries else None, 'last':entries[-1][0] if entries else None, 'users':[t[:350] for _,role,t in entries if role=='user' and not t.startswith('<')][-18:]}
    summaries.append(s)
print(json.dumps(summaries, ensure_ascii=False, indent=2))
(OUT/'conversation_inventory.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')

split=json.loads((ROOT/'ViPT/configs/fed/client_splits/lasher_depthtrack_visevent_seed42.json').read_text(encoding='utf-8'))
print('SPLIT',json.dumps({k:v for k,v in split.items() if k!='clients'},ensure_ascii=False)[:2000])
for c in split['clients']:
    print('CLIENT',json.dumps({k:(len(v) if k=='sequence_ids' else v) for k,v in c.items()},ensure_ascii=False))
for p in sorted((ROOT/'ViPT/output/fed').glob('*/*/round_metrics.jsonl')):
    rows=[json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
    print('RUN',p.relative_to(ROOT),len(rows),'rounds',[rows[0]['round'],rows[-1]['round']], 'seconds',sum(r['seconds'] for r in rows),'clients',Counter(len(r['clients']) for r in rows),'steps',Counter(r['local_steps'] for r in rows),'phases',Counter(r['phase'] for r in rows),'first',json.dumps(rows[0])[:600])
