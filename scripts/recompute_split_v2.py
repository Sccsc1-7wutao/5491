"""Recompute split weights without changing sequence assignment."""
import hashlib, json, os, sys
sys.path.insert(0, r"D:\cvresearch\ViPT")
from lib.federated.split_builder import DATASETS, count_frames

src = sys.argv[1]
dst = sys.argv[2]
with open(src, encoding="utf-8") as f: doc = json.load(f)
old_fp = doc["fingerprint"]
for c in doc["clients"]:
    cfg = DATASETS[c["dataset"]]
    c["num_samples"] = sum(count_frames(os.path.join(cfg["root"], s), cfg["anno"])
                            for s in c["sequence_ids"])
    c["train_weight"] = c["num_samples"]
doc["total_samples"] = sum(c["num_samples"] for c in doc["clients"])
payload = [(c["client_id"], c["num_samples"]) for c in doc["clients"]]
doc["weights_fingerprint"] = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()[:16]
doc["data_version"] = "common-modal-frame-prefix-v2"
doc["note"] = "num_samples uses min(annotation, RGB files, auxiliary files); sequence assignment unchanged"
assert doc["fingerprint"] == old_fp == "bf493c676d666971"
assert doc["total_samples"] == 881097
changed = [c["client_id"] for c in doc["clients"] if c["num_samples"] != next(x["num_samples"] for x in json.load(open(src, encoding="utf-8"))["clients"] if x["client_id"] == c["client_id"])]
assert changed == ["rgbd_001"], changed
old = json.load(open(src, encoding="utf-8"))
assert old["clients"][[c["client_id"] for c in old["clients"]].index("rgbd_001")]["num_samples"] - next(c["num_samples"] for c in doc["clients"] if c["client_id"] == "rgbd_001") == 39
with open(dst, "w", encoding="utf-8", newline="\n") as f: json.dump(doc, f, ensure_ascii=True, indent=2); f.write("\n")
print(dst, doc["total_samples"], changed, doc["weights_fingerprint"])
