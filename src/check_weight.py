# src/check_weights.py
import os, fnmatch
p = "models/DistilBERT"
if not os.path.isdir(p):
    print("Directory not found:", p)
    raise SystemExit(1)

matches = []
for root, dirs, files in os.walk(p):
    for pat in ("*.bin","*.pt","*.safetensors","*.h5","*.ckpt"):
        for f in fnmatch.filter(files, pat):
            matches.append(os.path.join(root, f))
if not matches:
    print("No model weight files found in", p)
else:
    print("Found weight files:")
    for m in matches:
        print(" ", m)
