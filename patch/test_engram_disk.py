import json, struct, tempfile, time, os, torch
d = tempfile.mkdtemp()
R, dim, sb = 200000, 256, 8
torch.manual_seed(0)
w = torch.randint(0, 256, (R, dim), dtype=torch.uint8)
s = torch.randint(118, 136, (R, sb), dtype=torch.uint8)
hdr = {"layers.1.engram.embed.weight": {"dtype": "F8_E4M3", "shape": [R, dim], "data_offsets": [0, R*dim]},
       "layers.1.engram.embed.scale": {"dtype": "F8_E8M0", "shape": [R, sb], "data_offsets": [R*dim, R*dim + R*sb]},
       "__metadata__": {"format": "pt"}}
hb = json.dumps(hdr).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
with open(d + "/model-00001-of-00001.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(hb))); f.write(hb); f.write(w.numpy().tobytes()); f.write(s.numpy().tobytes())
json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in hdr if k != "__metadata__"}}, open(d + "/model.safetensors.index.json", "w"))
from vllm.models.deepseek_v4_1.common import engram as E
print("module:", E.__file__, "| has DiskEngramTable:", hasattr(E, "DiskEngramTable"), "| env:", E._DSV41_ENGRAM_DISK)
t = E.DiskEngramTable(d, 1, dim, 32)
n = 5000
idx = torch.randint(0, R, (n,), dtype=torch.int64); owned = torch.ones(n, dtype=torch.bool); owned[::7] = False
out = t.gather_dequant(idx, owned)
vals = w[idx].view(torch.float8_e4m3fn).to(torch.float32).view(-1, sb, 32)
scale = (s[idx].to(torch.int32) << 23).view(torch.float32)
ref = (vals * scale[:, :, None]).reshape(-1, dim); ref[~owned] = 0; ref = ref.to(torch.bfloat16)
ok = torch.allclose(out.float(), ref.float(), equal_nan=True) and out.shape == ref.shape
print("correctness:", "PASS" if ok else "FAIL", "| shape", tuple(out.shape), "| nan rows", torch.isnan(out.float()).any(dim=1).sum().item())
# cross-check against safetensors' own reader for a few rows
from safetensors import safe_open
with safe_open(d + "/model-00001-of-00001.safetensors", framework="pt") as f:
    W = f.get_tensor("layers.1.engram.embed.weight"); S = f.get_tensor("layers.1.engram.embed.scale")
r = int(idx[1]); print("safetensors row match:", torch.equal(W[r].view(torch.uint8), w[r]), "scale:", torch.equal(S[r].view(torch.uint8), s[r]))
for n2 in (48, 4096, 49152):
    idx2 = torch.randint(0, R, (n2,), dtype=torch.int64); own2 = torch.ones(n2, dtype=torch.bool)
    t0 = time.time(); o2 = t.gather_dequant(idx2, own2); dt = time.time() - t0
    print(f"{n2:6d} rows: {dt*1000:8.1f} ms  ({n2/dt:,.0f} rows/s)")
