"""AMORTIZED-HEALING prerequisite: is the heal delta STRUCTURED?

Surgery (the winning recipe, scaled to 0.5B) -> short heal -> per-matrix SVD of (healed - surgical).
If the deltas are low-rank (top-r explains most energy at r << dim) and consistent across layers,
a learned operator can plausibly PREDICT them (amortized healing) — and an immediate practical win
exists (LoRA-heal: train only low-rank factors). If dense/isotropic, weight-level amortization is dead.

Run (RTX 3070, ~2-3h):  uv run python scripts/heal_delta.py --heal_tokens 3000000
"""
import argparse, copy, sys, torch
import torch.nn.functional as F
sys.path.insert(0, ".")

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
p.add_argument("--heal_tokens", type=int, default=3_000_000)
p.add_argument("--seqlen", type=int, default=256)
p.add_argument("--lr", type=float, default=3e-5)
args = p.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.arch import get_arch, prune_hidden_width
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained(args.model)
teacher = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device).eval()
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
arch = get_arch(model)

# --- surgery: the winning recipe scaled to 0.5B (896 -> 744 width; keep attn/FFN full; no depth for cleanliness) ---
calib_txt = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"][:200])
calib = tok(calib_txt, return_tensors="pt").input_ids[:, :2048].to(device)
prune_hidden_width(model, arch, 744, calib, device)
model.config.hidden_size = 744
pre = {n: q.detach().clone() for n, q in model.named_parameters()}          # SURGICAL state
print(f"[surgery] width 896->744; snapshot {len(pre)} tensors")

# --- short heal: CE + logit-KD from the parent (the recipe's heal, miniaturized) ---
ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
buf, ntok = [], 0
for ex in ds:
    ids = tok(ex["text"], return_tensors="pt").input_ids
    buf.append(ids); ntok += ids.size(1)
    if ntok >= args.heal_tokens: break
train = torch.cat(buf, 1)
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr)
S = args.seqlen; n_seq = train.size(1) // S
steps = 0
model.train()
for i in range(n_seq):
    inp = train[:, i * S:(i + 1) * S].to(device)
    if inp.size(1) < S: break
    out = model(inp, labels=inp)
    with torch.no_grad():
        tl = teacher(inp).logits
    kd = F.kl_div(F.log_softmax(out.logits.float().reshape(-1, out.logits.size(-1)) / 2, -1),
                  F.softmax(tl.float().reshape(-1, tl.size(-1)) / 2, -1), reduction="batchmean") * 4
    loss = 0.5 * out.loss + 0.5 * kd
    opt.zero_grad(); loss.backward(); opt.step()
    steps += 1
    if steps % 200 == 0: print(f"  step {steps}/{n_seq} loss {loss.item():.3f}")
print(f"[heal] {steps} steps ({steps*S/1e6:.1f}M tokens)")

# --- the question: structure of (healed - surgical) per weight matrix ---
print("\n=== HEAL-DELTA STRUCTURE (per-matrix: ||delta||/||W||, and rank needed for 90% of delta energy) ===")
rows = []
for n, q in model.named_parameters():
    if q.dim() != 2 or min(q.shape) < 64: continue
    d = (q.detach().float() - pre[n].float())
    rel = (d.norm() / pre[n].float().norm()).item()
    try:
        sv = torch.linalg.svdvals(d)
    except Exception:
        continue
    e = sv ** 2; cum = torch.cumsum(e, 0) / e.sum().clamp_min(1e-12)
    r90 = int((cum < 0.90).sum().item()) + 1
    r50 = int((cum < 0.50).sum().item()) + 1
    frac90 = r90 / min(d.shape)
    kind = ("attn" if "attn" in n else "mlp" if "mlp" in n else "embed" if "embed" in n or "lm_head" in n else "other")
    rows.append((n, kind, rel, r50, r90, frac90, min(d.shape)))
rows.sort(key=lambda r: -r[2])
print(f"{'matrix':52s} {'kind':6s} {'|d|/|W|':>8s} {'r50':>5s} {'r90':>5s} {'r90/dim':>8s}")
for n, kind, rel, r50, r90, frac90, dim in rows[:18]:
    print(f"{n[:52]:52s} {kind:6s} {rel:8.4f} {r50:5d} {r90:5d} {frac90:8.2f}")
import statistics
f90s = [r[5] for r in rows]; rels = [r[2] for r in rows]
print(f"\nSUMMARY: {len(rows)} matrices | median |d|/|W| = {statistics.median(rels):.4f} | "
      f"median r90/dim = {statistics.median(f90s):.2f}")
print("VERDICT: " + ("STRUCTURED (low-rank) -> amortized healing / LoRA-heal plausible"
                     if statistics.median(f90s) < 0.25 else
                     "DENSE (r90 ~ full dim) -> weight-level amortized healing implausible; delta is diffuse"))
