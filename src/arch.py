"""Architecture adapters for decoder-LM surgery (GPT-2, GPT-NeoX/Pythia, Llama/Mistral).

The compression logic (remove a span, splice one warm-started block, prune kept-block
FFN width) is identical across families — only the *accessors* differ (where the layer
list lives, the block class, the FFN linear names, the final norm / head). This module
abstracts those so `run_generative.py` is arch-agnostic.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Arch:
    family: str
    get_layers: Callable          # model -> ModuleList of decoder blocks
    set_layers: Callable          # (model, ModuleList) -> None
    n_layer_attr: str             # config field holding the layer count
    make_block: Callable          # (model, layer_idx) -> fresh decoder block
    prune_ffn: Callable           # (block, keep_k) -> bool  (prune FFN intermediate width)
    ffn_width: Callable           # block -> current FFN intermediate width
    ffn_down: Callable            # block -> the down-projection Linear (ffn->hidden); its INPUT is the intermediate
    trainable_extra: Callable     # model -> list of (final norm, head) modules to heal


# ---------------------------------------------------------------- low-rank SVD factorization (RANK axis)
class LowRankLinear(nn.Module):
    """y = up(down(x)); weight W (out,in) ≈ A·B with A:(out,r), B:(r,in). PRESERVES the parent's learned
    transformation in its top-r principal directions instead of DELETING it. Params: r(out+in) vs out·in."""
    def __init__(self, A, B, bias=None, Ae=None, Be=None):
        super().__init__()
        out_f, r = A.shape; r2, in_f = B.shape
        self.down = nn.Linear(in_f, r, bias=False).to(B.dtype)
        self.up = nn.Linear(r, out_f, bias=bias is not None).to(A.dtype)
        with torch.no_grad():
            self.down.weight.copy_(B); self.up.weight.copy_(A)
            if bias is not None:
                self.up.bias.copy_(bias)
        self.eora = Ae is not None
        if self.eora:                                     # EoRA error-compensation path (trainable repair)
            re = Ae.shape[1]
            self.down_e = nn.Linear(in_f, re, bias=False).to(Be.dtype)
            self.up_e = nn.Linear(re, out_f, bias=False).to(Ae.dtype)
            with torch.no_grad():
                self.down_e.weight.copy_(Be); self.up_e.weight.copy_(Ae)

    def forward(self, x):
        y = self.up(self.down(x))
        if self.eora:
            y = y + self.up_e(self.down_e(x))
        return y


@torch.no_grad()
def asvd_factorize(linear, act_scale, rank, eora_rank=0, out_scale=None, cov=None):
    """Activation-aware SVD (ASVD-style): scale W's columns by per-input-channel activation magnitude so
    the SVD keeps directions important for REAL DATA, factorize to rank, then undo the scaling. Returns a
    LowRankLinear, or the original linear if low-rank wouldn't save params.
    out_scale (OUTPUT-AWARE / RRR): also row-scale by per-output importance.
    cov (SVD-LLM full whitening): factorize in the input-covariance metric instead of a diagonal scale -> truncation
    is output-error-optimal AND captures the post-RMSNorm-gain input correlations (RMSNorm-aware capture)."""
    W = linear.weight.float()                              # (out, in)
    out_f, in_f = W.shape
    if rank * (out_f + in_f) >= out_f * in_f:             # low-rank not smaller -> keep original
        return linear
    if cov is not None:                                   # SVD-LLM full-covariance whitening (RMSNorm-aware)
        nd = cov.shape[0]; Cf = cov.float()
        L = torch.linalg.cholesky(Cf + (1e-3 * Cf.diagonal().mean().clamp_min(1e-8)) * torch.eye(nd, device=Cf.device))
        U, sig, Vt = torch.linalg.svd(W @ L, full_matrices=False)
        rs = sig[:rank].clamp_min(0).sqrt(); dt = linear.weight.dtype
        A = (U[:, :rank] * rs.unsqueeze(0))
        B = torch.linalg.solve_triangular(L.t(), (rs.unsqueeze(1) * Vt[:rank, :]).t(), upper=True).t()  # (rs Vt) L^-1
        Ae = Be = None
        if eora_rank > 0:
            re = min(eora_rank, sig.shape[0] - rank)
            if re > 0:
                rse = sig[rank:rank + re].clamp_min(0).sqrt()
                Ae = (U[:, rank:rank + re] * rse.unsqueeze(0)).to(dt)
                Be = torch.linalg.solve_triangular(L.t(), (rse.unsqueeze(1) * Vt[rank:rank + re, :]).t(), upper=True).t().to(dt)
        return LowRankLinear(A.to(dt), B.to(dt), linear.bias, Ae, Be)
    S = act_scale.float()                                 # (in,) activation importance per input channel
    med = S[S > 0].median() if (S > 0).any() else torch.tensor(1.0)
    S = S.clamp(med * 1e-2, med * 1e2)                    # RELATIVE clamp -> bounded ratio, no /tiny blowup
    M = W * S.unsqueeze(0)
    T = None
    if out_scale is not None:                             # OUTPUT-AWARE row weighting
        T = out_scale.float(); tmed = T[T > 0].median() if (T > 0).any() else torch.tensor(1.0)
        T = T.clamp(tmed * 1e-2, tmed * 1e2); M = T.unsqueeze(1) * M
    U, sig, Vt = torch.linalg.svd(M, full_matrices=False)
    rs = sig[:rank].clamp_min(0).sqrt()                   # sqrt-split sig across A,B -> balanced magnitudes
    A = (U[:, :rank] * rs.unsqueeze(0))                   # (out, rank)
    if T is not None: A = A / T.unsqueeze(1)              # undo row scaling
    B = (rs.unsqueeze(1) * Vt[:rank, :] / S.unsqueeze(0)) # (rank, in) — undo column scaling
    dt = linear.weight.dtype
    Ae = Be = None
    if eora_rank > 0:                                     # EoRA: error-compensation path = NEXT directions
        re = min(eora_rank, sig.shape[0] - rank)          # init from the compression-error eigenspace
        if re > 0:
            rse = sig[rank:rank + re].clamp_min(0).sqrt()
            Ae = (U[:, rank:rank + re] * rse.unsqueeze(0))
            if T is not None: Ae = Ae / T.unsqueeze(1)
            Ae = Ae.to(dt); Be = (rse.unsqueeze(1) * Vt[rank:rank + re, :] / S.unsqueeze(0)).to(dt)
    return LowRankLinear(A.to(dt), B.to(dt), linear.bias, Ae, Be)


class HybridFFN(nn.Module):
    """CAPTURE-top + COMPRESS-remaining FFN (the residual add-channels recipe ported to the FFN intermediate).
    Keeps the top-K *important* intermediate units EXACT (full gate/up/down), and represents the REMAINING units'
    projections LOW-RANK (activation-aware SVD) instead of dropping them (obs-prune) — so their nonlinear function
    is captured, not deleted. SwiGLU runs on both branches. out = down_top(act(gate_top x)*up_top x) + down_rem(...)."""
    def __init__(self, gate_top, up_top, down_top, gate_rem, up_rem, down_rem, act_fn):
        super().__init__()
        self.gate_top, self.up_top, self.down_top = gate_top, up_top, down_top
        self.gate_rem, self.up_rem, self.down_rem = gate_rem, up_rem, down_rem
        self.act_fn = act_fn

    def forward(self, x):
        out = self.down_top(self.act_fn(self.gate_top(x)) * self.up_top(x))          # exact top-K units
        out = out + self.down_rem(self.act_fn(self.gate_rem(x)) * self.up_rem(x))    # low-rank-captured remaining units
        return out


def _sub_linear(lin, idx, dim):
    """A bias-free Linear from a row (dim=0) or column (dim=1) slice of `lin.weight`."""
    W = lin.weight.data
    Wsub = W[idx, :] if dim == 0 else W[:, idx]
    n = nn.Linear(Wsub.shape[1], Wsub.shape[0], bias=False).to(W.device, W.dtype)
    n.weight.data.copy_(Wsub); return n


class GhostBlock(nn.Module):
    """CAPTURE-not-DROP for DEPTH: replaces a deleted transformer block with a low-rank NONLINEAR mini-FFN
    (x + up(act(down(norm(x))))) — keeps a rank-R nonlinear approximation of the block's residual contribution
    instead of dropping it + linear-splicing. up init=0 (no-op at init, like EoRA); the heal trains it to capture
    the deleted function. Matches the DecoderLayer call signature (returns a 1-tuple)."""
    def __init__(self, norm, down, up, act):
        nn.Module.__init__(self)                                  # skip DecoderLayer.__init__ when subclassed (needs config/layer_idx)
        self.norm = norm; self.down = down; self.up = up; self.act = act
    def forward(self, hidden_states, **kwargs):
        hh = self.up(self.act(self.down(self.norm(hidden_states))))
        return hidden_states + hh                                 # this tf version: layers return a tensor, not a tuple


@torch.no_grad()
def add_depth_ghosts(model, arch, n_ghost, rank, calib_ids, device):
    """Select the n_ghost least-important blocks (lowest input->output change) and replace each with a rank-R
    GhostBlock — depth capture-not-drop. down init from the block's gate input-directions, up=0 (heal-trained)."""
    inner = model.model; layers = arch.get_layers(model); cfg = model.config; h = cfg.hidden_size
    io = {}
    def mk(li):
        def hk(m, inp, out):
            x = inp[0].detach().float(); y = (out[0] if isinstance(out, tuple) else out).detach().float()
            io[li] = (x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]))
        return hk
    hooks = [b.register_forward_hook(mk(i)) for i, b in enumerate(layers)]
    model.eval(); model(calib_ids.to(device), use_cache=False)
    for hk in hooks: hk.remove()
    imp = {i: (1 - torch.nn.functional.cosine_similarity(x, y, dim=-1).mean()).item() for i, (x, y) in io.items()}
    cand = sorted(range(1, len(layers) - 1), key=lambda i: imp[i])[:n_ghost]   # least-important interior blocks
    dt = inner.embed_tokens.weight.dtype
    DLCls = type(layers[cand[0]]) if cand else None                          # real decoder-layer class -> ghost must be isinstance
    GhostCls = type("GhostBlockDL", (GhostBlock, DLCls), {}) if DLCls else GhostBlock  # so output_hidden_states records the ghost
    for i in cand:
        blk = layers[i]
        down = nn.Linear(h, rank, bias=False); up = nn.Linear(rank, h, bias=False)
        Wg = blk.mlp.gate_proj.weight.data.float()                            # [inter, h] -> top input directions
        _, _, Vt = torch.linalg.svd(Wg, full_matrices=False)
        down.weight.data.copy_(Vt[:rank].to(dt)); up.weight.data.zero_()      # read top dirs; up=0 (heal learns)
        layers[i] = GhostCls(blk.post_attention_layernorm, down, up, blk.mlp.act_fn).to(device, dt)
    model.config._ghost_idx = sorted(cand)                                    # for targeted ghost-distill (heal supervises these locally)
    print(f"[surgery] DEPTH-CAPTURE: {len(cand)} least-important blocks -> rank-{rank} ghost mini-FFNs (heal-trained), "
          f"idx={sorted(cand)}")
    return len(cand)


class CanonMix(nn.Module):
    """NET-NEW cheap computation (Canon-layer style): causal depthwise conv (k tokens) on the residual, zero-init
    gated (gamma=0 -> exact no-op at surgery; the heal grows it). Adds LOCAL TOKEN MIXING — a computation type the
    deleted layers' attention provided — at ~k*H params per layer. Additive path: touches no pretrained weight."""
    def __init__(self, h, k=4):
        super().__init__()
        self.conv = nn.Conv1d(h, h, k, groups=h, bias=False)
        self.gamma = nn.Parameter(torch.zeros(h))
        self.k = k
        nn.init.normal_(self.conv.weight, std=0.02)

    def forward(self, x):                                     # (B, S, H)
        y = self.conv(nn.functional.pad(x.transpose(1, 2), (self.k - 1, 0))).transpose(1, 2)  # causal
        return x + self.gamma * y


def add_canon_mix(model, arch, k=4):
    """Attach a CanonMix to every remaining layer (registered on the block -> optimizer sees it) and apply it to
    the layer INPUT via a forward-pre-hook. No pretrained weight touched; exact no-op at init."""
    layers = arch.get_layers(model); h = model.config.hidden_size
    dt = next(model.parameters()).dtype; dev = next(model.parameters()).device
    def mk(blk):
        def hook(m, args, kwargs):
            if not args: return None
            return (blk.canon(args[0]),) + args[1:], kwargs
        return hook
    n = 0
    for blk in layers:
        blk.canon = CanonMix(h, k).to(dev, dt)
        blk.register_forward_pre_hook(mk(blk), with_kwargs=True)
        n += 1
    print(f"[surgery] CANON-MIX: {n} causal depthwise conv mixers (k={k}, zero-init gate = no-op at init, "
          f"{n * (k * h + h) / 1e3:.0f}K params) — net-new token-mixing capacity for the heal")
    return n


@torch.no_grad()
def add_layer_tie(model, arch, n_tie, calib_ids, device):
    """LAYER TYING (MobileLLM-style repeat): the n_tie least-important layers SHARE their predecessor's
    Parameters (unique params == deleting them) but stay in the stack — full depth of compute, ZERO weights
    deleted. The third depth operator: delete / merge(average) / TIE(share). Distinct module objects keep
    their own layer_idx (KV cache correct); nn.Module.parameters() dedupes shared tensors for param counts."""
    layers = arch.get_layers(model)
    io = {}
    def mk(li):
        def hk(m, inp, out):
            x = inp[0].detach().float(); y = (out[0] if isinstance(out, tuple) else out).detach().float()
            io[li] = (x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]))
        return hk
    hooks = [b.register_forward_hook(mk(i)) for i, b in enumerate(layers)]
    model.eval(); model(calib_ids.to(device), use_cache=False)
    for hk in hooks: hk.remove()
    imp = {i: (1 - torch.nn.functional.cosine_similarity(x, y, dim=-1).mean()).item() for i, (x, y) in io.items()}
    cand = sorted(sorted(range(1, len(layers) - 1), key=lambda i: imp[i])[:n_tie])
    tied = set(cand)
    for i in cand:
        t = i - 1
        while t in tied: t -= 1                                      # tie to the nearest untied predecessor
        src, dst = layers[t], layers[i]
        smap = dict(src.named_parameters())
        for name, _ in list(dst.named_parameters()):
            mod = dst
            parts = name.split(".")
            for p in parts[:-1]: mod = getattr(mod, p)
            mod._parameters[parts[-1]] = smap[name]                  # SHARE the Parameter object itself
    print(f"[surgery] LAYER-TIE: {len(cand)} least-important layers now SHARE their predecessor's params "
          f"(depth stays {len(layers)}, unique params == deleting {len(cand)}), idx={cand}")
    return len(cand)


class BusReadLinear(nn.Module):
    """Reader on the FULL-width residual bus: gathers this layer's own top-w channels, then a w-input Linear.
    Params = width-w pruned reader, but the bus keeps ALL channels (nothing globally deleted)."""
    def __init__(self, lin, idx):
        super().__init__()
        self.register_buffer("idx", idx)
        n = nn.Linear(idx.numel(), lin.out_features, bias=lin.bias is not None).to(lin.weight.device, lin.weight.dtype)
        n.weight.data.copy_(lin.weight.data[:, idx])
        if lin.bias is not None: n.bias.data.copy_(lin.bias.data)
        self.lin = n
    def forward(self, x):
        return self.lin(x.index_select(-1, self.idx))


class BusWriteLinear(nn.Module):
    """Writer to the FULL-width residual bus: w-output Linear scattered into this layer's own top-w channels."""
    def __init__(self, lin, idx, full_dim):
        super().__init__()
        self.register_buffer("idx", idx); self.full_dim = full_dim
        n = nn.Linear(lin.in_features, idx.numel(), bias=False).to(lin.weight.device, lin.weight.dtype)
        n.weight.data.copy_(lin.weight.data[idx, :])
        self.lin = n
    def forward(self, x):
        y = self.lin(x)
        out = y.new_zeros(*y.shape[:-1], self.full_dim)
        return out.index_copy(-1, self.idx, y)


@torch.no_grad()
def prune_hidden_bus(model, arch, keep_w, calib_ids, device):
    """FULL-WIDTH-BUS reallocation: the residual stream KEEPS all H channels (activations are free — only the
    embedding/norms pay full width); each layer's readers/writers are pruned to their OWN top-keep_w channels
    (4 index sets per layer: attn-read, mlp-read from input energy; attn-write, mlp-write from output energy).
    vs global width-prune: same body params at width keep_w, but no channel is deleted everywhere — layer 5 can
    keep channels layer 20 drops. The per-layer reallocation of the width budget."""
    layers = arch.get_layers(model); H = model.config.hidden_size
    rd, wr = {}, {}
    def mk_rd(key):
        def h(m, inp):
            a = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            rd[key] = rd.get(key, 0) + (a ** 2).sum(0)
        return h
    def mk_wr(key):
        def h(m, inp, out):
            a = out.detach().float().reshape(-1, out.shape[-1])
            wr[key] = wr.get(key, 0) + (a ** 2).sum(0)
        return h
    hooks = []
    for i, b in enumerate(layers):
        hooks += [b.self_attn.q_proj.register_forward_pre_hook(mk_rd(("a", i))),
                  b.mlp.gate_proj.register_forward_pre_hook(mk_rd(("m", i))),
                  b.self_attn.o_proj.register_forward_hook(mk_wr(("a", i))),
                  b.mlp.down_proj.register_forward_hook(mk_wr(("m", i)))]
    model.eval(); model(calib_ids.to(device), use_cache=False)
    for hk in hooks: hk.remove()
    top = lambda e: torch.topk(e, keep_w).indices.sort().values
    for i, b in enumerate(layers):
        ra, rm = top(rd[("a", i)]), top(rd[("m", i)])
        wa, wm = top(wr[("a", i)]), top(wr[("m", i)])
        at = b.self_attn
        at.q_proj = BusReadLinear(at.q_proj, ra); at.k_proj = BusReadLinear(at.k_proj, ra); at.v_proj = BusReadLinear(at.v_proj, ra)
        at.o_proj = BusWriteLinear(at.o_proj, wa, H)
        b.mlp.gate_proj = BusReadLinear(b.mlp.gate_proj, rm); b.mlp.up_proj = BusReadLinear(b.mlp.up_proj, rm)
        b.mlp.down_proj = BusWriteLinear(b.mlp.down_proj, wm, H)
    print(f"[surgery] BUS-WIDTH: residual bus stays {H}, per-layer read/write subsets w={keep_w} "
          f"({len(layers)} layers x 4 index sets, per-layer energy selection)")
    return True


@torch.no_grad()
def add_layer_merge(model, arch, n_merge, calib_ids, device, alpha=0.5):
    """LAYER-MERGE (LaCo-style): fold each of the n_merge least-important interior layers into its PREDECESSOR by
    parameter interpolation (pred = (1-alpha)*pred + alpha*merged), then drop the merged layer. A DIFFERENT depth
    operator from deletion (keeps a trace of the merged layer's function in full-rank weights) and from ghost-capture
    (no fresh low-rank module — reuses the predecessor's full capacity). Same param count as deleting n_merge."""
    layers = arch.get_layers(model)
    io = {}
    def mk(li):
        def hk(m, inp, out):
            x = inp[0].detach().float(); y = (out[0] if isinstance(out, tuple) else out).detach().float()
            io[li] = (x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]))
        return hk
    hooks = [b.register_forward_hook(mk(i)) for i, b in enumerate(layers)]
    model.eval(); model(calib_ids.to(device), use_cache=False)
    for hk in hooks: hk.remove()
    imp = {i: (1 - torch.nn.functional.cosine_similarity(x, y, dim=-1).mean()).item() for i, (x, y) in io.items()}
    cand = sorted(sorted(range(1, len(layers) - 1), key=lambda i: imp[i])[:n_merge])   # least-important interior, ascending
    merged = set(cand)
    for i in cand:                                                     # fold i into nearest UNMERGED predecessor
        t = i - 1
        while t in merged: t -= 1
        tgt, src = layers[t], layers[i]
        for (pn, tp), (_, sp) in zip(tgt.named_parameters(), src.named_parameters()):
            tp.data.mul_(1 - alpha).add_(sp.data, alpha=alpha)
    new = nn.ModuleList([b for j, b in enumerate(layers) if j not in merged])
    arch.set_layers(model, new); setattr(model.config, arch.n_layer_attr, len(new))
    reassign_layer_idx(new)
    print(f"[surgery] LAYER-MERGE: folded {len(merged)} least-important layers into predecessors (alpha={alpha}), "
          f"{len(layers)} -> {len(new)} layers, idx={cand}")
    return len(merged)


@torch.no_grad()
def _capture_layer(blk, cfg, qa, oa, ga, da, keep_k, rank_f, keep_h, rank_a, use_cov):
    """Capture ONE block's FFN + attn (hybrid) from its (possibly drifted) input activations qa/oa/ga/da."""
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads); nh = cfg.num_attention_heads
    # --- FFN ---
    mlp = blk.mlp; inter = mlp.gate_proj.out_features
    imp = (da ** 2).sum(0); topk = torch.topk(imp, keep_k).indices.sort().values
    keep = torch.zeros(inter, dtype=torch.bool, device=da.device); keep[topk] = True; rem = (~keep).nonzero(as_tuple=True)[0]
    hs = ga.abs().amax(0).clamp_min(1e-6); hcov = (ga.t() @ ga) if use_cov else None
    gate_top, up_top = _sub_linear(mlp.gate_proj, topk, 0), _sub_linear(mlp.up_proj, topk, 0)
    down_top = _sub_linear(mlp.down_proj, topk, 1)
    gate_rem = asvd_factorize(_sub_linear(mlp.gate_proj, rem, 0), hs, rank_f, cov=hcov)
    up_rem = asvd_factorize(_sub_linear(mlp.up_proj, rem, 0), hs, rank_f, cov=hcov)
    down_rem = asvd_factorize(_sub_linear(mlp.down_proj, rem, 1), imp[rem].sqrt().clamp_min(1e-6), rank_f)
    blk.mlp = HybridFFN(gate_top, up_top, down_top, gate_rem, up_rem, down_rem, mlp.act_fn)
    # --- ATTN ---
    attn = blk.self_attn; qp, op = attn.q_proj, attn.o_proj; Wq = qp.weight.data.float()
    he = (oa ** 2).sum(0).reshape(nh, hd).sum(1) * Wq.reshape(nh, hd, -1).norm(dim=(1, 2))
    toph = torch.topk(he, keep_h).indices.sort().values
    keepm = torch.zeros(nh, dtype=torch.bool, device=qa.device); keepm[toph] = True; remh = (~keepm).nonzero(as_tuple=True)[0]
    di = torch.arange(hd, device=qa.device)
    top_idx = torch.cat([h * hd + di for h in toph]); rem_idx = torch.cat([h * hd + di for h in remh])
    qsc = qa.abs().amax(0).clamp_min(1e-6); qcov = (qa.t() @ qa) if use_cov else None
    q_top = _sub_linear(qp, top_idx, 0); o_top = _sub_linear(op, top_idx, 1)
    q_rem = asvd_factorize(_sub_linear(qp, rem_idx, 0), qsc, rank_a, cov=qcov)
    o_rem = asvd_factorize(_sub_linear(op, rem_idx, 1), (oa ** 2).sum(0).sqrt().clamp_min(1e-6)[rem_idx], rank_a)
    q_up = getattr(q_rem, "up", q_rem); q_dn = getattr(q_rem, "down", nn.Identity())
    o_up = getattr(o_rem, "up", o_rem); o_dn = getattr(o_rem, "down", nn.Identity())
    attn.q_proj = HybridRowLinear(q_top, top_idx, q_up, q_dn, rem_idx, qp.bias.data if qp.bias is not None else None, Wq.shape[0])
    attn.o_proj = HybridColLinear(o_top, top_idx, o_up, o_dn, rem_idx, op.bias.data if op.bias is not None else None)


@torch.no_grad()
def capture_sequential(model, arch, keep_k, rank_f, keep_h, rank_a, calib_ids, device):
    """SEQUENTIAL / GPTQ-style capture: step layer-by-layer; each layer is captured on the DRIFTED input from the
    already-compressed upstream (not the original), so cross-layer compression error doesn't accumulate."""
    inner = model.model; layers = arch.get_layers(model); cfg = model.config
    use_cov = getattr(cfg, "_hybrid_cov", False)
    ci = calib_ids.to(device); S = ci.shape[1]
    res = inner.embed_tokens(ci); pos_ids = torch.arange(S, device=device)[None, :]
    pos_emb = inner.rotary_emb(res, pos_ids)
    mask = torch.triu(torch.full((S, S), float("-inf"), device=device, dtype=torch.float32), 1)[None, None].to(res.dtype)
    for li, blk in enumerate(layers):
        acc = {}
        def mk(nm):
            def h(m, inp): acc[nm] = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            return h
        hks = [blk.self_attn.q_proj.register_forward_pre_hook(mk("q")), blk.self_attn.o_proj.register_forward_pre_hook(mk("o")),
               blk.mlp.gate_proj.register_forward_pre_hook(mk("g")), blk.mlp.down_proj.register_forward_pre_hook(mk("d"))]
        a_out = blk.self_attn(blk.input_layernorm(res), position_embeddings=pos_emb, attention_mask=mask)[0]  # original (collect)
        r2 = res + a_out
        _ = blk.mlp(blk.post_attention_layernorm(r2))                            # original mlp (collect g/d)
        for h in hks: h.remove()
        _capture_layer(blk, cfg, acc["q"], acc["o"], acc["g"], acc["d"], keep_k, rank_f, keep_h, rank_a, use_cov)
        blk.to(device)                                                          # captured modules (LowRankLinear) init on CPU -> move before forwarding
        res = res + blk.self_attn(blk.input_layernorm(res), position_embeddings=pos_emb, attention_mask=mask)[0]  # CAPTURED
        res = res + blk.mlp(blk.post_attention_layernorm(res))                   # CAPTURED -> drift for next layer
    print(f"[surgery] SEQUENTIAL capture: {len(layers)} layers, each calibrated on drifted compressed-upstream input"
          f"{' (cov)' if use_cov else ''}")
    return len(layers)


@torch.no_grad()
def add_hybrid_ffn(model, arch, keep_k, rank, calib_ids, device):
    """Replace each kept block's MLP with a HybridFFN: top-`keep_k` intermediate channels (by activation energy)
    kept exact, the remaining represented low-rank (rank `rank`) via activation-aware SVD."""
    layers = arch.get_layers(model)
    use_cov = getattr(model.config, "_hybrid_cov", False)        # SVD-LLM full-covariance whitening on residual-readers
    inter_imp, hid_scale, hid_cov = {}, {}, {}
    def mk_down(li):
        def h(m, inp):
            a = inp[0].detach().float().reshape(-1, inp[0].shape[-1]); inter_imp[li] = inter_imp.get(li, 0) + (a ** 2).sum(0)
        return h
    def mk_gate(li):
        def h(m, inp):
            a = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            hid_scale[li] = torch.maximum(hid_scale[li], a.abs().amax(0)) if li in hid_scale else a.abs().amax(0)
            if use_cov: hid_cov[li] = hid_cov.get(li, 0) + a.t() @ a       # [hidden,hidden] residual covariance
        return h
    hooks = []
    for i, b in enumerate(layers):
        hooks += [b.mlp.down_proj.register_forward_pre_hook(mk_down(i)), b.mlp.gate_proj.register_forward_pre_hook(mk_gate(i))]
    model.eval(); model(calib_ids.to(device), use_cache=False)
    for hk in hooks: hk.remove()
    n = 0
    for i, b in enumerate(layers):
        mlp = b.mlp; inter = mlp.gate_proj.out_features
        if i not in inter_imp: continue
        imp = inter_imp[i]; topk = torch.topk(imp, keep_k).indices.sort().values
        keep = torch.zeros(inter, dtype=torch.bool, device=imp.device); keep[topk] = True
        rem = (~keep).nonzero(as_tuple=True)[0]
        hs = hid_scale[i].clamp_min(1e-6)
        gate_top, up_top = _sub_linear(mlp.gate_proj, topk, 0), _sub_linear(mlp.up_proj, topk, 0)
        down_top = _sub_linear(mlp.down_proj, topk, 1)
        oa = getattr(model.config, "_hybrid_out_aware", False); her = getattr(model.config, "_hybrid_eora_rank", 0)
        hcov = hid_cov.get(i) if use_cov else None                       # residual covariance for gate/up readers (RMSNorm-aware)
        gate_rem = asvd_factorize(_sub_linear(mlp.gate_proj, rem, 0), hs, rank, eora_rank=her, cov=hcov)
        up_rem = asvd_factorize(_sub_linear(mlp.up_proj, rem, 0), hs, rank, eora_rank=her, cov=hcov)
        down_rem = asvd_factorize(_sub_linear(mlp.down_proj, rem, 1), imp[rem].sqrt().clamp_min(1e-6), rank,
                                  eora_rank=her, out_scale=hs if oa else None)         # capture + EoRA error-compensation
        b.mlp = HybridFFN(gate_top, up_top, down_top, gate_rem, up_rem, down_rem, mlp.act_fn); n += 1
    print(f"[surgery] HybridFFN: {n} MLPs -> top-{keep_k} exact + rank-{rank} captured-remaining (capture-not-drop)")
    return n


class HybridRowLinear(nn.Module):
    """Output-row hybrid (for q_proj): rows of top heads kept EXACT, remaining heads' rows LOW-RANK (A@B),
    scattered back to original row order so the attention reshape/rotary/GQA-grouping is unchanged."""
    def __init__(self, top, top_idx, rem_up, rem_down, rem_idx, bias, out_dim):
        super().__init__()
        self.top, self.rem_up, self.rem_down = top, rem_up, rem_down
        self.register_buffer("top_idx", top_idx); self.register_buffer("rem_idx", rem_idx)
        self.out_dim = out_dim
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

    def forward(self, x):
        y = x.new_zeros(*x.shape[:-1], self.out_dim)
        y[..., self.top_idx] = self.top(x)
        y[..., self.rem_idx] = self.rem_up(self.rem_down(x))
        if self.bias is not None: y = y + self.bias
        return y


class HybridColLinear(nn.Module):
    """Input-col hybrid (for o_proj): cols of top heads kept EXACT, remaining heads' cols LOW-RANK.
    out = top(x[:,top_idx]) + rem_up(rem_down(x[:,rem_idx])) + bias."""
    def __init__(self, top, top_idx, rem_up, rem_down, rem_idx, bias):
        super().__init__()
        self.top, self.rem_up, self.rem_down = top, rem_up, rem_down
        self.register_buffer("top_idx", top_idx); self.register_buffer("rem_idx", rem_idx)
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

    def forward(self, x):
        y = self.top(x[..., self.top_idx]) + self.rem_up(self.rem_down(x[..., self.rem_idx]))
        if self.bias is not None: y = y + self.bias
        return y


@torch.no_grad()
def add_hybrid_attn(model, arch, keep_heads, rank, calib_ids, device):
    """Capture-not-drop attention: keep top-`keep_heads` query heads EXACT (wanda importance) + represent the
    remaining heads' q_proj rows / o_proj cols LOW-RANK (activation-aware SVD). k/v untouched. Heads stay in place."""
    layers = arch.get_layers(model); cfg = model.config
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads); nh = cfg.num_attention_heads
    use_cov = getattr(model.config, "_hybrid_cov", False)
    qin, qsc, attsc, qcov = {}, {}, {}, {}
    def mk_q(li):
        def h(m, inp):
            a = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            qsc[li] = torch.maximum(qsc[li], a.abs().amax(0)) if li in qsc else a.abs().amax(0)
            if use_cov: qcov[li] = qcov.get(li, 0) + a.t() @ a            # residual covariance for q reader (RMSNorm-aware)
        return h
    def mk_o(li):
        def h(m, inp):                                       # o_proj input = per-head attention output
            a = inp[0].detach().float().reshape(-1, inp[0].shape[-1]); attsc[li] = attsc.get(li, 0) + (a ** 2).sum(0)
        return h
    hooks = []
    for i, b in enumerate(layers):
        hooks += [b.self_attn.q_proj.register_forward_pre_hook(mk_q(i)), b.self_attn.o_proj.register_forward_pre_hook(mk_o(i))]
    model.eval(); model(calib_ids.to(device), use_cache=False)
    for hk in hooks: hk.remove()
    n = 0
    for i, b in enumerate(layers):
        attn = b.self_attn; qp, op = attn.q_proj, attn.o_proj
        if i not in attsc: continue
        Wq = qp.weight.data.float()                          # [nh*hd, hidden]
        he = attsc[i].reshape(nh, hd).sum(1) * Wq.reshape(nh, hd, -1).norm(dim=(1, 2))   # wanda-ish per-head importance
        toph = torch.topk(he, keep_heads).indices.sort().values
        keepm = torch.zeros(nh, dtype=torch.bool, device=device); keepm[toph] = True
        remh = (~keepm).nonzero(as_tuple=True)[0]
        di = torch.arange(hd, device=device)
        top_idx = torch.cat([h * hd + di for h in toph]); rem_idx = torch.cat([h * hd + di for h in remh])
        qsc_i = qsc[i].clamp_min(1e-6)
        q_top = _sub_linear(qp, top_idx, 0)
        her = getattr(model.config, "_hybrid_eora_rank", 0)
        qcov_i = qcov.get(i) if use_cov else None
        q_rem = asvd_factorize(_sub_linear(qp, rem_idx, 0), qsc_i, rank, eora_rank=her, cov=qcov_i)  # LowRankLinear or plain
        attsc_i = attsc[i].sqrt().clamp_min(1e-6)
        o_top = _sub_linear(op, top_idx, 1)
        oa = getattr(model.config, "_hybrid_out_aware", False)                          # o_proj writes residual -> output-aware rows
        o_rem = asvd_factorize(_sub_linear(op, rem_idx, 1), attsc_i[rem_idx], rank, eora_rank=her, out_scale=qsc_i if oa else None)
        q_up = getattr(q_rem, "up", q_rem); q_dn = getattr(q_rem, "down", nn.Identity())  # plain Linear -> up=Linear, down=Identity
        o_up = getattr(o_rem, "up", o_rem); o_dn = getattr(o_rem, "down", nn.Identity())
        attn.q_proj = HybridRowLinear(q_top, top_idx, q_up, q_dn, rem_idx,
                                      qp.bias.data if qp.bias is not None else None, Wq.shape[0])
        attn.o_proj = HybridColLinear(o_top, top_idx, o_up, o_dn, rem_idx, op.bias.data if op.bias is not None else None)
        n += 1
    print(f"[surgery] HybridAttn: {n} attns -> top-{keep_heads}/{nh} heads exact + rank-{rank} captured-remaining (q/o)")
    return n


class EoRAAdd(nn.Module):
    """EoRA-on-top: y = base(x) + up_e(down_e(x)). Adds a trainable low-rank error-compensation path IN PARALLEL
    to an already-compressed/reconstructed linear (e.g. obs-reconstructed down_proj). down_e is initialised to the
    top-r eigenvectors of the input activation covariance (the EoRA eigenspace — the directions the layer's inputs
    actually occupy), up_e is zero (path starts as a no-op so surgery is unchanged at init); the final heal trains
    both. Tests whether dedicated low-rank capacity, positioned in the activation eigenspace, helps beyond OBS."""
    def __init__(self, base, down_e, up_e):
        super().__init__()
        self.base = base
        r = down_e.shape[0]
        self.down_e = nn.Linear(down_e.shape[1], r, bias=False).to(down_e.device, down_e.dtype)
        self.up_e = nn.Linear(r, up_e.shape[0], bias=False).to(up_e.device, up_e.dtype)
        with torch.no_grad():
            self.down_e.weight.copy_(down_e); self.up_e.weight.copy_(up_e)

    def forward(self, x):
        return self.base(x) + self.up_e(self.down_e(x))


@torch.no_grad()
def add_eora(model, arch, calib_ids, device, rank):
    """Wrap each kept block's down_proj with an EoRAAdd low-rank path, init in the input-activation eigenspace."""
    layers = arch.get_layers(model)
    cov = {}
    def mk(li):
        def hook(mod, inp):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            cov[li] = cov.get(li, 0) + x.t() @ x
        return hook
    hooks = [layers[i].mlp.down_proj.register_forward_pre_hook(mk(i)) for i in range(len(layers))]
    model.eval(); model(calib_ids.to(device), use_cache=False)
    for h in hooks: h.remove()
    n = 0
    for i, blk in enumerate(layers):
        dp = blk.mlp.down_proj
        if i not in cov: continue
        C = cov[i]; r = min(rank, C.shape[0] - 1)
        evec = torch.linalg.eigh(C)[1][:, -r:].t().contiguous()          # [r, in] top-r input directions
        down_e = evec.to(dp.weight.dtype)
        up_e = torch.zeros(dp.out_features, r, device=device, dtype=dp.weight.dtype)   # no-op init; heal trains
        blk.mlp.down_proj = EoRAAdd(dp, down_e, up_e); n += 1
    print(f"[surgery] EoRA-on-top: added rank-{rank} paths to {n} down_proj (eigenspace-init, heal-trained)")
    return n


def obs_ffn_prune(model, arch, layers, calib_ids, device, keep_frac):
    """Local-OBS FFN-width pruning (SlimGPT/OSSCAR family, 2412.18110): prune intermediate channels then
    RECONSTRUCT down_proj via per-layer least-squares so the surviving channels ABSORB the removed ones'
    contribution (concentrate damage locally — the opposite of SliceGPT's global rotation that distributed
    it and lost). down_new = Yᵀ·Aₚ·(AₚᵀAₚ + λI)⁻¹ where A = intermediate activation, Y = full FFN output."""
    UP_NAMES = ("gate_proj", "up_proj", "c_fc", "dense_h_to_4h")
    DOWN_NAMES = ("down_proj", "dense_4h_to_h", "c_proj")
    targets = []
    for b in layers:
        mlp = b.mlp
        up = [(nm, getattr(mlp, nm)) for nm in UP_NAMES if hasattr(mlp, nm)]
        dn = next((nm for nm in DOWN_NAMES if hasattr(mlp, nm)), None)
        if up and dn is not None:
            targets.append((mlp, up, dn, getattr(mlp, dn)))
    acc = {id(d): [] for _, _, _, d in targets}
    def mk(d):
        def hook(mod, inp): acc[id(d)].append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu())
        return hook
    hooks = [d.register_forward_pre_hook(mk(d)) for _, _, _, d in targets]
    model.eval()
    with torch.no_grad():
        for i in range(0, min(calib_ids.size(1), 2048), 256):
            model(calib_ids[:, i:i + 256].to(device), use_cache=False)
    for h in hooks:
        h.remove()
    n = 0
    for mlp, up, dn_name, down in targets:
        A = torch.cat(acc[id(down)], 0).to(device)                  # (N, inter) intermediate activation
        inter = A.shape[1]; keep_k = max(1, int(keep_frac * inter))
        if keep_k >= inter:
            continue
        W = down.weight.data.float()                                # (hidden, inter)
        Y = A @ W.t()                                               # (N, hidden) full FFN output (target)
        score = W.norm(dim=0) * up[0][1].weight.float().norm(dim=1) # channel importance
        idx = torch.topk(score, keep_k).indices.sort().values
        Ap = A[:, idx]                                              # (N, keep_k)
        G = Ap.t() @ Ap + 1e-2 * torch.eye(keep_k, device=device)  # local Hessian
        Wnew = torch.linalg.solve(G, Ap.t() @ Y).t()               # (hidden, keep_k) reconstructed down_proj
        dev, dt = down.weight.device, down.weight.dtype
        for uname, u in up:                                         # slice up/gate rows -> SWAP into the mlp
            nu = nn.Linear(u.in_features, keep_k, bias=u.bias is not None).to(dev, dt)
            nu.weight.data.copy_(u.weight.data[idx])
            if u.bias is not None: nu.bias.data.copy_(u.bias.data[idx])
            setattr(mlp, uname, nu)
        nd = nn.Linear(keep_k, down.out_features, bias=down.bias is not None).to(dev, dt)
        nd.weight.data.copy_(Wnew.to(dt))
        if down.bias is not None: nd.bias.data.copy_(down.bias.data)
        setattr(mlp, dn_name, nd); n += 1                          # REAL swap (not _replace_child marker)
        del A, Y, Ap, G, Wnew; torch.cuda.empty_cache()
    return n


def obs_attn_prune(model, arch, layers, calib_ids, device, keep_frac):
    """Local-OBS ATTENTION-head pruning (extends obs_ffn to the attention axis): drop low-importance heads +
    RECONSTRUCT o_proj via per-layer least-squares so surviving heads absorb the removed ones. UNIFORM head
    count across pruned layers (so the per-layer head count stays consistent), but each layer keeps its OWN
    important heads. gpt_neox only for now (interleaved qkv); Llama/Qwen GQA is a follow-up."""
    if arch.family != "gpt_neox":
        print(f"[obs_attn] only gpt_neox supported (got {arch.family}); skipping"); return 0
    nh = model.config.num_attention_heads
    dh = model.config.hidden_size // nh
    keep_h = max(1, int(keep_frac * nh))
    if keep_h >= nh:
        return 0
    tgt = [(b, b.attention, b.attention.dense, b.attention.query_key_value) for b in layers]
    acc = {id(o): [] for _, _, o, _ in tgt}
    def mk(o):
        def hook(mod, inp): acc[id(o)].append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu())
        return hook
    hooks = [o.register_forward_pre_hook(mk(o)) for _, _, o, _ in tgt]
    model.eval()
    with torch.no_grad():
        for i in range(0, min(calib_ids.size(1), 2048), 256):
            model(calib_ids[:, i:i + 256].to(device), use_cache=False)
    for h in hooks:
        h.remove()
    n = 0
    for (b, attn, o, qkv) in tgt:
        A = torch.cat(acc[id(o)], 0).to(device)                       # (N, nh*dh) concatenated head outputs
        W = o.weight.data.float()                                     # (hidden, nh*dh)
        Y = A @ W.t()                                                 # (N, hidden) full o_proj output (target)
        score = torch.stack([A[:, h*dh:(h+1)*dh].norm() * W[:, h*dh:(h+1)*dh].norm() for h in range(nh)])
        kept = torch.topk(score, keep_h).indices.sort().values
        cols = torch.cat([torch.arange(h*dh, (h+1)*dh) for h in kept]).to(device)
        Ap = A[:, cols]
        G = Ap.t() @ Ap + 1e-2 * torch.eye(cols.numel(), device=device)
        Wnew = torch.linalg.solve(G, Ap.t() @ Y).t()                  # (hidden, keep_h*dh) reconstructed o_proj
        dev, dt = o.weight.device, o.weight.dtype
        no = nn.Linear(cols.numel(), o.out_features, bias=o.bias is not None).to(dev, dt)
        no.weight.data.copy_(Wnew.to(dt))
        if o.bias is not None: no.bias.data.copy_(o.bias.data)
        attn.dense = no                                              # REAL swap (gpt_neox o_proj)
        Wq = qkv.weight.data.view(nh, 3*dh, -1)[kept].reshape(-1, qkv.in_features)   # keep heads' q,k,v rows
        nq = nn.Linear(qkv.in_features, Wq.shape[0], bias=qkv.bias is not None).to(dev, dt)
        nq.weight.data.copy_(Wq.to(dt))
        if qkv.bias is not None:
            nq.bias.data.copy_(qkv.bias.data.view(nh, 3*dh)[kept].reshape(-1).to(dt))
        attn.query_key_value = nq                                    # REAL swap
        attn.num_attention_heads = keep_h                            # per-layer head count (forward reads self.)
        if hasattr(attn, "hidden_size"): attn.hidden_size = keep_h * dh
        n += 1
        del A, Y, Ap, G, Wnew; torch.cuda.empty_cache()
    return n


def svd_compress(model, arch, layers, calib_ids, device, rank_frac, do_attn=False, eora_frac=0.0):
    """Factorize FFN (and optionally attn) weights in `layers` via activation-aware SVD to rank_frac × full
    rank. eora_frac>0 adds an EoRA error-compensation path (init from the compression-error eigenspace) —
    train it (optionally with a frozen backbone) to repair the compression cheaply."""
    targets = []                                          # (parent_module, attr_name, linear)
    for b in layers:
        mlp = b.mlp
        for nm in ("gate_proj", "up_proj", "down_proj", "c_fc", "c_proj", "dense_h_to_4h", "dense_4h_to_h"):
            if hasattr(mlp, nm) and isinstance(getattr(mlp, nm), nn.Linear):
                targets.append((mlp, nm, getattr(mlp, nm)))
        if do_attn:
            at = getattr(b, "self_attn", None) or getattr(b, "attention", None)
            if at is not None:
                # Llama/Qwen: q/k/v/o_proj · GPT-NeoX: query_key_value/dense · GPT2: c_attn/c_proj
                for nm in ("q_proj", "k_proj", "v_proj", "o_proj", "query_key_value", "dense", "c_attn", "c_proj"):
                    if hasattr(at, nm) and isinstance(getattr(at, nm), nn.Linear):
                        targets.append((at, nm, getattr(at, nm)))
    # accumulate sum of squares of each target's INPUT, per input channel
    acc = {id(lin): torch.zeros(lin.in_features, device=device) for _, _, lin in targets}
    cnt = {id(lin): 0 for _, _, lin in targets}
    def mk(lin):
        def hook(mod, inp):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            acc[id(lin)] += (x * x).sum(0); cnt[id(lin)] += x.shape[0]
        return hook
    hooks = [lin.register_forward_pre_hook(mk(lin)) for _, _, lin in targets]
    model.eval()
    with torch.no_grad():
        for i in range(0, min(calib_ids.size(1), 4096), 512):
            model(calib_ids[:, i:i + 512].to(device), use_cache=False)
    for h in hooks:
        h.remove()
    n_fact = 0
    for parent, nm, lin in targets:
        scale = (acc[id(lin)] / max(1, cnt[id(lin)])).sqrt()        # per-channel activation RMS
        full = min(lin.out_features, lin.in_features)
        r = max(1, int(rank_frac * full))
        er = int(eora_frac * full)
        new = asvd_factorize(lin, scale, r, eora_rank=er)
        if new is not lin:
            setattr(parent, nm, new.to(device)); n_fact += 1
    return n_fact


def slicegpt_rotate(model, calib_ids, device, importance=False, solver="pca"):
    """SliceGPT rotation (Llama/Qwen RMSNorm): fold RMSNorm gains into the weights that READ them, then
    rotate the whole residual stream into its PCA basis via an orthogonal Q. Output is UNCHANGED (pure
    change of basis) — afterward the dominant directions are first, so a subsequent prune_hidden_width
    keeps the DOMINANT SUBSPACE. If importance=True, weight the covariance by gradient saliency (Flex-KD
    idea) so we rank directions by TASK importance, not just variance (VkD: the variance-tail can be
    discriminative). Returns True on success."""
    layers = model.model.layers
    H = model.config.hidden_size
    emb = model.model.embed_tokens
    head = model.lm_head
    fnorm = model.model.norm

    if head.weight.data_ptr() == emb.weight.data_ptr():
        # UNTIE: tied head can't take the folded final-norm gain (corrupts the embedding role). Give the
        # lm_head its own weight; the extra params are recoverable via --svd_embed factorization.
        new_head = nn.Linear(H, head.weight.shape[0], bias=head.bias is not None).to(head.weight.device, head.weight.dtype)
        new_head.weight.data.copy_(head.weight.data)
        model.lm_head = new_head; model.config.tie_word_embeddings = False
        head = new_head
    def fold_gain(norm, readers):                         # fold RMSNorm gain g into reader linears, set g=1
        g = norm.weight.data.float()
        for lin in readers:
            lin.weight.data = (lin.weight.data.float() * g.unsqueeze(0)).to(lin.weight.dtype)
        norm.weight.data = torch.ones_like(norm.weight.data)
    for blk in layers:
        fold_gain(blk.input_layernorm, [blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj])
        fold_gain(blk.post_attention_layernorm, [blk.mlp.gate_proj, blk.mlp.up_proj])
    fold_gain(fnorm, [head])

    # collect residual-stream covariance over calib data (optionally gradient-SALIENCY weighted)
    cov = torch.zeros(H, H, device=device, dtype=torch.float32)
    act_diag = torch.zeros(H, device=device, dtype=torch.float32)   # per-dim activation energy (for outlier detection)
    model.eval()
    if importance and solver == "pca": solver = "saliency"   # back-compat: --slice_importance == saliency solver
    if solver in ("saliency", "fisher", "outlier_fisher", "outlier_saliency"):
        # Flex-KD/Fisher idea: rank directions by saliency (activation × loss-gradient), not pure variance,
        # so the discriminative variance-tail (VkD) isn't discarded. cov of (x ⊙ |g|).
        saved_rg = [(p, p.requires_grad) for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad_(True)                        # graph must exist so hidden states are differentiable
        for i in range(0, min(calib_ids.size(1), 2048), 256):
            ids = calib_ids[:, i:i + 256].to(device)
            grads = {}
            def mk_hook(k):
                def hk(g): grads[k] = g.detach()
                return hk
            out = model(ids, output_hidden_states=True, use_cache=False)
            handles = [h.register_hook(mk_hook(j)) for j, h in enumerate(out.hidden_states) if h.requires_grad]
            lg = out.logits[:, :-1].reshape(-1, out.logits.size(-1))
            loss = F.cross_entropy(lg.float(), ids[:, 1:].reshape(-1))
            loss.backward()
            for j, h in enumerate(out.hidden_states):
                if j in grads:
                    if solver in ("fisher", "outlier_fisher"):  # empirical Fisher: cov of the loss-GRADIENT (g g^T) -> loss-sensitive subspace
                        g = grads[j][0].float().reshape(-1, H); cov += g.t() @ g
                        if solver == "outlier_fisher":          # also track activation energy to locate the sink/outlier dims
                            x = h[0].detach().float().reshape(-1, H); act_diag += (x * x).sum(0)
                    else:                                     # saliency: cov of (activation x loss-gradient)
                        x = h[0].detach().float()
                        sal = (x * grads[j][0].float().abs().clamp_min(1e-6)).reshape(-1, H)
                        cov += sal.t() @ sal
                        if solver == "outlier_saliency":      # COMBINE the two winners: keep sink exact + saliency-rotate the rest
                            act_diag += (x.reshape(-1, H) ** 2).sum(0)
            for hd in handles:
                hd.remove()
            model.zero_grad(set_to_none=True)
        for p, rg in saved_rg:
            p.requires_grad_(rg)                          # restore original grad state
    else:
        with torch.no_grad():
            for i in range(0, min(calib_ids.size(1), 4096), 256):
                hs = model(calib_ids[:, i:i + 256].to(device), output_hidden_states=True, use_cache=False).hidden_states
                for h in hs:
                    x = h[0].float().reshape(-1, H)
                    cov += x.t() @ x
    if solver == "random":                                # CONTROL: random orthogonal basis (how much do smart solvers buy?)
        Q = torch.linalg.qr(torch.randn(H, H, device=device, dtype=torch.float32))[0].contiguous()
    elif solver == "outlier":                             # keep top massive-activation dims EXACT, PCA the complement
        K = max(1, H // 48)
        out_idx = torch.topk(cov.diag(), K).indices
        sel = torch.ones(H, dtype=torch.bool, device=device); sel[out_idx] = False
        rest = sel.nonzero(as_tuple=True)[0]
        _, vr = torch.linalg.eigh(cov[rest][:, rest]); vr = vr.flip(1)   # PCA of the non-outlier subspace
        Q = torch.zeros(H, H, device=device, dtype=torch.float32)
        Q[out_idx, torch.arange(K, device=device)] = 1.0                 # outlier unit-vector columns (kept exact)
        Q[rest.unsqueeze(1), (K + torch.arange(rest.numel(), device=device)).unsqueeze(0)] = vr
        Q = Q.contiguous()
    elif solver in ("outlier_fisher", "outlier_saliency"): # keep activation-SINK dims exact + (fisher|saliency)-rotate the rest
        K = max(1, H // 48)
        out_idx = torch.topk(act_diag, K).indices         # outliers by ACTIVATION energy (the sinks)
        sel = torch.ones(H, dtype=torch.bool, device=device); sel[out_idx] = False
        rest = sel.nonzero(as_tuple=True)[0]
        _, vr = torch.linalg.eigh(cov[rest][:, rest]); vr = vr.flip(1)   # LOSS (Fisher) eigenbasis of the non-outlier subspace
        Q = torch.zeros(H, H, device=device, dtype=torch.float32)
        Q[out_idx, torch.arange(K, device=device)] = 1.0
        Q[rest.unsqueeze(1), (K + torch.arange(rest.numel(), device=device)).unsqueeze(0)] = vr
        Q = Q.contiguous()
    else:                                                 # pca / saliency / fisher: eigenbasis of the collected cov
        evals, evecs = torch.linalg.eigh(cov)             # ascending
        Q = evecs.flip(1).contiguous()                    # columns = top directions, descending

    def rot_out(lin):                                     # weight WRITES residual (out=H): new = Q^T W
        lin.weight.data = (Q.t().to(torch.float32) @ lin.weight.data.float()).to(lin.weight.dtype)
        if lin.bias is not None:
            lin.bias.data = (Q.t().to(torch.float32) @ lin.bias.data.float()).to(lin.bias.dtype)
    def rot_in(lin):                                      # weight READS residual (in=H): new = W Q
        lin.weight.data = (lin.weight.data.float() @ Q.to(torch.float32)).to(lin.weight.dtype)
    emb.weight.data = (emb.weight.data.float() @ Q.to(torch.float32)).to(emb.weight.dtype)  # rows are residual
    for blk in layers:
        rot_in(blk.self_attn.q_proj); rot_in(blk.self_attn.k_proj); rot_in(blk.self_attn.v_proj)
        rot_out(blk.self_attn.o_proj)
        rot_in(blk.mlp.gate_proj); rot_in(blk.mlp.up_proj); rot_out(blk.mlp.down_proj)
    rot_in(head)                                          # head is untied above -> always rotate its input
    return True


class FactorizedEmbedding(nn.Module):
    """Embedding lookup factored as E ≈ U·V (ALBERT-style): store U (vocab×r) + V (r×hidden). The input
    embedding is often the LARGEST matrix in a small model; this keeps the parent's token directions in
    fewer params. forward: lookup U row -> project by V."""
    def __init__(self, U, V, padding_idx=None):
        super().__init__()
        vocab, r = U.shape
        self.emb = nn.Embedding(vocab, r, padding_idx=padding_idx)
        self.proj = nn.Linear(r, V.shape[1], bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(U); self.proj.weight.copy_(V.t())
        self.emb.weight.data = self.emb.weight.data.to(U.dtype)
        self.proj.weight.data = self.proj.weight.data.to(V.dtype)

    def forward(self, idx):
        return self.proj(self.emb(idx))


class TiedFactorizedHead(nn.Module):
    """lm_head for TIED embeddings after factorization: logits = h @ Vfᵀ @ Ufᵀ, SHARING the FactorizedEmbedding's
    factors (no params of its own — preserves the tie, so total params = body + Uf + Vf only)."""
    def __init__(self, femb):
        super().__init__(); self.femb = (femb,)          # tuple: keep out of the module tree (params counted once, at the embedding)
    def forward(self, h):
        femb = self.femb[0]
        return nn.functional.linear(nn.functional.linear(h, femb.proj.weight.t()), femb.emb.weight)


@torch.no_grad()
def factorize_embeddings(model, rank_frac, device):
    """Factorize the input embedding (and lm_head) via SVD to rank_frac × hidden. TIED case: factorize once and
    share the factors through TiedFactorizedHead (naive untie would ADD a full head copy). Returns count."""
    n = 0
    emb = model.get_input_embeddings()
    head = model.get_output_embeddings()
    tied = isinstance(head, nn.Linear) and isinstance(emb, nn.Embedding) and head.weight.data_ptr() == emb.weight.data_ptr()
    if isinstance(emb, nn.Embedding):
        W = emb.weight.float()                            # (vocab, hidden)
        r = max(1, int(rank_frac * W.shape[1]))
        if r * (W.shape[0] + W.shape[1]) < W.shape[0] * W.shape[1]:
            U, sig, Vt = torch.linalg.svd(W, full_matrices=False)
            rs = sig[:r].clamp_min(0).sqrt()
            Uf = (U[:, :r] * rs.unsqueeze(0)).to(emb.weight.dtype)
            Vf = (rs.unsqueeze(1) * Vt[:r, :]).to(emb.weight.dtype)
            femb = FactorizedEmbedding(Uf, Vf, emb.padding_idx).to(device)
            model.set_input_embeddings(femb)
            n += 1
            if tied:                                      # share factors both ways; keep the tie param-free
                model.set_output_embeddings(TiedFactorizedHead(femb).to(device))
                model.config.tie_word_embeddings = False  # the tie now lives in TiedFactorizedHead, not HF's mechanism
                return n
    if isinstance(head, nn.Linear) and head.weight.data_ptr() != emb.weight.data_ptr():
        W = head.weight.float()                           # (vocab, hidden)
        r = max(1, int(rank_frac * W.shape[1]))
        if r * (W.shape[0] + W.shape[1]) < W.shape[0] * W.shape[1]:
            U, sig, Vt = torch.linalg.svd(W, full_matrices=False)
            rs = sig[:r].clamp_min(0).sqrt()
            A = (U[:, :r] * rs.unsqueeze(0)).to(head.weight.dtype)
            B = (rs.unsqueeze(1) * Vt[:r, :]).to(head.weight.dtype)
            model.set_output_embeddings(LowRankLinear(A, B, head.bias).to(device))
            n += 1
    return n


# ---------------------------------------------------------------- FFN pruners
@torch.no_grad()
def _prune_linear_ffn(up_list, down, keep_k: int) -> bool:
    """Prune the intermediate dim of a Linear-based FFN.

    up_list: linears that MAP INTO the intermediate dim (prune their out_features rows)
             — GPT-NeoX: [dense_h_to_4h]; Llama SwiGLU: [gate_proj, up_proj].
    down:    linear that maps FROM the intermediate dim (prune its in_features cols)
             — GPT-NeoX: dense_4h_to_h; Llama: down_proj.
    """
    m = up_list[0].out_features
    if keep_k >= m:
        return False
    # importance per intermediate neuron: product of incoming + outgoing weight norms
    score = down.weight.norm(dim=0)  # over the intermediate (in) dim of down
    for up in up_list:
        score = score * up.weight.norm(dim=1)
    idx = torch.topk(score, keep_k).indices.sort().values
    dev, dt = down.weight.device, down.weight.dtype
    for up in up_list:
        n = nn.Linear(up.in_features, keep_k, bias=up.bias is not None).to(dev, dt)
        n.weight.copy_(up.weight[idx, :])
        if up.bias is not None:
            n.bias.copy_(up.bias[idx])
        _replace_child(up, n)
    nd = nn.Linear(keep_k, down.out_features, bias=down.bias is not None).to(dev, dt)
    nd.weight.copy_(down.weight[:, idx])
    if down.bias is not None:
        nd.bias.copy_(down.bias)
    _replace_child(down, nd)
    return True


@torch.no_grad()
def realloc_ffn_to_depth(model, arch, target_layers, calib_ids, device, keep_frac, n_depth):
    """Closed-form WIDTH->DEPTH reallocation (the aspect-ratio reshape done by weight transfer, not just heal):
    narrow each target layer's FFN to keep_frac (keep top-importance intermediate channels, OBS-reconstruct
    down_proj on them), and DIRECTLY TRANSFER the removed channels' weights into n_depth NEW FFN-only depth
    blocks — warm-init each new block's FFN from a slice of the POOLED removed channels (attention zeroed =
    identity). The lost width capacity is PLACED into new depth closed-form, not dropped + hoped-recovered.
    Returns (n_narrowed, [new_blocks]); caller inserts the new blocks into the stack and heals."""
    fam = arch.family
    UP = ("gate_proj", "up_proj", "c_fc", "dense_h_to_4h"); DN = ("down_proj", "dense_4h_to_h", "c_proj")
    tg = []
    for b in target_layers:
        mlp = b.mlp
        up = [(nm, getattr(mlp, nm)) for nm in UP if hasattr(mlp, nm)]
        dn = next((nm for nm in DN if hasattr(mlp, nm)), None)
        if up and dn: tg.append((mlp, up, dn, getattr(mlp, dn)))
    acc = {id(d): [] for _, _, _, d in tg}
    hk = [d.register_forward_pre_hook((lambda dd: (lambda m, i: acc[id(dd)].append(i[0].detach().float().reshape(-1, i[0].shape[-1]).cpu())))(d)) for _, _, _, d in tg]
    model.eval()
    with torch.no_grad():
        for i in range(0, min(calib_ids.size(1), 2048), 256):
            model(calib_ids[:, i:i + 256].to(device), use_cache=False)
    for h in hk: h.remove()
    pool_up = {nm: [] for nm, _ in tg[0][1]}; pool_dn = []     # removed channels pooled across layers
    n = 0
    for mlp, up, dn_name, down in tg:
        A = torch.cat(acc[id(down)], 0).to(device); inter = A.shape[1]
        keep_k = max(1, int(keep_frac * inter))
        if keep_k >= inter: continue
        W = down.weight.data.float(); Y = A @ W.t()
        score = W.norm(dim=0) * up[0][1].weight.float().norm(dim=1)
        order = torch.argsort(score, descending=True)
        kept = order[:keep_k].sort().values; removed = order[keep_k:]
        Ap = A[:, kept]; G = Ap.t() @ Ap + 1e-2 * torch.eye(keep_k, device=device)
        Wnew = torch.linalg.solve(G, Ap.t() @ Y).t()          # OBS-reconstruct down on KEPT
        dev, dt = down.weight.device, down.weight.dtype
        for uname, u in up:                                   # narrow up/gate to kept
            nu = nn.Linear(u.in_features, keep_k, bias=u.bias is not None).to(dev, dt)
            nu.weight.data.copy_(u.weight.data[kept])
            if u.bias is not None: nu.bias.data.copy_(u.bias.data[kept])
            pool_up[uname].append(u.weight.data[removed].clone())   # SAVE removed up/gate rows
            setattr(mlp, uname, nu)
        nd = nn.Linear(keep_k, down.out_features, bias=down.bias is not None).to(dev, dt)
        nd.weight.data.copy_(Wnew.to(dt))
        if down.bias is not None: nd.bias.data.copy_(down.bias.data)
        pool_dn.append(down.weight.data[:, removed].clone())  # SAVE removed down columns
        setattr(mlp, dn_name, nd); n += 1
        del A, Y, Ap, G, Wnew; torch.cuda.empty_cache()
    # build n_depth NEW FFN-only blocks, warm-init from slices of the pooled removed channels
    pu = {nm: torch.cat(v, 0) for nm, v in pool_up.items()}   # (total_removed, hidden) each
    pdn = torch.cat(pool_dn, 1)                               # (hidden, total_removed)
    tot = pdn.shape[1]; per = max(1, tot // n_depth); new_blocks = []
    import copy as _copy
    for k in range(n_depth):
        sl = slice(k * per, (k + 1) * per if k < n_depth - 1 else tot)
        blk = _copy.deepcopy(target_layers[-1]).to(device)    # template block
        bdt = next(blk.parameters()).dtype
        zero_init_residual(blk, fam)                          # start attention as identity
        mlp = blk.mlp; r = sl.stop - sl.start; Hd = model.config.hidden_size
        for uname, _ in tg[0][1]:                             # FFN warm-init from removed channels
            nu = nn.Linear(Hd, r, bias=False).to(device, bdt)
            nu.weight.data.copy_(pu[uname][sl].to(bdt)); setattr(mlp, uname, nu)
        dn_name = next(nm for nm in DN if hasattr(mlp, nm))
        nd = nn.Linear(r, Hd, bias=False).to(device, bdt)
        nd.weight.data.copy_(pdn[:, sl].to(bdt)); setattr(mlp, dn_name, nd)
        new_blocks.append(blk)
    return n, new_blocks


@torch.no_grad()
class _FFNOnlyBlock(nn.Module):
    """Adjacent realloc block carrying ONLY the FFN (no attention params). out = h + mlp(post_norm(h)) --
    FUNCTION-IDENTICAL to a zero-init full block (whose attention o_proj=0 already adds nothing), but without
    the dead q/k/v/o params. Mimics the decoder-layer interface (kwargs in, (hidden,) out) for the layer loop."""
    def __init__(self, norm, mlp):
        super().__init__(); self.post_attention_layernorm = norm; self.mlp = mlp
    def forward(self, hidden_states, *args, **kwargs):
        if isinstance(hidden_states, tuple): hidden_states = hidden_states[0]   # tolerate either calling convention
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))   # modern layers return a bare tensor


def realloc_ffn_adjacent(model, arch, target_layers, calib_ids, device, keep_frac, realloc_frac, ffn_only=False):
    """NEAR-ZERO-HEAL width->depth (function-preserving, Net2DeeperNet-style). Split each FFN's intermediate
    channels by importance: KEEP top (exact, in place) + REALLOC middle (moved to a NEW FFN-only block inserted
    IMMEDIATELY AFTER the layer -> near-exact since the new block reads r'+F_kept, off only by the small narrowed
    output) + DROP bottom (least-important = the only lost capacity). No reconstruction -- kept is exact by
    slicing, realloc is preserved by adjacency. Returns (n, new_blocks, rebuilt_layer_list)."""
    import copy as _copy
    fam = arch.family
    UP = ("gate_proj", "up_proj", "c_fc", "dense_h_to_4h"); DN = ("down_proj", "dense_4h_to_h", "c_proj")
    tg = []
    for b in target_layers:
        mlp = b.mlp; up = [(nm, getattr(mlp, nm)) for nm in UP if hasattr(mlp, nm)]
        dn = next((nm for nm in DN if hasattr(mlp, nm)), None)
        if up and dn: tg.append((b, mlp, up, dn, getattr(mlp, dn)))
    acc = {id(d): [] for _, _, _, _, d in tg}
    hk = [d.register_forward_pre_hook((lambda dd: (lambda m, i: acc[id(dd)].append(i[0].detach().float().reshape(-1, i[0].shape[-1]).cpu())))(d)) for _, _, _, _, d in tg]
    model.eval()
    with torch.no_grad():
        for i in range(0, min(calib_ids.size(1), 2048), 256):
            model(calib_ids[:, i:i + 256].to(device), use_cache=False)
    for h in hk: h.remove()
    adj = {}; n = 0; Hd = model.config.hidden_size
    for b, mlp, up, dn_name, down in tg:
        A = torch.cat(acc[id(down)], 0); inter = A.shape[1]
        keep_k = max(1, int(keep_frac * inter)); realloc_k = max(0, int(realloc_frac * inter))
        if keep_k + realloc_k > inter: realloc_k = inter - keep_k
        # Wanda channel importance: |down weight col| * ||intermediate activation|| (drop the low-importance tail)
        score = down.weight.data.float().norm(dim=0) * A.norm(dim=0).to(down.weight.device)
        order = torch.argsort(score, descending=True)
        kept = order[:keep_k].sort().values; rea = order[keep_k:keep_k + realloc_k].sort().values
        dev, dt = down.weight.device, down.weight.dtype
        for uname, u in up:                                       # narrow in place to KEPT (exact slice)
            nu = nn.Linear(u.in_features, keep_k, bias=u.bias is not None).to(dev, dt)
            nu.weight.data.copy_(u.weight.data[kept])
            if u.bias is not None: nu.bias.data.copy_(u.bias.data[kept])
            setattr(mlp, uname, nu)
        nd = nn.Linear(keep_k, down.out_features, bias=down.bias is not None).to(dev, dt)
        nd.weight.data.copy_(down.weight.data[:, kept])
        if down.bias is not None: nd.bias.data.copy_(down.bias.data)  # FFN bias stays on the in-place layer
        setattr(mlp, dn_name, nd)
        if realloc_k > 0:                                         # adjacent block carries the REALLOC channels
            if ffn_only:                                          # FFN-only adjacent: no dead attention params (same function)
                _norm = _copy.deepcopy(b.post_attention_layernorm).to(device)
                blk = _FFNOnlyBlock(_norm, _copy.deepcopy(b.mlp).to(device)); bm = blk.mlp
            else:
                blk = _copy.deepcopy(b).to(device); zero_init_residual(blk, fam); bm = blk.mlp
            for uname, u in up:
                nu2 = nn.Linear(u.in_features, realloc_k, bias=u.bias is not None).to(dev, dt)
                nu2.weight.data.copy_(u.weight.data[rea])
                if u.bias is not None: nu2.bias.data.copy_(u.bias.data[rea])
                setattr(bm, uname, nu2)
            dn2 = next(nm for nm in DN if hasattr(bm, nm))
            nd2 = nn.Linear(realloc_k, Hd, bias=down.bias is not None).to(dev, dt)
            nd2.weight.data.copy_(down.weight.data[:, rea])
            if down.bias is not None: nd2.bias.data.zero_()      # bias already added in-place -> no double-count
            setattr(bm, dn2, nd2)
            _ra = rea.to(A.device)                               # A is CPU-collected, rea on GPU -> index on A's device (device-safe)
            blk._realloc_target = (A[:, _ra].float() @ down.weight.data.float()[:, rea].t().to(A.device)).cpu()  # F_realloc (orig) for correction
            adj[id(b)] = blk; n += 1
        del A; torch.cuda.empty_cache()
    L = []                                                       # rebuild: each adjacent block right after source
    for l in arch.get_layers(model):
        L.append(l)
        if id(l) in adj: L.append(adj[id(l)])
    return n, list(adj.values()), L


@torch.no_grad()
def realloc_lstsq_correct(model, arch, new_blocks, calib_ids, device, ridge=1e-2):
    """Correct the adjacent realloc blocks: re-fit each block's down_proj via least-squares to map its ACTUAL
    (shifted-input) activation to the stored F_realloc target. EMPIRICALLY the strongest correction (12x lower
    zero-heal MSE on real models) -- the ReplaceMe-style output re-fit, NOT input-shift inversion."""
    DN = ("down_proj", "dense_4h_to_h", "c_proj")
    blks = [b for b in new_blocks if hasattr(b, "_realloc_target")]
    downs = {id(b): next(getattr(b.mlp, nm) for nm in DN if hasattr(b.mlp, nm)) for b in blks}
    acc = {id(b): [] for b in blks}
    hk = [downs[id(b)].register_forward_pre_hook((lambda bb: (lambda m, i: acc[id(bb)].append(i[0].detach().float().reshape(-1, i[0].shape[-1]).cpu())))(b)) for b in blks]
    model.eval()
    with torch.no_grad():
        for i in range(0, min(calib_ids.size(1), 2048), 256):
            model(calib_ids[:, i:i + 256].to(device), use_cache=False)
    for h in hk: h.remove()
    for b in blks:
        d = downs[id(b)]; A = torch.cat(acc[id(b)], 0); tgt = b._realloc_target.float()[:A.shape[0]]
        G = A.t() @ A + ridge * torch.eye(A.shape[1])
        Wn = torch.linalg.solve(G, A.t() @ tgt).t().to(d.weight.dtype).to(d.weight.device)
        d.weight.data.copy_(Wn)
        if d.bias is not None: d.bias.data.zero_()
        del b._realloc_target
    return len(blks)


@torch.no_grad()
def zero_init_residual(block, family):
    """LLaMA-Pro (2401.02415) identity init: zero the residual-WRITING projections (attn o_proj + FFN
    down_proj) so the block adds 0 to the residual -> starts as an exact no-op (identity). Combined with a
    heal, the block is EMPTY CAPACITY the optimizer fills by REALLOCATING function into it (vs a duplicate
    that starts redundant). 'Warm-init with nothing'."""
    if family == "gpt_neox":
        for m in (block.attention.dense, block.mlp.dense_4h_to_h):
            m.weight.zero_();  m.bias is not None and m.bias.zero_()
    else:
        for m in (block.self_attn.o_proj, block.mlp.down_proj):
            m.weight.zero_()
            if getattr(m, "bias", None) is not None: m.bias.zero_()


def _replace_child(old: nn.Module, new: nn.Module):
    """Swap a submodule in-place by mutating its parent via the registered ref."""
    old._streamliner_new = new  # marker; actual swap done by caller-held parent setattr


@torch.no_grad()
def _prune_gptneox(block, keep_k: int) -> bool:
    mlp = block.mlp
    ok = _prune_linear_ffn([mlp.dense_h_to_4h], mlp.dense_4h_to_h, keep_k)
    if ok:
        mlp.dense_h_to_4h = mlp.dense_h_to_4h._streamliner_new
        mlp.dense_4h_to_h = mlp.dense_4h_to_h._streamliner_new
    return ok


@torch.no_grad()
def _prune_llama(block, keep_k: int) -> bool:
    mlp = block.mlp
    ok = _prune_linear_ffn([mlp.gate_proj, mlp.up_proj], mlp.down_proj, keep_k)
    if ok:
        mlp.gate_proj = mlp.gate_proj._streamliner_new
        mlp.up_proj = mlp.up_proj._streamliner_new
        mlp.down_proj = mlp.down_proj._streamliner_new
    return ok


@torch.no_grad()
def _prune_gpt2(block, keep_k: int) -> bool:
    from transformers.pytorch_utils import Conv1D
    cfc, cproj = block.mlp.c_fc, block.mlp.c_proj  # Conv1D: weight (in, out)
    m = cfc.weight.shape[1]
    if keep_k >= m:
        return False
    score = cfc.weight.norm(dim=0) * cproj.weight.norm(dim=1)
    idx = torch.topk(score, keep_k).indices.sort().values
    nfc = Conv1D(keep_k, cfc.weight.shape[0]).to(cfc.weight.device, cfc.weight.dtype)
    nfc.weight.copy_(cfc.weight[:, idx]); nfc.bias.copy_(cfc.bias[idx])
    nproj = Conv1D(cproj.weight.shape[1], keep_k).to(cproj.weight.device, cproj.weight.dtype)
    nproj.weight.copy_(cproj.weight[idx, :]); nproj.bias.copy_(cproj.bias)
    block.mlp.c_fc, block.mlp.c_proj = nfc, nproj
    return True


@torch.no_grad()
@torch.no_grad()
def obs_gqa_prune(model, arch, layers, calib_ids, device, keep_q_per_group, metric="wanda"):
    """GQA-aware QUERY-head pruning (Qwen/Llama) — the OBS recipe ported from gpt_neox to GQA. Drops low-importance
    QUERY heads UNIFORMLY per KV group (keeps all KV heads + valid GQA: new grp = keep_q_per_group) + RECONSTRUCTS
    o_proj via per-layer least-squares (surviving heads absorb the removed). Matches the residual by trimming the
    over-sized attention. metric: 'norm'(weight |W|, the old prune_gqa_heads signal) | 'energy'(activation) |
    'wanda'(act*|W|). The HEAD analog of the width sweep — activation+reconstruction predicted to beat weight-norm.
    Keeps head_dim fixed (no RoPE desync). Uniform head count across layers (each layer keeps its OWN heads)."""
    if arch.family not in ("llama", "mistral", "qwen2", "qwen3"):
        print(f"[obs_gqa] Llama-family only (got {arch.family}); skipping"); return 0
    cfg = model.config
    Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // Hq
    grp = Hq // Hkv
    if keep_q_per_group >= grp or keep_q_per_group < 1:
        return 0
    tgt = [(b.self_attn, b.self_attn.o_proj) for b in layers]
    acc = {id(o): [] for _, o in tgt}
    hooks = [o.register_forward_pre_hook(
        lambda m, inp, o=o: acc[id(o)].append(inp[0].detach().float().reshape(-1, inp[0].shape[-1])[:4096].cpu()))
        for _, o in tgt]
    model.eval()
    for i in range(0, min(calib_ids.size(1), 2048), 256):
        model(calib_ids[:, i:i + 256].to(device), use_cache=False)
    for h in hooks:
        h.remove()
    n = 0
    for (attn, o) in tgt:
        A = torch.cat(acc[id(o)], 0).to(device)                  # (N, Hq*hd) concatenated query-head outputs
        W = o.weight.data.float()                                # (hidden, Hq*hd)
        Y = A @ W.t()                                            # full o_proj output (LS reconstruction target)
        keep_heads = []
        for g in range(Hkv):                                     # keep top heads PER GROUP (uniform -> valid GQA)
            hs = list(range(g * grp, (g + 1) * grp)); sc = []
            for hh in hs:
                c = slice(hh * hd, (hh + 1) * hd)
                if metric == "norm":     s = W[:, c].norm()
                elif metric == "energy": s = A[:, c].norm()
                else:                    s = A[:, c].norm() * W[:, c].norm()   # wanda
                sc.append(s.item())
            top = sorted(range(grp), key=lambda k: -sc[k])[:keep_q_per_group]
            keep_heads += [hs[k] for k in sorted(top)]
        qcols = torch.cat([torch.arange(hh * hd, (hh + 1) * hd) for hh in keep_heads]).to(device)
        Ap = A[:, qcols]                                         # LS-reconstruct o_proj over kept head columns
        G = Ap.t() @ Ap + 1e-2 * torch.eye(qcols.numel(), device=device)
        Wnew = torch.linalg.solve(G, Ap.t() @ Y).t()            # (hidden, keep*hd)
        dev, dt = o.weight.device, o.weight.dtype
        nq = nn.Linear(attn.q_proj.in_features, len(qcols), bias=attn.q_proj.bias is not None).to(dev, dt)
        nq.weight.data.copy_(attn.q_proj.weight.data[qcols])     # q_proj rows == o_proj cols (same head layout)
        if attn.q_proj.bias is not None: nq.bias.data.copy_(attn.q_proj.bias.data[qcols])
        attn.q_proj = nq
        no = nn.Linear(len(qcols), o.out_features, bias=o.bias is not None).to(dev, dt)
        no.weight.data.copy_(Wnew.to(dt))
        if o.bias is not None: no.bias.data.copy_(o.bias.data)
        attn.o_proj = no
        new_Hq = keep_q_per_group * Hkv
        for attr in ("num_heads", "num_attention_heads"):
            if hasattr(attn, attr): setattr(attn, attr, new_Hq)
        if hasattr(attn, "config"): attn.config.num_attention_heads = new_Hq
        del A, Y, Ap, G, Wnew; torch.cuda.empty_cache()
        n += 1
    cfg.num_attention_heads = keep_q_per_group * Hkv             # k/v heads unchanged (GQA preserved)
    print(f"[surgery] obs_gqa_prune ({metric}): query heads {Hq} -> {keep_q_per_group * Hkv} "
          f"({grp}->{keep_q_per_group} per KV group, {Hkv} KV heads kept) + LS o_proj recon, {n} layers")
    return n


def prune_gqa_heads(block, keep_kv_groups: int, config) -> bool:
    """GQA-aware attention-head pruning for Llama-family (qwen2/qwen3/llama/mistral). Prunes whole
    KV GROUPS (a KV head + its associated query heads) by importance, preserving GQA structure.
    Done the RIGHT way: keep the most important groups (vs a naive uniform target). Composes with
    depth/FFN/residual. Keeps head_dim fixed (no RoPE desync)."""
    a = block.self_attn
    Hq = config.num_attention_heads
    Hkv = config.num_key_value_heads
    hd = getattr(config, "head_dim", None) or config.hidden_size // Hq
    grp = Hq // Hkv                                  # query heads per KV group
    if keep_kv_groups >= Hkv or keep_kv_groups < 1:
        return False
    # importance per KV group = norm of its k/v rows * mean norm of its query heads' rows + o cols
    qW = a.q_proj.weight.data.view(Hkv, -1).float()             # (kv, grp*hd*hidden)
    kW = a.k_proj.weight.data.view(Hkv, -1).float()
    vW = a.v_proj.weight.data.view(Hkv, -1).float()
    oW = a.o_proj.weight.data.view(-1, Hkv, grp * hd).permute(1, 0, 2).reshape(Hkv, -1).float()
    score = qW.norm(dim=1) + kW.norm(dim=1) + vW.norm(dim=1) + oW.norm(dim=1)  # per-KV-group importance
    keep = torch.topk(score, keep_kv_groups).indices.sort().values
    qrows = torch.cat([torch.arange(g * grp * hd, (g + 1) * grp * hd) for g in keep])  # query rows kept
    kvrows = torch.cat([torch.arange(g * hd, (g + 1) * hd) for g in keep])             # kv rows kept
    dev, dt = a.q_proj.weight.device, a.q_proj.weight.dtype

    def _rows(lin, rows):
        n = nn.Linear(lin.in_features, len(rows), bias=lin.bias is not None).to(dev, dt)
        n.weight.data.copy_(lin.weight.data[rows]);
        if lin.bias is not None: n.bias.data.copy_(lin.bias.data[rows])
        return n
    a.q_proj = _rows(a.q_proj, qrows); a.k_proj = _rows(a.k_proj, kvrows); a.v_proj = _rows(a.v_proj, kvrows)
    no = nn.Linear(len(qrows), a.o_proj.out_features, bias=a.o_proj.bias is not None).to(dev, dt)
    no.weight.data.copy_(a.o_proj.weight.data[:, qrows])
    if a.o_proj.bias is not None: no.bias.data.copy_(a.o_proj.bias.data)
    a.o_proj = no
    # per-module head counts (HF reads config at forward; set on the module if present)
    new_q = keep_kv_groups * grp
    for attr, val in [("num_heads", new_q), ("num_key_value_heads", keep_kv_groups),
                      ("num_attention_heads", new_q), ("config", None)]:
        if attr != "config" and hasattr(a, attr):
            setattr(a, attr, val)
    return True


@torch.no_grad()
@torch.no_grad()
def greedy_hetero_allocate(model, arch, layers, bi, calib_ids, device, target_frac, protect_edges=1):
    """GREEDY HETEROGENEOUS allocator (the per-location axis-choice idea): score every candidate reduction by
    COST-PER-PARAM-SAVED and greedily pick the cheapest until `target_frac` of params is removed. Candidates:
      - DELETE layer i  -> cost = BI[i] (block influence; low = redundant), saves the whole layer
      - NARROW layer i's FFN to width w -> cost = BI[i] * energy_dropped(w), saves (fw-w)*3H
    Each layer gets AT MOST ONE axis (deleting voids its FFN option) -> axes are DISTRIBUTED across layers,
    never STACKED on the same layer -> avoids the uniform-multi-axis compounding (73.7<80). FFN narrowing is
    applied in-place here (before deletion, so kept layers survive narrowed); returns the delete-set for the
    existing deletion+ReplaceMe path. Returns (removed_set, ffn_plan={layer_idx: keep_width})."""
    import math
    n = len(layers); H = model.config.hidden_size
    total = sum(p.numel() for p in model.parameters()); target = target_frac * total
    # per-layer FFN intermediate energy curve (cumulative singular-value energy of the down-proj input)
    acts = {i: [] for i in range(n)}
    hooks = [arch.ffn_down(layers[i]).register_forward_pre_hook(
        lambda m, inp, i=i: acts[i].append(inp[0].detach().float().reshape(-1, inp[0].shape[-1])[:4096])) for i in range(n)]
    model(calib_ids[:, :1024].to(device), use_cache=False)
    for h in hooks: h.remove()
    fe = {}
    for i in range(n):
        A = torch.cat(acts[i], 0); A = A - A.mean(0, keepdim=True)
        s = torch.linalg.svdvals(A) ** 2
        fe[i] = (torch.cumsum(s, 0) / s.sum()).cpu()
    def lp(b): return sum(p.numel() for p in b.parameters())
    cands = []                                            # (cost_per_param, params_saved, kind, layer, keep_width)
    for i in range(n):
        if protect_edges <= i < n - protect_edges:        # protect layer 0 + last (never delete)
            cands.append((bi[i] / lp(layers[i]), lp(layers[i]), "del", i, None))
        fw = arch.ffn_width(layers[i])
        for frac in (0.75, 0.5, 0.25):
            w = max(64, 8 * math.ceil(int(fw * frac) / 8))
            if w >= fw: continue
            saved = (fw - w) * 3 * H                       # gate+up (in cols) + down (out rows) ~ 3H per neuron
            ed = 1 - fe[i][min(w, len(fe[i]) - 1)].item()  # FFN energy dropped at rank w
            cands.append(((bi[i] * ed + 1e-9) / saved, saved, "ffn", i, w))
    cands.sort(key=lambda c: c[0])
    used = set(); removed = set(); ffn_plan = {}; dropped = 0.0
    for cpp, saved, kind, i, w in cands:                  # greedy cheapest-first, one axis per layer
        if i in used or dropped >= target: continue
        if kind == "del": removed.add(i)
        else: ffn_plan[i] = w
        used.add(i); dropped += saved
    for i, w in ffn_plan.items():                         # apply FFN narrowing in-place (kept layers survive narrowed)
        arch.prune_ffn(layers[i], w)
    print(f"[hetero] target {target_frac:.0%}: DELETE {sorted(removed)} | NARROW-FFN "
          f"{{{', '.join(f'{i}:{w}' for i,w in sorted(ffn_plan.items()))}}} | dropped ~{dropped/total:.1%} params")
    return removed, ffn_plan


def cba_allocate(model, arch, kept_layers, calib_ids, device, ffn_energy=None, hidden_energy=None):
    """Training-free, measurement-driven PER-LAYER allocation (our differentiator from Minitron's
    NAS-style target search). Instead of forcing a uniform width, MEASURE what each layer can give up:
      - per-layer FFN width = effective rank of its intermediate activations at `ffn_energy` energy
      - residual width = #hidden channels holding `hidden_energy` of the global activation energy
    The total compression EMERGES from the quality thresholds rather than being forced. Returns a dict.
    """
    import math
    report = {"ffn_widths": {}, "hidden": None}
    # --- per-layer FFN effective width (hook each block's down-proj input = the intermediate) ---
    if ffn_energy is not None:
        acts = {id(b): [] for b in kept_layers}
        hooks = [arch.ffn_down(b).register_forward_pre_hook(
            lambda m, inp, key=id(b): acts[key].append(inp[0].detach().float().reshape(-1, inp[0].shape[-1])[:4096].cpu()))
            for b in kept_layers]
        model(calib_ids, use_cache=False)
        for h in hooks:
            h.remove()
        for b in kept_layers:
            A = torch.cat(acts[id(b)], 0); A = A - A.mean(0, keepdim=True)
            s = torch.linalg.svdvals(A.float()) ** 2
            r = int((torch.cumsum(s, 0) / s.sum() < ffn_energy).sum().item()) + 1
            w = min(arch.ffn_width(b), max(64, 8 * math.ceil(r / 8)))   # multiple of 8, capped at current
            report["ffn_widths"][id(b)] = w
            arch.prune_ffn(b, w)
    # --- residual width by global channel energy (-> a single keep_h, then prune_hidden_width) ---
    if hidden_energy is not None and arch.family in ("llama", "mistral", "qwen2", "qwen3"):
        H = model.config.hidden_size
        hs = model(calib_ids, output_hidden_states=True, use_cache=False).hidden_states
        imp = torch.zeros(H, device=device)
        for h in hs:
            imp += (h.float() ** 2).sum(dim=(0, 1))
        s = torch.sort(imp, descending=True).values
        keep_h = int((torch.cumsum(s, 0) / s.sum() < hidden_energy).sum().item()) + 1
        keep_h = max(256, 8 * math.ceil(keep_h / 8))
        report["hidden"] = keep_h
        if keep_h < H:
            prune_hidden_width(model, arch, keep_h, calib_ids, device)
    return report


@torch.no_grad()
@torch.no_grad()
def prune_hidden_perlayer(model, arch, keep_h, calib_ids, device):
    """PER-LAYER (advanced) fold absorption (GPTQ/SparseGPT-grade): each layer's readers fold their OWN
    dropped->kept regression M_l, computed from that layer's ACTUAL projection inputs -- SEPARATE M for
    attention (q/k/v) vs MLP (gate/up) since they read DIFFERENT residuals (pre- vs post-attention). Fixes
    the global M's cross-layer + cross-read-point mismatch that hurt at aggressive compression. Residual->keep_h."""
    if arch.family not in ("llama", "mistral", "qwen2", "qwen3"):
        raise ValueError(f"prune_hidden_perlayer: Llama-family only, got {arch.family}")
    H = model.config.hidden_size; head_dim = getattr(model.config, "head_dim", None) or H // model.config.num_attention_heads
    model.eval(); layers = arch.get_layers(model)
    hs = model(calib_ids, output_hidden_states=True, use_cache=False).hidden_states
    imp = torch.zeros(H, device=device)
    for h in hs: imp += (h.float().reshape(-1, H) ** 2).sum(0)
    idx = torch.topk(imp, keep_h).indices.sort().values
    drop = torch.tensor([i for i in range(H) if i not in set(idx.tolist())], device=device)
    wt = None                                            # LOSS-AWARE: weight the solve by per-token teacher loss (hard tokens matter more)
    if getattr(model.config, "_absorb_lossw", False):
        ci = calib_ids[:, :1024].to(device)
        lg = model(ci, use_cache=False).logits.float()
        ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), ci[:, 1:].reshape(-1), reduction="none").reshape(ci.size(0), -1)
        wfull = torch.cat([ce, ce.mean(1, keepdim=True)], 1).reshape(-1)   # per-token CE, last position = mean
        wt = (wfull / wfull.mean().clamp_min(1e-6)).clamp(0.1, 10.0)[:4096]  # normalized + clamped, aligned to the [:4096] cap
    acc = {}
    def mk(key):
        def hook(m, inp): acc.setdefault(key, []).append(inp[0].detach().float().reshape(-1, inp[0].shape[-1])[:4096])
        return hook
    handles = []
    for li, blk in enumerate(layers):
        handles.append(blk.self_attn.q_proj.register_forward_pre_hook(mk(("a", li))))
        handles.append(blk.mlp.gate_proj.register_forward_pre_hook(mk(("m", li))))
    model(calib_ids[:, :1024].to(device), use_cache=False)
    for h in handles: h.remove()
    def Mfor(key):                                       # per-layer per-read-point regression M (dropped<-kept)
        A = torch.cat(acc[key], 0); Ck = A[:, idx]; Cd = A[:, drop]
        if wt is not None:                               # WEIGHTED least squares: focus reconstruction on loss-critical tokens
            w = wt[:Ck.size(0)].unsqueeze(1); Ckw = Ck * w
            return torch.linalg.solve(Ckw.t() @ Ck + 1e-2 * torch.eye(keep_h, device=device), Ckw.t() @ Cd).t()
        return torch.linalg.solve(Ck.t() @ Ck + 1e-2 * torch.eye(keep_h, device=device), Ck.t() @ Cd).t()
    def in_M(lin, M):                                    # fold this reader's OWN M
        W = lin.weight.data.float()
        n = nn.Linear(keep_h, lin.out_features, bias=lin.bias is not None).to(lin.weight.device, lin.weight.dtype)
        n.weight.data.copy_((W[:, idx] + W[:, drop] @ M).to(lin.weight.dtype))
        if lin.bias is not None: n.bias.data.copy_(lin.bias.data)
        return n
    def out_(lin):
        n = nn.Linear(lin.in_features, keep_h, bias=lin.bias is not None).to(lin.weight.device, lin.weight.dtype)
        n.weight.data.copy_(lin.weight.data[idx, :])
        if lin.bias is not None: n.bias.data.copy_(lin.bias.data[idx])
        return n
    emb = model.get_input_embeddings(); tied = bool(getattr(model.config, "tie_word_embeddings", False))
    emb.weight = nn.Parameter(emb.weight.data[:, idx]); emb.embedding_dim = keep_h
    for li, layer in enumerate(layers):
        Ma, Mm = Mfor(("a", li)), Mfor(("m", li))
        a = layer.self_attn; a.q_proj, a.k_proj, a.v_proj = in_M(a.q_proj, Ma), in_M(a.k_proj, Ma), in_M(a.v_proj, Ma); a.o_proj = out_(a.o_proj)
        m = layer.mlp; m.gate_proj, m.up_proj, m.down_proj = in_M(m.gate_proj, Mm), in_M(m.up_proj, Mm), out_(m.down_proj)
        for mod in layer.modules():
            w = getattr(mod, "weight", None)
            if w is not None and w.dim() == 1 and w.numel() == H: mod.weight = nn.Parameter(w.data[idx])
    extras = arch.trainable_extra(model); fnorm, head = extras[0], extras[-1]
    if getattr(fnorm, "weight", None) is not None and fnorm.weight.numel() == H: fnorm.weight = nn.Parameter(fnorm.weight.data[idx])
    if tied: head.weight = emb.weight
    elif head.weight.shape[-1] == H: head.weight = nn.Parameter(head.weight.data[:, idx])
    model.config.hidden_size = keep_h; model.config.head_dim = head_dim
    print(f"[surgery] perlayer-absorb: {keep_h} kept, per-layer M (attn+mlp SEPARATE) folded into readers")
    return True


@torch.no_grad()
def prune_hidden_seqsweep(model, arch, keep_h, calib_ids, device):
    """PROPER SparseGPT single-sweep sequential: ONE forward pass through the layers. For each layer (in order),
    re-solve its readers from the ACTUAL drifted (narrowed) residual to the original teacher output, then RUN the
    corrected layer to PROPAGATE the correction forward so downstream layers see it. Errors don't accumulate."""
    if arch.family not in ("qwen2", "qwen3", "llama", "mistral"):
        raise ValueError(f"prune_hidden_seqsweep: Llama-family only, got {arch.family}")
    H = model.config.hidden_size; head_dim = getattr(model.config, "head_dim", None) or H // model.config.num_attention_heads
    model.eval(); layers = list(arch.get_layers(model))
    ci = calib_ids[:, :2048].to(device); B, S = ci.shape
    def readers(blk): return [("q", blk.self_attn.q_proj), ("k", blk.self_attn.k_proj), ("v", blk.self_attn.v_proj),
                              ("g", blk.mlp.gate_proj), ("u", blk.mlp.up_proj)]
    tgt = {}                                              # capture ORIGINAL reader outputs (targets) + residual energy
    def ohk(key):
        def h(m, inp, out): tgt[key] = out.detach().float().reshape(-1, out.shape[-1])
        return h
    hh = [lin.register_forward_hook(ohk((li, nm))) for li, blk in enumerate(layers) for nm, lin in readers(blk)]
    out0 = model(ci, output_hidden_states=True, use_cache=False)
    for h in hh: h.remove()
    imp = torch.zeros(H, device=device)
    for hsr in out0.hidden_states: imp += (hsr.float().reshape(-1, H) ** 2).sum(0)
    idx = torch.topk(imp, keep_h).indices.sort().values
    inner = model.model if hasattr(model, "model") else model
    emb = model.get_input_embeddings(); tied = bool(getattr(model.config, "tie_word_embeddings", False))
    if tied: model.get_output_embeddings().weight = nn.Parameter(emb.weight.data.clone()); model.config.tie_word_embeddings = False
    emb.weight = nn.Parameter(emb.weight.data[:, idx]); emb.embedding_dim = keep_h
    def slice_out(lin):                                  # writers o/down: slice output rows to keep_h
        n = nn.Linear(lin.in_features, keep_h, bias=lin.bias is not None).to(lin.weight.device, lin.weight.dtype)
        n.weight.data.copy_(lin.weight.data[idx, :])
        if lin.bias is not None: n.bias.data.copy_(lin.bias.data[idx])
        return n
    for blk in layers:
        blk.self_attn.o_proj = slice_out(blk.self_attn.o_proj); blk.mlp.down_proj = slice_out(blk.mlp.down_proj)
        for mod in blk.modules():
            w = getattr(mod, "weight", None)
            if w is not None and w.dim() == 1 and w.numel() == H: mod.weight = nn.Parameter(w.data[idx])
    extras = arch.trainable_extra(model); fnorm, head = extras[0], extras[-1]
    if getattr(fnorm, "weight", None) is not None and fnorm.weight.numel() == H: fnorm.weight = nn.Parameter(fnorm.weight.data[idx])
    oh = model.get_output_embeddings()                   # lm_head reads the final residual -> slice to keep_h
    if oh is not None and getattr(oh, "weight", None) is not None and oh.weight.shape[-1] == H:
        oh.weight = nn.Parameter(oh.weight.data[:, idx])
    model.config.hidden_size = keep_h; model.config.head_dim = head_dim
    def resolve(lin, X, Y):                              # re-solve keep_h->out reader: W' = lstsq(drifted X -> teacher Y)
        X = X.float()                                    # lstsq in fp32 (X arrives bf16)
        n = min(X.size(0), Y.size(0)); X, Y = X[:n], Y[:n]
        Wp = torch.linalg.solve(X.t() @ X + 1e-2 * torch.eye(keep_h, device=device), X.t() @ Y).t()
        new = nn.Linear(keep_h, Y.size(1), bias=lin.bias is not None).to(lin.weight.device, lin.weight.dtype)
        new.weight.data.copy_(Wp.to(lin.weight.dtype))
        if lin.bias is not None: new.bias.data.copy_(lin.bias.data)
        return new
    dt = emb.weight.dtype                                # SINGLE SWEEP: propagate the corrected residual layer by layer
    res = emb(ci); pos_ids = torch.arange(S, device=device)[None, :]
    pos_emb = inner.rotary_emb(res, pos_ids)
    mask = torch.triu(torch.full((S, S), float("-inf"), device=device, dtype=torch.float32), 1)[None, None].to(dt)
    for li, blk in enumerate(layers):
        za = blk.input_layernorm(res).reshape(-1, keep_h)
        blk.self_attn.q_proj = resolve(blk.self_attn.q_proj, za, tgt[(li, "q")].to(device))
        blk.self_attn.k_proj = resolve(blk.self_attn.k_proj, za, tgt[(li, "k")].to(device))
        blk.self_attn.v_proj = resolve(blk.self_attn.v_proj, za, tgt[(li, "v")].to(device))
        res = res + blk.self_attn(blk.input_layernorm(res), position_embeddings=pos_emb, attention_mask=mask)[0]
        zm = blk.post_attention_layernorm(res).reshape(-1, keep_h)
        blk.mlp.gate_proj = resolve(blk.mlp.gate_proj, zm, tgt[(li, "g")].to(device))
        blk.mlp.up_proj = resolve(blk.mlp.up_proj, zm, tgt[(li, "u")].to(device))
        res = res + blk.mlp(blk.post_attention_layernorm(res))
    print(f"[surgery] seqsweep-absorb: {keep_h} kept, single forward-sweep (re-solve readers from drifted residual + propagate)")
    return True


@torch.no_grad()
def prune_hidden_sequential(model, arch, keep_h, calib_ids, device, iters=2):
    """[superseded by prune_hidden_seqsweep] Jacobi iterative variant -- DIVERGES (all-at-once re-solve oscillates).
    SEQUENTIAL (SparseGPT-style) absorption: after the initial per-layer fold, ITERATIVELY re-solve each
    reader (q/k/v/gate/up) to map its ACTUAL drifted (narrowed) input to the ORIGINAL teacher reader output.
    Each pass propagates upstream corrections -> errors from narrowing don't ACCUMULATE down the network
    (Jacobi iteration converging to the sequential fixed point). Re-solves the full keep_h->out reader weight."""
    if arch.family not in ("llama", "mistral", "qwen2", "qwen3"):
        raise ValueError(f"prune_hidden_sequential: Llama-family only, got {arch.family}")
    model.eval(); layers = list(arch.get_layers(model))
    ci = calib_ids[:, :4096].to(device)   # need >> keep_h tokens to avoid underdetermined re-solve
    def readers(blk): return [("q", blk.self_attn.q_proj), ("k", blk.self_attn.k_proj), ("v", blk.self_attn.v_proj),
                              ("g", blk.mlp.gate_proj), ("u", blk.mlp.up_proj)]
    tgt = {}                                              # 1. capture ORIGINAL teacher reader OUTPUTS (the targets)
    def ohk(key):
        def h(m, inp, out): tgt[key] = out.detach().float().reshape(-1, out.shape[-1])[:4096].cpu()
        return h
    hh = [lin.register_forward_hook(ohk((li, nm))) for li, blk in enumerate(layers) for nm, lin in readers(blk)]
    model(ci, use_cache=False)
    for h in hh: h.remove()
    prune_hidden_perlayer(model, arch, keep_h, calib_ids, device)   # 2. initial per-layer fold (narrows + folds)
    layers = list(arch.get_layers(model))
    for it in range(iters):                              # 3. refine: re-solve each reader from its DRIFTED input
        ia = {}
        def ihk(key):
            def h(m, inp): ia[key] = inp[0].detach().float().reshape(-1, inp[0].shape[-1])[:4096]
            return h
        hh = [lin.register_forward_pre_hook(ihk((li, nm))) for li, blk in enumerate(layers) for nm, lin in readers(blk)]
        model(ci, use_cache=False)
        for h in hh: h.remove()
        for li, blk in enumerate(layers):
            for nm, lin in readers(blk):
                X = ia[(li, nm)]; Y = tgt[(li, nm)].to(device); n = min(X.size(0), Y.size(0))
                X, Y = X[:n], Y[:n]
                Wp = torch.linalg.solve(X.t() @ X + 1e-2 * torch.eye(keep_h, device=device), X.t() @ Y).t()
                lin.weight.data.copy_(Wp.to(lin.weight.dtype))
        del ia; torch.cuda.empty_cache() if device.type == "cuda" else None
    print(f"[surgery] sequential-absorb: {keep_h} kept, {iters} refinement passes (re-solve readers from drifted inputs)")
    return True


@torch.no_grad()
def prune_hidden_optimal(model, arch, keep_h, r, calib_ids, device):
    """OPTIMAL absorption = FOLD (predictable) + ADD-ORTHOGONAL (residual). M = dropped~kept regression folds the
    PREDICTABLE part into kept; the r new channels capture the ORTHOGONAL residual via Ur = top-r of the SCHUR
    COMPLEMENT Cdd - Cdk Ckk^-1 Ckd (dropped covariance CONDITIONED on kept = the independent directions/outliers).
    Reader cols [W[:,idx]+W[:,drop]@M | W[:,drop]@Ur]; writer rows [W[idx] ; Ur^T@(W[drop]-M@W[idx])] (M-coupled).
    New hidden = keep_h + r. Captures BOTH the redundant (fold) AND independent (add) parts of the dropped channels."""
    if arch.family not in ("llama", "mistral", "qwen2", "qwen3"):
        raise ValueError(f"prune_hidden_optimal: Llama-family only, got {arch.family}")
    H = model.config.hidden_size; head_dim = getattr(model.config, "head_dim", None) or H // model.config.num_attention_heads
    model.eval()
    def fold(norm, readers):
        g = norm.weight.data.float()
        for lin in readers: lin.weight.data = (lin.weight.data.float() * g.unsqueeze(0)).to(lin.weight.dtype)
        norm.weight.data = torch.ones_like(norm.weight.data)
    extras = arch.trainable_extra(model); fnorm, head = extras[0], extras[-1]
    emb = model.get_input_embeddings(); tied = bool(getattr(model.config, "tie_word_embeddings", False))
    if tied and head.weight is emb.weight:
        head.weight = nn.Parameter(emb.weight.data.clone()); model.config.tie_word_embeddings = False
    for blk in arch.get_layers(model):
        fold(blk.input_layernorm, [blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj])
        fold(blk.post_attention_layernorm, [blk.mlp.gate_proj, blk.mlp.up_proj])
    fold(fnorm, [head])
    hs = model(calib_ids, output_hidden_states=True, use_cache=False).hidden_states
    Cov = torch.zeros(H, H, device=device, dtype=torch.float32); imp = torch.zeros(H, device=device)
    for h in hs:
        x = h.float().reshape(-1, H); Cov += x.t() @ x; imp += (x ** 2).sum(0)
    idx = torch.topk(imp, keep_h).indices.sort().values
    drop = torch.tensor([i for i in range(H) if i not in set(idx.tolist())], device=device)
    Ckk = Cov[idx][:, idx] + 1e-2 * torch.eye(keep_h, device=device); Ckd = Cov[idx][:, drop]; Cdd = Cov[drop][:, drop]
    M = torch.linalg.solve(Ckk, Ckd).t().contiguous()              # [n_drop, keep_h]: dropped <- kept regression (predictable)
    Schur = Cdd - Ckd.t() @ torch.linalg.solve(Ckk, Ckd)           # [n_drop, n_drop]: dropped cov CONDITIONED on kept
    Ur = torch.linalg.eigh(Schur)[1][:, -r:].contiguous()          # [n_drop, r]: top-r ORTHOGONAL/independent directions
    newH = keep_h + r
    def reader(lin):                                               # [W[:,idx]+W[:,drop]@M | W[:,drop]@Ur]
        W = lin.weight.data.float()
        n = nn.Linear(newH, lin.out_features, bias=lin.bias is not None).to(lin.weight.device, lin.weight.dtype)
        n.weight.data.copy_(torch.cat([W[:, idx] + W[:, drop] @ M, W[:, drop] @ Ur], 1).to(lin.weight.dtype))
        if lin.bias is not None: n.bias.data.copy_(lin.bias.data)
        return n
    def writer(lin):                                              # [W[idx] ; Ur^T@(W[drop]-M@W[idx])]
        W = lin.weight.data.float()
        crows = Ur.t() @ (W[drop, :] - M @ W[idx, :])
        n = nn.Linear(lin.in_features, newH, bias=lin.bias is not None).to(lin.weight.device, lin.weight.dtype)
        n.weight.data.copy_(torch.cat([W[idx, :], crows], 0).to(lin.weight.dtype))
        if lin.bias is not None:
            b = lin.bias.data.float(); n.bias.data.copy_(torch.cat([b[idx], Ur.t() @ (b[drop] - M @ b[idx])]).to(lin.weight.dtype))
        return n
    E = emb.weight.data.float()
    emb.weight = nn.Parameter(torch.cat([E[:, idx], (E[:, drop] - E[:, idx] @ M.t()) @ Ur], 1).to(emb.weight.dtype)); emb.embedding_dim = newH
    for layer in arch.get_layers(model):
        a = layer.self_attn; a.q_proj, a.k_proj, a.v_proj, a.o_proj = reader(a.q_proj), reader(a.k_proj), reader(a.v_proj), writer(a.o_proj)
        m = layer.mlp; m.gate_proj, m.up_proj, m.down_proj = reader(m.gate_proj), reader(m.up_proj), writer(m.down_proj)
        for mod in layer.modules():
            w = getattr(mod, "weight", None)
            if w is not None and w.dim() == 1 and w.numel() == H:
                mod.weight = nn.Parameter(torch.ones(newH, device=w.device, dtype=w.dtype))
    if getattr(fnorm, "weight", None) is not None and fnorm.weight.numel() == H:
        fnorm.weight = nn.Parameter(torch.ones(newH, device=fnorm.weight.device, dtype=fnorm.weight.dtype))
    nh = reader(head); model.set_output_embeddings(nh) if hasattr(model, "set_output_embeddings") else setattr(model, "lm_head", nh)
    model.config.hidden_size = newH; model.config.head_dim = head_dim
    print(f"[surgery] optimal-width: {keep_h} raw +fold + {r} ORTHOGONAL(Schur) = {newH}; captures predictable AND independent dropped info")
    return True


@torch.no_grad()
def prune_hidden_hybrid(model, arch, keep_h, r, calib_ids, device):
    """ADD-CHANNELS (vs fold-absorb): keep the top keep_h RAW channels EXACT, then ADD r channels = top-r
    principal directions of the DROPPED subspace (the independent info that folding-absorption projects away).
    New hidden = keep_h + r. Readers gain cols [W[:,idx] | W[:,drop]@Ur]; writers gain rows [W[idx] ; Ur^T@W[drop]].
    RMSNorm gains folded into readers first so the new channels need no separate gain. Q^T==Q^+ (Ur orthonormal)
    -> tied emb/head stay tied. Head-dim/attn-internal untouched."""
    if arch.family not in ("llama", "mistral", "qwen2", "qwen3"):
        raise ValueError(f"prune_hidden_hybrid: Llama-family only, got {arch.family}")
    H = model.config.hidden_size; head_dim = getattr(model.config, "head_dim", None) or H // model.config.num_attention_heads
    model.eval()
    def fold(norm, readers):                              # fold RMSNorm gain into readers -> norm becomes gainless
        g = norm.weight.data.float()
        for lin in readers: lin.weight.data = (lin.weight.data.float() * g.unsqueeze(0)).to(lin.weight.dtype)
        norm.weight.data = torch.ones_like(norm.weight.data)
    extras = arch.trainable_extra(model); fnorm, head = extras[0], extras[-1]
    emb = model.get_input_embeddings(); tied = bool(getattr(model.config, "tie_word_embeddings", False))
    if tied and head.weight is emb.weight:                # SEPARATE the shared emb/head tensor BEFORE folding fnorm into head
        head.weight = nn.Parameter(emb.weight.data.clone()); model.config.tie_word_embeddings = False
    for blk in arch.get_layers(model):
        fold(blk.input_layernorm, [blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj])
        fold(blk.post_attention_layernorm, [blk.mlp.gate_proj, blk.mlp.up_proj])
    fold(fnorm, [head])                                   # now folds into the SEPARATE head (emb untouched)
    hs = model(calib_ids, output_hidden_states=True, use_cache=False).hidden_states
    Cov = torch.zeros(H, H, device=device, dtype=torch.float32); imp = torch.zeros(H, device=device); mx = torch.zeros(H, device=device)
    for h in hs:
        x = h.float().reshape(-1, H); Cov += x.t() @ x; imp += (x ** 2).sum(0)
        mx = torch.maximum(mx, x.abs().amax(0))            # per-channel MAX activation (outlier/sink signal)
    idx = torch.topk(imp, keep_h).indices.sort().values
    drop = torch.tensor([i for i in range(H) if i not in set(idx.tolist())], device=device)
    Cdd = Cov[drop][:, drop]
    if getattr(model.config, "_addchan_ortho", False):              # SCHUR: dropped variance CONDITIONED on kept (independent/new info)
        Ckk = Cov[idx][:, idx] + 1e-2 * torch.eye(keep_h, device=device); Ckd = Cov[idx][:, drop]
        Cdd = Cdd - Ckd.t() @ torch.linalg.solve(Ckk, Ckd)          # orthogonal complement -> no redundancy with kept
    if getattr(model.config, "_addchan_wanda", False):              # WANDA/output-aware: weight dropped chans by reader USAGE (act x |W|)
        wn = torch.zeros(H, device=device)
        for blk in arch.get_layers(model):
            for lin in (blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj, blk.mlp.gate_proj, blk.mlp.up_proj):
                wn += (lin.weight.data.float() ** 2).sum(0)          # how much the readers USE each residual channel
        D = wn[drop].sqrt().clamp_min(1e-6)
        Cdd = D.unsqueeze(1) * Cdd * D.unsqueeze(0)                  # capture directions that are high-variance AND heavily-read
    if getattr(model.config, "_addchan_pls", False):                # PLS (teacher-supervised): dropped directions most PREDICTIVE of the output
        Yf = hs[-1].float().reshape(-1, H)                          # teacher final hidden = the downstream target
        Sxy = torch.zeros(len(drop), H, device=device)             # cross-covariance: dropped residual <-> output
        for h in hs[:-1]:
            Sxy += h.float().reshape(-1, H)[:, drop].t() @ Yf       # accumulate over layers (full downstream usage)
        Ur = torch.linalg.eigh(Sxy @ Sxy.t())[1][:, -r:].contiguous()   # top-r left singular vectors (max covariance with output)
    elif getattr(model.config, "_addchan_rrr", False):              # SVD-LLM / RRR: OUTPUT-optimal whitened directions (verified exact)
        nd = len(drop)
        G = torch.zeros(nd, nd, device=device)                      # reader Gram: how the downstream readers USE the dropped channels
        for blk in arch.get_layers(model):
            for lin in (blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj, blk.mlp.gate_proj, blk.mlp.up_proj):
                Wd = lin.weight.data.float()[:, drop]; G += Wd.t() @ Wd
        oh = model.get_output_embeddings()
        if oh is not None and oh.weight.shape[-1] == H: Wd = oh.weight.data.float()[:, drop]; G += Wd.t() @ Wd
        C = Cov[drop][:, drop] + (1e-3 * Cov[drop][:, drop].diagonal().mean()) * torch.eye(nd, device=device)  # damped
        w, V = torch.linalg.eigh(C); w = w.clamp_min(1e-8)
        Chalf = (V * w.sqrt()) @ V.t(); Cinv = (V * w.rsqrt()) @ V.t()   # C^{1/2}, C^{-1/2}
        Q = torch.linalg.eigh(Chalf @ G @ Chalf)[1][:, -r:]         # top-r output-energy directions in whitened coords
        Ur = torch.linalg.qr(Cinv @ Q)[0][:, :r].contiguous()       # un-whiten + orthonormalize
    elif getattr(model.config, "_addchan_combine3", False):         # 3-WAY DIVERSITY: PCA(variance) + orthogonal(independent) + OUTLIER(heavy-tail/sink)
        Cfull = Cov[drop][:, drop]; r3 = r // 3
        Upca = torch.linalg.eigh(Cfull)[1][:, -r3:]                 # variance directions
        Ckk = Cov[idx][:, idx] + 1e-2 * torch.eye(keep_h, device=device); Ckd = Cov[idx][:, drop]
        Uort = torch.linalg.eigh(Cfull - Ckd.t() @ torch.linalg.solve(Ckk, Ckd))[1][:, -r3:]   # independent-of-kept
        Do = mx[drop].clamp_min(1e-6)                              # max-activation weighting -> heavy-tail/sink directions
        Uout = torch.linalg.eigh(Do.unsqueeze(1) * Cfull * Do.unsqueeze(0))[1][:, -(r - 2 * r3):]
        Ur = torch.linalg.qr(torch.cat([Upca, Uort, Uout], 1))[0][:, :r].contiguous()   # union of 3 criteria, orthonormalized
    elif getattr(model.config, "_addchan_combine", False):          # COMPLEMENT PCA: keep PCA-of-dropped dirs + ADD orthogonal(Schur) dirs
        rh = r // 2; Cfull = Cov[drop][:, drop]
        Upca = torch.linalg.eigh(Cfull)[1][:, -rh:]                 # PCA variance directions (kept, as you asked)
        Ckk = Cov[idx][:, idx] + 1e-2 * torch.eye(keep_h, device=device); Ckd = Cov[idx][:, drop]
        Uort = torch.linalg.eigh(Cfull - Ckd.t() @ torch.linalg.solve(Ckk, Ckd))[1][:, -(r - rh):]   # orthogonal complement
        Ur = torch.linalg.qr(torch.cat([Upca, Uort], 1))[0][:, :r].contiguous()   # union, orthonormalized
    else:
        Ur = torch.linalg.eigh(Cdd)[1][:, -r:].contiguous()         # [n_drop, r] top-r directions (PCA by default)
    newH = keep_h + r
    s = torch.ones(r, device=device)
    if getattr(model.config, "_addchan_renorm", False):  # balance captured-channel residual scale to kept -> DECOUPLE the shared RMS
        cap_e = (Ur * (Cov[drop][:, drop] @ Ur)).sum(0)  # per-captured-channel energy (Ur^T Cdd Ur diagonal)
        s = (imp[idx].median() / cap_e.clamp_min(1e-8)).sqrt().clamp(0.1, 10.0)   # scale captured residual to kept median
    def reader(lin):                                     # residual is INPUT: cols [W[:,idx] | (W[:,drop]@Ur)/s]
        W = lin.weight.data.float()
        n = nn.Linear(newH, lin.out_features, bias=lin.bias is not None).to(lin.weight.device, lin.weight.dtype)
        n.weight.data.copy_(torch.cat([W[:, idx], (W[:, drop] @ Ur) / s], 1).to(lin.weight.dtype))
        if lin.bias is not None: n.bias.data.copy_(lin.bias.data)
        return n
    def writer(lin):                                     # residual is OUTPUT: rows [W[idx] ; s*(Ur^T@W[drop])]
        W = lin.weight.data.float()
        n = nn.Linear(lin.in_features, newH, bias=lin.bias is not None).to(lin.weight.device, lin.weight.dtype)
        n.weight.data.copy_(torch.cat([W[idx, :], s.unsqueeze(1) * (Ur.t() @ W[drop, :])], 0).to(lin.weight.dtype))
        if lin.bias is not None:
            n.bias.data.copy_(torch.cat([lin.bias.data.float()[idx], s * (Ur.t() @ lin.bias.data.float()[drop])]).to(lin.weight.dtype))
        return n
    E = emb.weight.data.float()                           # emb is the ORIGINAL (un-folded) embedding; WRITES residual -> *s
    emb.weight = nn.Parameter(torch.cat([E[:, idx], (E[:, drop] @ Ur) * s], 1).to(emb.weight.dtype)); emb.embedding_dim = newH
    for layer in arch.get_layers(model):
        a = layer.self_attn; a.q_proj, a.k_proj, a.v_proj, a.o_proj = reader(a.q_proj), reader(a.k_proj), reader(a.v_proj), writer(a.o_proj)
        m = layer.mlp; m.gate_proj, m.up_proj, m.down_proj = reader(m.gate_proj), reader(m.up_proj), writer(m.down_proj)
        for mod in layer.modules():
            w = getattr(mod, "weight", None)
            if w is not None and w.dim() == 1 and w.numel() == H:
                mod.weight = nn.Parameter(torch.ones(newH, device=w.device, dtype=w.dtype))
    if getattr(fnorm, "weight", None) is not None and fnorm.weight.numel() == H:
        fnorm.weight = nn.Parameter(torch.ones(newH, device=fnorm.weight.device, dtype=fnorm.weight.dtype))
    nh = reader(head); model.set_output_embeddings(nh) if hasattr(model, "set_output_embeddings") else setattr(model, "lm_head", nh)
    model.config.hidden_size = newH; model.config.head_dim = head_dim   # untied: head carries the folded fnorm gain, emb does not
    print(f"[surgery] hybrid-width: {keep_h} raw + {r} low-rank(dropped) = {newH} (vs {H}); add-channels for dropped subspace")
    return True


def prune_hidden_width(model, arch, keep_h: int, calib_ids, device) -> bool:
    """Prune the RESIDUAL-STREAM width (hidden_size) — the axis Minitron/Sheared-LLaMA use to reshape
    a large model toward a small model's narrow geometry. The most global/destructive axis: every weight
    touching the residual stream is sliced to the top-`keep_h` channels (by activation importance).
    Composes with depth removal + FFN-width + head pruning. Llama-family only (qwen2/qwen3/llama/mistral).

    Slices: embed (out), each q/k/v_proj (in), o_proj (out), gate/up_proj (in), down_proj (out),
    all residual-sized RMSNorms, final norm, lm_head (in). Attention internal dim (heads*head_dim) and
    FFN intermediate are UNTOUCHED (separate axes), so head_dim is pinned via config.head_dim.
    """
    if arch.family not in ("llama", "mistral", "qwen2", "qwen3"):
        raise ValueError(f"prune_hidden_width: Llama-family only, got {arch.family}")
    H = model.config.hidden_size
    if keep_h >= H:
        return False
    sink_pin = getattr(model.config, "_prune_sink_pin", 0)   # OUTLIER+RAW: force-pin the massive-activation sink dims (by max-act), no rotation
    head_dim = getattr(model.config, "head_dim", None) or H // model.config.num_attention_heads
    model.eval()
    hs = model(calib_ids, output_hidden_states=True, use_cache=False).hidden_states
    imp = torch.zeros(H, device=device, dtype=torch.float32)
    mx = torch.zeros(H, device=device, dtype=torch.float32)
    for h in hs:
        imp += (h.float() ** 2).sum(dim=(0, 1))          # per-channel activation energy (raw selection signal)
        mx = torch.maximum(mx, h.float().abs().amax(dim=(0, 1)))   # per-channel MAX activation (sharp sink signal)
    sel = imp.clone()
    _pins = 0
    if sink_pin > 0:                                      # outlier-style sink detection + raw energy fill (no rotation)
        sink = torch.topk(mx, min(sink_pin, keep_h)).indices
        sel[sink] = float("inf"); _pins += len(sink)     # force the sink dims into the kept set
        print(f"[surgery] outlier+raw: pinned {len(sink)} max-activation sink dims + {keep_h - len(sink)} by energy")
    math_ids = getattr(model.config, "_prune_math_calib", None)      # CAPABILITY-AWARE: reserve budget for math channels
    reserve = getattr(model.config, "_prune_math_reserve", 0)
    if math_ids is not None and reserve > 0:
        imp_m = torch.zeros(H, device=device, dtype=torch.float32)
        with torch.no_grad():                                        # NO autograd graph (the OOM cause: a 2nd full-fwd graph)
            hm = model(math_ids.to(device), output_hidden_states=True, use_cache=False).hidden_states
            for h in hm: imp_m += (h.float() ** 2).sum(dim=(0, 1))
            del hm
        torch.cuda.empty_cache()
        diff = imp_m / imp_m.sum().clamp_min(1e-9) - imp / imp.sum().clamp_min(1e-9)   # DIFFERENTIAL: math-specific, not generic sinks
        pin_m = torch.topk(diff, min(reserve, keep_h - _pins)).indices
        sel[pin_m] = float("inf"); _pins += len(pin_m)
        print(f"[surgery] CAPABILITY-AWARE: reserved {len(pin_m)} math-differential channels + {keep_h - _pins} by C4 energy")
    idx = torch.topk(sel, keep_h).indices.sort().values

    # CLOSED-FORM ABSORPTION (SliceGPT/OBS-style): instead of DROPPING the removed channels, fold their
    # contribution into the KEPT channels. Since h_dropped ~= M @ h_kept (residual is low-rank), the optimal
    # read weight is W_kept + W_dropped @ M -> minimizes output error at compression time (no heal needed).
    absorb = getattr(model.config, "_absorb_width", False)
    _keepset = set(idx.tolist())
    drop = torch.tensor([i for i in range(H) if i not in _keepset], device=device)
    M = None
    if absorb and len(drop) > 0:
        Cov = torch.zeros(H, H, device=device, dtype=torch.float32)
        for h in hs:                                          # uncentered 2nd moment of the residual (calib)
            x = h.float().reshape(-1, H); Cov += x.t() @ x
        Ckk = Cov[idx][:, idx] + 1e-2 * torch.eye(keep_h, device=device)
        Ckd = Cov[idx][:, drop]
        M = torch.linalg.solve(Ckk, Ckd).t().contiguous()     # [n_drop, keep_h]: dropped <- kept regression
        print(f"[surgery] absorb-width: folded {len(drop)} dropped channels into {keep_h} kept via closed-form lstsq (no heal)")

    def in_(lin):   # slice input features (residual-facing) -- ABSORB the dropped cols into the kept
        n = nn.Linear(keep_h, lin.out_features, bias=lin.bias is not None).to(lin.weight.device, lin.weight.dtype)
        if M is not None:
            W = lin.weight.data.float()
            n.weight.data.copy_((W[:, idx] + W[:, drop] @ M).to(lin.weight.dtype))
        else:
            n.weight.data.copy_(lin.weight.data[:, idx])
        if lin.bias is not None:
            n.bias.data.copy_(lin.bias.data)
        return n

    def out_(lin):  # slice output features (residual-facing)
        n = nn.Linear(lin.in_features, keep_h, bias=lin.bias is not None).to(lin.weight.device, lin.weight.dtype)
        n.weight.data.copy_(lin.weight.data[idx, :])
        if lin.bias is not None:
            n.bias.data.copy_(lin.bias.data[idx])
        return n

    emb = model.get_input_embeddings()
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    emb.weight = nn.Parameter(emb.weight.data[:, idx]); emb.embedding_dim = keep_h
    for layer in arch.get_layers(model):
        a = layer.self_attn
        a.q_proj, a.k_proj, a.v_proj = in_(a.q_proj), in_(a.k_proj), in_(a.v_proj)
        a.o_proj = out_(a.o_proj)
        m = layer.mlp
        m.gate_proj, m.up_proj, m.down_proj = in_(m.gate_proj), in_(m.up_proj), out_(m.down_proj)
        for mod in layer.modules():                       # residual-sized RMSNorms (NOT head-dim q/k norms)
            w = getattr(mod, "weight", None)
            if w is not None and w.dim() == 1 and w.numel() == H:
                mod.weight = nn.Parameter(w.data[idx])
    extras = arch.trainable_extra(model)                  # [final_norm, lm_head]
    fnorm, head = extras[0], extras[-1]
    if getattr(fnorm, "weight", None) is not None and fnorm.weight.dim() == 1 and fnorm.weight.numel() == H:
        fnorm.weight = nn.Parameter(fnorm.weight.data[idx])
    if tied:
        head.weight = emb.weight                          # re-tie to the sliced embedding
        head.in_features = keep_h
    else:
        new = in_(head)
        if hasattr(model, "lm_head"):
            model.lm_head = new
    model.config.hidden_size = keep_h
    model.config.head_dim = head_dim                      # pin attention internal dim (don't recompute from hidden)
    return True


class _RebaseLayer(nn.Module):
    """Wraps a decoder layer with a (keep x keep) re-base adapter applied to the incoming residual: converts the
    residual from the PREVIOUS layer's per-layer basis into THIS layer's basis. Enables a different loss-relevant
    subspace per stage on a shared additive residual (the cost of per-layer SliceGPT)."""
    def __init__(self, adapter, inner):
        super().__init__()
        self.adapter = nn.Parameter(adapter, requires_grad=False)   # keep x keep
        self.inner = inner
    def forward(self, hidden_states, *args, **kwargs):
        ht = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        ht = (ht.to(self.adapter.dtype) @ self.adapter.t())          # re-base into this layer's basis
        return self.inner(ht, *args, **kwargs)


@torch.no_grad()
def slicegpt_rotate_perlayer(model, calib_ids, device, keep_dim, solver="fisher"):
    """PER-LAYER SliceGPT (Llama/Qwen): each layer's residual gets its OWN loss-aware (Fisher) eigenbasis, sliced to
    keep_dim, with inter-layer re-base adapters. Per-stage Fisher subspaces differ (~140-dim middle) -> each layer
    carries only ITS loss-relevant subspace, not the global union -> far more aggressive width compression than one
    global basis. Returns keep_dim. Llama-family only."""
    fam = get_arch(model).family
    if fam not in ("llama", "mistral", "qwen2", "qwen3"):
        raise ValueError(f"perlayer SliceGPT: Llama-family only, got {fam}")
    layers = model.model.layers; H = model.config.hidden_size; L = len(layers)
    emb = model.model.embed_tokens; head = model.lm_head; fnorm = model.model.norm
    head_dim = getattr(model.config, "head_dim", None) or H // model.config.num_attention_heads
    if head.weight.data_ptr() == emb.weight.data_ptr():               # untie (folded final-norm gain corrupts shared emb)
        nh = nn.Linear(H, head.weight.shape[0], bias=head.bias is not None).to(head.weight.device, head.weight.dtype)
        nh.weight.data.copy_(head.weight.data); model.lm_head = nh; model.config.tie_word_embeddings = False; head = nh
    def fold_gain(norm, readers):
        g = norm.weight.data.float()
        for lin in readers: lin.weight.data = (lin.weight.data.float() * g.unsqueeze(0)).to(lin.weight.dtype)
        norm.weight.data = torch.ones_like(norm.weight.data)
    for blk in layers:
        fold_gain(blk.input_layernorm, [blk.self_attn.q_proj, blk.self_attn.k_proj, blk.self_attn.v_proj])
        fold_gain(blk.post_attention_layernorm, [blk.mlp.gate_proj, blk.mlp.up_proj])
    fold_gain(fnorm, [head])
    # per-layer Fisher cov of the residual ENTERING each layer (hidden_states[j])
    covs = [torch.zeros(H, H, device=device, dtype=torch.float32) for _ in range(L)]
    with torch.enable_grad():
        saved_rg = [(p, p.requires_grad) for p in model.parameters()]    # restore (don't freeze the student for the heal)
        for p in model.parameters(): p.requires_grad_(True)
        for s in range(0, min(calib_ids.size(1), 2048), 256):
            ids = calib_ids[:, s:s + 256].to(device); grads = {}
            def mk(k):
                def hk(g): grads[k] = g.detach()
                return hk
            out = model(ids, output_hidden_states=True, use_cache=False)
            hh = [h.register_hook(mk(j)) for j, h in enumerate(out.hidden_states) if h.requires_grad]
            F.cross_entropy(out.logits[:, :-1].reshape(-1, out.logits.size(-1)).float(), ids[:, 1:].reshape(-1)).backward()
            for j in range(L):
                if j in grads:
                    if solver == "saliency":              # act x grad (the global-sweep WINNER): both content AND loss-relevance
                        sal = (out.hidden_states[j][0].detach().float() * grads[j][0].float().abs().clamp_min(1e-6)).reshape(-1, H)
                        covs[j] += sal.t() @ sal
                    else:                                 # fisher: pure loss-gradient cov
                        g = grads[j][0].float().reshape(-1, H); covs[j] += g.t() @ g
            for h_ in hh: h_.remove()
            model.zero_grad(set_to_none=True)
        for p, rg in saved_rg: p.requires_grad_(rg)
    Qs = [torch.linalg.eigh(c)[1].flip(1)[:, :keep_dim].contiguous() for c in covs]   # H x keep per layer (top Fisher dirs)
    dt = emb.weight.dtype
    def rin(lin, Q):  lin.weight.data = (lin.weight.data.float() @ Q).to(dt)            # read residual: out x H -> out x keep
    def rout(lin, Q): lin.weight.data = (Q.t() @ lin.weight.data.float()).to(dt)        # write residual: H x in -> keep x in
    emb.weight.data = (emb.weight.data.float() @ Qs[0]).to(dt)                          # emb output in layer-0 basis
    for j, blk in enumerate(layers):
        Q = Qs[j]
        a = blk.self_attn
        rin(a.q_proj, Q); rin(a.k_proj, Q); rin(a.v_proj, Q); rout(a.o_proj, Q)
        rin(blk.mlp.gate_proj, Q); rin(blk.mlp.up_proj, Q); rout(blk.mlp.down_proj, Q)
        blk.input_layernorm.weight.data = blk.input_layernorm.weight.data[:keep_dim].clone()
        blk.post_attention_layernorm.weight.data = blk.post_attention_layernorm.weight.data[:keep_dim].clone()
    fnorm.weight.data = fnorm.weight.data[:keep_dim].clone()
    rin(head, Qs[L - 1])                                                                # head reads last layer's basis
    new_layers = [layers[0]]                                                            # layer 0: emb already in Q_0
    for j in range(1, L):
        A = (Qs[j].t() @ Qs[j - 1]).to(dt)                                             # keep x keep: re-base Q_{j-1} -> Q_j
        new_layers.append(_RebaseLayer(A, layers[j]))
    model.model.layers = nn.ModuleList(new_layers)
    model.config.hidden_size = keep_dim
    model.config.head_dim = head_dim
    return keep_dim


class _AttnResCoord:
    """Shared per-forward checkpoint buffer for attention-residual layers (plain object, not a Module)."""
    def __init__(self):
        self.ckpts = []


class AttnResLayer(nn.Module):
    """Gated attention-residual retrofit (Moonshot 'Attention Residuals', adapted to be retrofit-safe).
    Wraps a decoder layer; before it runs, INJECTS a learned softmax-weighted combination of EARLIER
    layer outputs into the residual stream — gated, ZERO-INIT, so at start it's the exact original model
    and healing can only ADD a re-routing pathway. After layer-removal disrupts the residual stream's
    depth balance, this lets a surviving layer pull earlier features where a pruned span used to feed.
    Acts on residual TOPOLOGY (not layer content) — complementary to a spliced reconstruction block."""
    def __init__(self, layer, coord, hidden, idx, dtype=None):
        super().__init__()
        self.layer = layer
        self._coord = coord                      # plain object, not registered
        self.idx = idx
        self.w = nn.Parameter(torch.zeros(hidden))   # pseudo-query: weights earlier checkpoints
        self.gate = nn.Parameter(torch.zeros(1))     # init 0 -> identity (no perturbation)
        self.norm = nn.LayerNorm(hidden)
        if dtype is not None:
            self.w.data = self.w.data.to(dtype); self.gate.data = self.gate.data.to(dtype); self.norm.to(dtype)

    def forward(self, hidden_states, *args, **kwargs):
        h = hidden_states
        if self.idx == 0:
            self._coord.ckpts = [h]              # new forward pass -> reset history with the embedding
        ck = self._coord.ckpts
        V = torch.stack(ck, 0)                    # [n_ckpt, B, L, D]
        scores = (self.norm(V) * self.w).sum(-1)  # [n_ckpt, B, L] — learned weight per earlier output
        agg = (scores.softmax(0).unsqueeze(-1) * V).sum(0)   # [B, L, D]
        h = h + self.gate * agg                  # GATED additive injection (gate 0 -> h unchanged)
        out = self.layer(h, *args, **kwargs)
        out_t = out[0] if isinstance(out, tuple) else out
        self._coord.ckpts.append(out_t)          # this layer's output joins the history
        return out


def attn_res_retrofit(arch, model, dtype, device):
    """Wrap every layer with AttnResLayer (shared checkpoint buffer). Returns the AttnRes params to heal."""
    coord = _AttnResCoord()
    layers = list(arch.get_layers(model))
    wrapped = [AttnResLayer(l, coord, model.config.hidden_size, i, dtype).to(device)
               for i, l in enumerate(layers)]
    arch.set_layers(model, nn.ModuleList(wrapped))
    params = []
    for w in wrapped:
        params += [w.w, w.gate] + list(w.norm.parameters())   # the small AttnRes params (heal these)
    return params, wrapped


class ReplaceMeFusion(nn.Module):
    """Fuse ReplaceMe's analytic LINEAR span-map with a trainable nonlinear BLOCK. h_out = h@T + gate·block(h):
    the frozen ReplaceMe transform T does the bulk linear mapping for free (so h@T ≈ h_out at init), and the
    trainable block (gate init 0) learns ONLY the nonlinear RESIDUAL the linear map misses. Combines analytic
    init (cheap, expressive-limited) + block capacity (expressive, needs training) — the best of both."""
    def __init__(self, block, T):
        super().__init__()
        self.block = block
        self.T = nn.Parameter(T, requires_grad=False)    # frozen analytic linear backbone
        self.gate = nn.Parameter(torch.zeros(1))         # block enters gradually -> starts as pure ReplaceMe

    def forward(self, hidden_states, *args, **kwargs):
        lin = (hidden_states.to(self.T.dtype) @ self.T).to(hidden_states.dtype)
        out = self.block(hidden_states, *args, **kwargs)
        b = out[0] if isinstance(out, tuple) else out
        delta = b - hidden_states                         # the block's NONLINEAR contribution (attn+mlp)
        r = lin + self.gate.to(b.dtype) * delta           # ReplaceMe linear + gated nonlinear residual
        return (r,) if isinstance(out, tuple) else r


class TransformLayer(nn.Module):
    """ReplaceMe (arXiv 2505.02819): a fitted AFFINE map h@T + b replacing a DELETED layer span. The bias b
    is the mean/OFFSET correction — RMSNorm doesn't recenter, so the deletion-induced DC shift must be
    restored explicitly (normalization-agent finding). ZERO training. Returns bare tensor."""
    def __init__(self, T, bias=None):
        super().__init__()
        self.T = nn.Parameter(T, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, hidden_states, *args, **kwargs):
        out = hidden_states.to(self.T.dtype) @ self.T
        if self.bias is not None:
            out = out + self.bias
        return out.to(hidden_states.dtype)


@torch.no_grad()
def replaceme_fit(teacher, arch, start, count, calib_ids, device, ridge=1e-2, dtype=None, cosine=False):
    """CORRECT ReplaceMe (arXiv 2505.02819): fit T on the SUBSET objective M_i -> (L_{i+n} - Y_i), where
    M_i = pure MLP output of the last surviving block (pre-residual-add), Y_i = L_i - M_i (attn+skip path,
    passes through untouched). T = (M_iᵀM_i)⁻¹ M_iᵀ (L_{i+n} - L_i + M_i). Far better-specified than fitting
    the full residual: T only re-maps the MLP-output channel; the skip is carried by the residual."""
    tlayers = arch.get_layers(teacher)
    mlp = tlayers[start - 1].mlp
    buf = {}
    hh = mlp.register_forward_hook(lambda m, inp, out: buf.__setitem__("mi", out.detach()))
    H = teacher.config.hidden_size
    Mi, Li, Lin = [], [], []
    teacher.eval()
    for k in range(0, min(calib_ids.size(1), 8192), 256):
        hs = teacher(calib_ids[:, k:k + 256].to(device), output_hidden_states=True, use_cache=False).hidden_states
        Mi.append(buf["mi"].float().reshape(-1, H))
        Li.append(hs[start].float().reshape(-1, H))           # L_i  = residual after block start-1
        Lin.append(hs[start + count].float().reshape(-1, H))  # L_{i+n} = residual after the span
    hh.remove()
    Mi = torch.cat(Mi, 0); Li = torch.cat(Li, 0); Lin = torch.cat(Lin, 0)
    target = Lin - Li + Mi                                     # = L_{i+n} - Y_i
    A = Mi.t() @ Mi + ridge * torch.eye(H, device=Mi.device)
    T = torch.linalg.solve(A, Mi.t() @ target)                # (H, H) closed-form L2 init
    if cosine:                                                # refine on COSINE objective (ReplaceMe: +1.2 acc)
        with torch.enable_grad():                             # re-enable autograd (fn is @torch.no_grad)
            T = T.clone().requires_grad_(True)
            o = torch.optim.Adam([T], lr=1e-3)
            for _ in range(300):
                pred = Mi @ T
                loss = (1 - torch.nn.functional.cosine_similarity(pred, target, dim=-1)).mean()
                o.zero_grad(); loss.backward(); o.step()
        T = T.detach()
    return T.to(dtype) if dtype is not None else T


class SubFitLayer(nn.Module):
    """SubFit (arXiv 2606.02559): richer closed-form span replacement than ReplaceMe's single linear T.
    F(x) = g⊙x + b + ((x−μ)A)B — a per-feature GATED AFFINE (g,b fix the splice activation/magnitude &
    outlier mismatch that dominates layer-pruning damage) + a LOW-RANK residual bypass A·B (captures the
    dominant residual structure a single T smears). Training-free. Drop-in decoder layer (returns bare)."""
    def __init__(self, g, b, mu, A, B):
        super().__init__()
        self.g = nn.Parameter(g, requires_grad=False); self.b = nn.Parameter(b, requires_grad=False)
        self.mu = nn.Parameter(mu, requires_grad=False)
        self.A = nn.Parameter(A, requires_grad=False); self.B = nn.Parameter(B, requires_grad=False)

    def forward(self, hidden_states, *args, **kwargs):
        x = hidden_states.to(self.g.dtype)
        out = self.g * x + self.b + ((x - self.mu) @ self.A) @ self.B
        return out.to(hidden_states.dtype)


@torch.no_grad()
def subfit_fit(teacher, arch, start, count, calib_ids, device, rank=64, ridge=1e-2, dtype=None):
    """Fit a SubFit gated-affine + low-rank map for the span [start, start+count): per-feature affine g,b
    then a rank-r least-squares map of the centered input to the affine RESIDUAL."""
    H = teacher.config.hidden_size
    X, Y = [], []
    teacher.eval()
    for k in range(0, min(calib_ids.size(1), 8192), 256):
        hs = teacher(calib_ids[:, k:k + 256].to(device), output_hidden_states=True, use_cache=False).hidden_states
        X.append(hs[start][0].float().reshape(-1, H)); Y.append(hs[start + count][0].float().reshape(-1, H))
    X = torch.cat(X, 0); Y = torch.cat(Y, 0)
    mu = X.mean(0); ybar = Y.mean(0); xc = X - mu
    g = (xc * (Y - ybar)).sum(0) / (xc * xc).sum(0).clamp_min(1e-6)    # per-feature gated affine slope
    b = ybar - g * mu
    R = Y - (g * X + b)                                                # affine residual
    Gm = xc.t() @ xc + ridge * torch.eye(H, device=device)
    L = torch.linalg.solve(Gm, xc.t() @ R)                            # (H,H): R ≈ xc @ L
    U, S, Vt = torch.linalg.svd(L)
    r = min(rank, S.shape[0]); rs = S[:r].clamp_min(0).sqrt()
    A = U[:, :r] * rs.unsqueeze(0)                                    # (H, r)  — A·B = rank-r truncation of L
    B = rs.unsqueeze(1) * Vt[:r]                                      # (r, H)
    lay = SubFitLayer(g, b, mu, A, B)
    return lay.to(dtype) if dtype is not None else lay


@torch.no_grad()
def replaceme_fit_full(teacher, arch, start, count, calib_ids, device, ridge=1e-2, dtype=None):
    """Full-residual fit T: H_before @ T ≈ H_after (the WHOLE residual, not the MLP-output subset). Used by
    the FUSION (h@T as an analytic backbone, then a trainable block learns the nonlinear residual). Distinct
    from replaceme_fit (subset objective, for standalone ReplaceMe merged into down_proj)."""
    H = teacher.config.hidden_size
    Hb, Ha = [], []
    teacher.eval()
    for k in range(0, min(calib_ids.size(1), 8192), 256):
        hs = teacher(calib_ids[:, k:k + 256].to(device), output_hidden_states=True, use_cache=False).hidden_states
        Hb.append(hs[start].float().reshape(-1, H)); Ha.append(hs[start + count].float().reshape(-1, H))
    Hb = torch.cat(Hb, 0); Ha = torch.cat(Ha, 0)
    A = Hb.t() @ Hb + ridge * torch.eye(H, device=Hb.device)
    T = torch.linalg.solve(A, Hb.t() @ Ha)
    return T.to(dtype) if dtype is not None else T


@torch.no_grad()
def prune_comp_fit(teacher, start, count, calib_ids, device):
    """Prune&Comp (2507.18212): the DIAGONAL magnitude-compensation subset of ReplaceMe. Deletion shrinks
    the residual stream (removed layers' contribution gone); rescale per-channel to restore the post-span
    magnitude profile. s_c = rms(L_{i+n}[:,c]) / rms(L_i[:,c]). Returns a diagonal T for TransformLayer."""
    Li, Lin = [], []
    H = teacher.config.hidden_size
    teacher.eval()
    for k in range(0, min(calib_ids.size(1), 4096), 256):
        hs = teacher(calib_ids[:, k:k + 256].to(device), output_hidden_states=True, use_cache=False).hidden_states
        Li.append(hs[start].float().reshape(-1, H)); Lin.append(hs[start + count].float().reshape(-1, H))
    Li = torch.cat(Li, 0); Lin = torch.cat(Lin, 0)
    s = ((Lin.pow(2).mean(0).sqrt()) / (Li.pow(2).mean(0).sqrt() + 1e-6)).clamp(0.2, 5.0)  # per-channel scale
    b = Lin.mean(0) - s * Li.mean(0)                                      # OFFSET: restore the DC shift (RMSNorm)
    return torch.diag(s), b                                              # affine: h@diag(s) + b


@torch.no_grad()
def replaceme_apply(model, teacher, arch, start, count, calib_ids, device, ridge=1e-2, cosine=False):
    """Fit T and MERGE it into the (compressed) preceding block's down_proj: W_down <- Tᵀ·W_down. Zero
    added params. Assumes a single contiguous removed span so the preceding block is at index start-1."""
    T = replaceme_fit(teacher, arch, start, count, calib_ids, device, ridge=ridge, cosine=cosine)
    blk = arch.get_layers(model)[start - 1]
    dp = getattr(blk.mlp, "down_proj", None) or getattr(blk.mlp, "dense_4h_to_h", None) or getattr(blk.mlp, "c_proj", None)
    dp.weight.data = (T.t().to(torch.float32) @ dp.weight.data.float()).to(dp.weight.dtype)
    return T.shape[0]


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank adapter (B·A), B init 0 -> identity at start. For the
    frozen-regime seam: every conversion paper found FULLY freezing the backbone hurts; LoRA on the
    layers adjacent to the removed span lets the seam adapt cheaply."""
    def __init__(self, base, r=16, alpha=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        dt, dev = base.weight.dtype, base.weight.device
        self.A = nn.Linear(base.in_features, r, bias=False).to(dev, dt)
        self.B = nn.Linear(r, base.out_features, bias=False).to(dev, dt)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


def inject_lora_seam(model, arch, repl_blocks, r=16, span=1):
    """Wrap the q/k/v/o/gate/up/down projections of the kept blocks within `span` of each replacement/gap
    with LoRA adapters. Returns the trainable LoRA params."""
    layers = list(arch.get_layers(model))
    repl_ids = {id(b) for b in repl_blocks}
    seam_idx = set()
    for i, l in enumerate(layers):
        if id(l) in repl_ids:
            for d in range(1, span + 1):
                if i - d >= 0: seam_idx.add(i - d)
                if i + d < len(layers): seam_idx.add(i + d)
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "query_key_value", "dense",
             "gate_proj", "up_proj", "down_proj", "c_fc", "c_proj", "dense_h_to_4h", "dense_4h_to_h")
    params = []
    for i in seam_idx:
        blk = layers[i]
        for parent in (getattr(blk, "self_attn", None) or getattr(blk, "attention", None), getattr(blk, "mlp", None)):
            if parent is None:
                continue
            for nm in names:
                lin = getattr(parent, nm, None)
                if isinstance(lin, nn.Linear):
                    lora = LoRALinear(lin, r=r)
                    setattr(parent, nm, lora)
                    params += [lora.A.weight, lora.B.weight]
    return params, len(seam_idx)


class CausalLinAttn(nn.Module):
    """Causal linear attention (O(L), UNBOUNDED range) — a true GLOBAL side-path for the hybrid, vs the
    conv's fixed kernel. The causal/cheap form of the SPECTRE idea: phi=elu+1 feature map, causal via prefix
    cumsum of phi(k)⊗v. Zero-init output projection -> starts as a no-op, like the conv side-path. Computed
    in fp32 (linear attn is precision-sensitive)."""
    def __init__(self, hidden, n_heads, dtype=None):
        super().__init__()
        self.h = hidden; self.nh = n_heads; self.dh = hidden // n_heads
        self.q = nn.Linear(hidden, hidden); self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden); self.o = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)         # no-op at init
        if dtype is not None:
            self.to(dtype)

    def forward(self, x):
        B, L, _ = x.shape
        q = (F.elu(self.q(x).float()) + 1).view(B, L, self.nh, self.dh)
        k = (F.elu(self.k(x).float()) + 1).view(B, L, self.nh, self.dh)
        v = self.v(x).float().view(B, L, self.nh, self.dh)
        kv = torch.einsum('blhd,blhe->blhde', k, v).cumsum(1)              # (B,L,nh,dh,dh) prefix state
        ks = k.cumsum(1)                                                   # (B,L,nh,dh) prefix normalizer
        num = torch.einsum('blhd,blhde->blhe', q, kv)                      # (B,L,nh,dh)
        den = torch.einsum('blhd,blhd->blh', q, ks).clamp_min(1e-4).unsqueeze(-1)
        out = (num / den).reshape(B, L, self.h).to(x.dtype)
        return self.o(out)


class HybridBlock(nn.Module):
    """The 'keep a sliver of exact attention' replacement (Lizard/STILL family, 2507.09025/2602.02180).
    Conv/light-attn replacements plateau ~66% because they lack real attention; a FULL block gets ~80% but
    costs O(L²). This keeps REAL multi-head softmax attention but SLIDING-WINDOW (O(L·W)) + a cheap GLOBAL
    side-path (conv OR causal linear-attention) + FFN. WARM-INIT attention/FFN from the parent span (Lizard
    inits from parent projections — warm-start dominates architecture in the frozen regime). Returns bare."""
    def __init__(self, hidden, n_heads, window=128, conv_kernel=7, intermediate=None, dtype=None, global_path="conv"):
        super().__init__()
        self.h = hidden; self.nh = n_heads; self.dh = hidden // n_heads; self.window = window
        self.global_path = global_path
        self.n1 = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, hidden); self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden); self.o = nn.Linear(hidden, hidden)
        self.nc = nn.LayerNorm(hidden)
        if global_path == "linattn":                          # unbounded-range causal linear-attn side-path
            self.glob = CausalLinAttn(hidden, n_heads)
        else:                                                 # local depthwise conv side-path (fixed kernel)
            self.conv = nn.Conv1d(hidden, hidden, conv_kernel, padding=conv_kernel - 1, groups=hidden)
            nn.init.zeros_(self.conv.weight); nn.init.zeros_(self.conv.bias)   # side-path starts as no-op
        self.n2 = nn.LayerNorm(hidden)
        m = intermediate or 4 * hidden
        self.fc1 = nn.Linear(hidden, m); self.fc2 = nn.Linear(m, hidden)
        if dtype is not None:
            self.to(dtype)

    @torch.no_grad()
    def warm_init(self, src, family):
        """Copy attention q/k/v/o + FFN from a parent decoder layer `src` (warm-start, not random)."""
        def cp(dst, w, b=None):
            dst.weight.data.copy_(w.to(dst.weight.dtype))
            if b is not None and dst.bias is not None: dst.bias.data.copy_(b.to(dst.bias.dtype))
        if family == "gpt_neox":
            at, mlp = src.attention, src.mlp
            W = at.query_key_value.weight.data.float().view(self.nh, 3 * self.dh, self.h)  # per-head interleaved
            B = at.query_key_value.bias.data.float().view(self.nh, 3 * self.dh) if at.query_key_value.bias is not None else None
            cp(self.q, W[:, 0:self.dh].reshape(self.h, self.h), B[:, 0:self.dh].reshape(self.h) if B is not None else None)
            cp(self.k, W[:, self.dh:2*self.dh].reshape(self.h, self.h), B[:, self.dh:2*self.dh].reshape(self.h) if B is not None else None)
            cp(self.v, W[:, 2*self.dh:3*self.dh].reshape(self.h, self.h), B[:, 2*self.dh:3*self.dh].reshape(self.h) if B is not None else None)
            cp(self.o, at.dense.weight.data.float(), at.dense.bias.data.float() if at.dense.bias is not None else None)
            if mlp.dense_h_to_4h.weight.shape[0] == self.fc1.weight.shape[0]:   # only if FFN width matches
                cp(self.fc1, mlp.dense_h_to_4h.weight.data.float(), mlp.dense_h_to_4h.bias.data.float() if mlp.dense_h_to_4h.bias is not None else None)
                cp(self.fc2, mlp.dense_4h_to_h.weight.data.float(), mlp.dense_4h_to_h.bias.data.float() if mlp.dense_4h_to_h.bias is not None else None)
        else:                                            # Llama/Qwen best-effort (separate q/k/v/o, SwiGLU up/down)
            at, mlp = src.self_attn, src.mlp
            cp(self.q, at.q_proj.weight.data.float()); cp(self.o, at.o_proj.weight.data.float())
            if at.k_proj.weight.shape[0] == self.h:      # only if not GQA-shrunk
                cp(self.k, at.k_proj.weight.data.float()); cp(self.v, at.v_proj.weight.data.float())
            if mlp.up_proj.weight.shape[0] == self.fc1.weight.shape[0]:
                cp(self.fc1, mlp.up_proj.weight.data.float()); cp(self.fc2, mlp.down_proj.weight.data.float())
        if self.global_path == "linattn":                # WARM-INIT the linear-attn side-path from the same
            cp(self.glob.q, self.q.weight.data.float())  # parent projections (Mamba-in-Llama: warm-init is the
            cp(self.glob.k, self.k.weight.data.float())  # difference between a working side-path and chance).
            cp(self.glob.v, self.v.weight.data.float())  # glob.o stays zero-init -> side-path starts as no-op.

    def forward(self, hidden_states, *args, **kwargs):
        x = hidden_states; B, L, _ = x.shape
        h = self.n1(x)
        q = self.q(h).view(B, L, self.nh, self.dh).transpose(1, 2)
        k = self.k(h).view(B, L, self.nh, self.dh).transpose(1, 2)
        v = self.v(h).view(B, L, self.nh, self.dh).transpose(1, 2)
        idx = torch.arange(L, device=x.device)
        m = (idx[None, :] <= idx[:, None]) & (idx[:, None] - idx[None, :] < self.window)  # causal + window
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m[None, None])
        x = x + self.o(a.transpose(1, 2).reshape(B, L, self.h))
        if self.global_path == "linattn":                                   # unbounded-range global side-path
            x = x + self.glob(self.nc(x))
        else:                                                               # local conv side-path
            x = x + self.conv(self.nc(x).transpose(1, 2))[..., :L].transpose(1, 2)
        x = x + self.fc2(F.gelu(self.fc1(self.n2(x))))
        return x


class CheapBlock(nn.Module):
    """A cheap 'micro-model' replacement for a removed span (frozen-regime condenser). Depthwise CAUSAL
    conv for local token-mixing + optional light grouped self-attention for long-range + a small FFN.
    ~1/4 the params of a full decoder block. Drop-in: accepts the decoder-layer call signature (extra
    rotary/mask kwargs ignored — the conv is causal so no future leakage) and returns a 1-tuple.
    """
    def __init__(self, hidden, conv_kernel=7, ffn_mult=2.0, attn=False, attn_heads=8, dtype=None):
        super().__init__()
        self.n1 = nn.LayerNorm(hidden)
        self.conv = nn.Conv1d(hidden, hidden, conv_kernel, padding=conv_kernel - 1, groups=hidden)  # depthwise causal
        self.use_attn = attn
        if attn:
            self.na = nn.LayerNorm(hidden)
            self.h = attn_heads; self.hd = hidden // attn_heads
            self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
            self.po = nn.Linear(hidden, hidden, bias=False)
        self.n2 = nn.LayerNorm(hidden)
        m = int(hidden * ffn_mult)
        self.ffn = nn.Sequential(nn.Linear(hidden, m), nn.GELU(), nn.Linear(m, hidden))
        if dtype is not None:
            self.to(dtype)

    def forward(self, hidden_states, *args, **kwargs):
        x = hidden_states
        L = x.size(1)
        h = self.n1(x).transpose(1, 2)                       # [B,D,L]
        h = self.conv(h)[..., :L].transpose(1, 2)            # causal crop -> [B,L,D]
        x = x + h
        if self.use_attn:                                    # light causal self-attention
            B = x.size(0)
            q, k, v = self.qkv(self.na(x)).chunk(3, dim=-1)
            q = q.view(B, L, self.h, self.hd).transpose(1, 2)
            k = k.view(B, L, self.h, self.hd).transpose(1, 2)
            v = v.view(B, L, self.h, self.hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.po(a.transpose(1, 2).reshape(B, L, -1))
        x = x + self.ffn(self.n2(x))
        return x      # transformers 5.x decoder layers return a bare tensor


class FusionBlock(nn.Module):
    """The fusion reframe (done right): wrap a FULL decoder block (attention path 100% PRESERVED) and
    ADD gated, additive conv + MLP side-pathways. gates init 0 -> starts as the pure decoder block, so
    it can ONLY add new pathways, never dilute attention (the failure mode of the old concat-fusion).
    More capacity than a block, attention undiluted -> for the quality regime (exceed terminal state)."""
    def __init__(self, base_block, hidden, conv_kernel=7, ffn_mult=1.0, dtype=None):
        super().__init__()
        self.block = base_block                              # full decoder block (warm-started)
        self.nc = nn.LayerNorm(hidden)
        self.conv = nn.Conv1d(hidden, hidden, conv_kernel, padding=conv_kernel - 1, groups=hidden)
        self.gc = nn.Parameter(torch.zeros(1))               # gate -> 0 init (pure block at start)
        self.nm = nn.LayerNorm(hidden)
        m = int(hidden * ffn_mult)
        self.mlp = nn.Sequential(nn.Linear(hidden, m), nn.GELU(), nn.Linear(m, hidden))
        self.gm = nn.Parameter(torch.zeros(1))
        if dtype is not None:
            self.to(dtype)        # cast side-paths + gates to match the model (base block already is)

    def forward(self, hidden_states, *args, **kwargs):
        out = self.block(hidden_states, *args, **kwargs)     # FULL block (may return tensor or tuple)
        x = out[0] if isinstance(out, tuple) else out
        L = x.size(1)
        c = self.conv(self.nc(x).transpose(1, 2))[..., :L].transpose(1, 2)   # causal conv side-path
        x = x + self.gc * c
        x = x + self.gm * self.mlp(self.nm(x))               # MLP side-path
        return (x,) if isinstance(out, tuple) else x


def _make_block(cls):
    def _f(model, layer_idx):
        try:
            return cls(model.config, layer_idx=layer_idx)
        except TypeError:
            return cls(model.config)
    return _f


# ---------------------------------------------------------------- detection
def get_arch(model) -> Arch:
    mt = model.config.model_type
    if mt == "gpt2":
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block
        return Arch("gpt2",
            get_layers=lambda m: m.transformer.h,
            set_layers=lambda m, ls: setattr(m.transformer, "h", ls),
            n_layer_attr="n_layer",
            make_block=_make_block(GPT2Block),
            prune_ffn=_prune_gpt2,
            ffn_width=lambda b: b.mlp.c_fc.weight.shape[1],
            ffn_down=lambda b: b.mlp.c_proj,
            trainable_extra=lambda m: [m.transformer.ln_f, m.lm_head])
    if mt == "gpt_neox":
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
        return Arch("gpt_neox",
            get_layers=lambda m: m.gpt_neox.layers,
            set_layers=lambda m, ls: setattr(m.gpt_neox, "layers", ls),
            n_layer_attr="num_hidden_layers",
            make_block=_make_block(GPTNeoXLayer),
            prune_ffn=_prune_gptneox,
            ffn_width=lambda b: b.mlp.dense_h_to_4h.out_features,
            ffn_down=lambda b: b.mlp.dense_4h_to_h,
            trainable_extra=lambda m: [m.gpt_neox.final_layer_norm, m.embed_out])
    if mt in ("llama", "mistral", "qwen2", "qwen3"):
        # Qwen2/3 share Llama's structure (RMSNorm, SwiGLU mlp, GQA q/k/v/o_proj).
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        cls = LlamaDecoderLayer
        if mt == "mistral":
            from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
            cls = MistralDecoderLayer
        elif mt == "qwen2":
            from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
            cls = Qwen2DecoderLayer
        elif mt == "qwen3":
            from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
            cls = Qwen3DecoderLayer
        return Arch(mt,
            get_layers=lambda m: m.model.layers,
            set_layers=lambda m, ls: setattr(m.model, "layers", ls),
            n_layer_attr="num_hidden_layers",
            make_block=_make_block(cls),
            prune_ffn=_prune_llama,
            ffn_width=lambda b: b.mlp.gate_proj.out_features,
            ffn_down=lambda b: b.mlp.down_proj,
            trainable_extra=lambda m: [m.model.norm, m.lm_head])
    raise ValueError(f"unsupported model_type '{mt}' (add an adapter in src/arch.py)")


@torch.no_grad()
def prune_gptneox_heads(block, keep_heads: int, config) -> bool:
    """Prune attention heads of a GPT-NeoX block (fused query_key_value). Score heads by the
    combined weight norm of their q/k/v slices + output slice; keep the top `keep_heads`.
    Orthogonal structural axis: stacks on depth removal + FFN width."""
    attn = block.attention
    H = config.num_attention_heads
    hd = config.hidden_size // H
    if keep_heads >= H:
        return False
    qkv = attn.query_key_value  # Linear(hidden, 3*hidden); rows grouped [head, 3, head_dim] in NeoX
    dense = attn.dense          # Linear(hidden, hidden)
    W = qkv.weight.data.view(H, 3 * hd, -1)        # (heads, 3*hd, hidden)
    dW = dense.weight.data.view(-1, H, hd)         # (hidden, heads, hd)
    score = W.float().norm(dim=(1, 2)) * dW.float().norm(dim=(0, 2))  # per-head importance
    keep = torch.topk(score, keep_heads).indices.sort().values
    dev, dt = qkv.weight.device, qkv.weight.dtype
    nq = nn.Linear(qkv.in_features, keep_heads * 3 * hd, bias=qkv.bias is not None).to(dev, dt)
    nq.weight.data.copy_(W[keep].reshape(keep_heads * 3 * hd, -1))
    if qkv.bias is not None:
        nq.bias.data.copy_(qkv.bias.data.view(H, 3 * hd)[keep].reshape(-1))
    nd = nn.Linear(keep_heads * hd, dense.out_features, bias=dense.bias is not None).to(dev, dt)
    nd.weight.data.copy_(dW[:, keep, :].reshape(dense.out_features, keep_heads * hd))
    if dense.bias is not None:
        nd.bias.data.copy_(dense.bias.data)
    attn.query_key_value, attn.dense = nq, nd
    attn.num_attention_heads = keep_heads
    attn.head_size = hd
    if hasattr(attn, "hidden_size"):
        attn.hidden_size = keep_heads * hd
    return True


@torch.no_grad()
def svd_factorize_block(block, ratio: float, min_dim: int = 512) -> int:
    """Low-rank (SVD) factorization of a block's Linear weights — a SEPARATE parameter-removal
    axis that stacks with depth/width pruning. Each W (out x in) is replaced by A@B with rank
    r = ratio*min(in,out): Linear(in->r, no bias) then Linear(r->out, original bias). Only
    factorizes when it actually saves params. Returns #linears factorized.

    Stacks on top of layer removal: remove redundant DEPTH, then low-rank the survivors' WEIGHTS.
    """
    n = 0
    for parent_name, parent in list(block.named_modules()):
        for cname, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            din, dout = child.in_features, child.out_features
            if min(din, dout) < min_dim:
                continue
            r = max(1, int(ratio * min(din, dout)))
            if r * (din + dout) >= din * dout:  # no saving
                continue
            W = child.weight.data.float()  # (out, in)
            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except Exception:
                continue
            sq = S[:r].sqrt()
            A = (U[:, :r] * sq).to(child.weight.dtype)        # (out, r)
            B = (sq.unsqueeze(1) * Vh[:r, :]).to(child.weight.dtype)  # (r, in)
            lin1 = nn.Linear(din, r, bias=False).to(child.weight.device, child.weight.dtype)
            lin2 = nn.Linear(r, dout, bias=child.bias is not None).to(child.weight.device, child.weight.dtype)
            lin1.weight.data.copy_(B)
            lin2.weight.data.copy_(A)
            if child.bias is not None:
                lin2.bias.data.copy_(child.bias.data)
            setattr(parent, cname, nn.Sequential(lin1, lin2))
            n += 1
    return n


def reassign_layer_idx(layers):
    """KV-cache correctness after the stack shrinks: renumber every layer_idx found."""
    for i, blk in enumerate(layers):
        for obj in (blk, getattr(blk, "attn", None), getattr(blk, "attention", None),
                    getattr(blk, "self_attn", None)):
            if obj is not None and hasattr(obj, "layer_idx"):
                obj.layer_idx = i
