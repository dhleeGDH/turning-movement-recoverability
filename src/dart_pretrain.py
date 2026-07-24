"""Step 1 (F, and proper G): domain continued-pretraining of GraphPFN across a PRIOR of
(network scale, scenario, contiguous missing block) instances -- the SUMO pipeline reused
as a prior generator. Two config flags give the ablation ladder:

  E  = vanilla GraphPFN, no domain training (untrained eval)
  F  = domain continued-pretraining, cluster attention OFF
  G  = domain continued-pretraining, cluster attention ON   (fair test: learned in training)

Evaluated on a held-out (scale, real-U cluster) over held-out scenarios, skill by h.
rho=0 (clean) for this first pass; rho-robustness is a later extension.

    PYTHONPATH=. uv run --project external/graphpfn python -m src.dart_pretrain
"""

import numpy as np

from .graph.labels import exit_volume_per_movement
from .graph.movement_graph import assign_ou, build_movement_graph
from .graph.net_parser import parse_net
from .graph.observe import approach_observed_mask, exit_observed_mask
from .graph.scenarios import generate_cluster_configs, grid_coords, rect_block
from .graph.to_graphpfn import assemble_features, build_dgl_graph, static_features
from .models import cluster_attention as CA
from .pilot import load_scenarios, net_path
from .pilot_cluster import depth_h

CKPT = "hf://eremeev-d/graphpfn-1.3/graphpfn-adapters-1_3.pt"
TRAIN_SCALES = ["3by3", "5by5", "6by6", "8by8"]     # eval on 9by9 (held-out scale)
EVAL_SCALE = "9by9"
SHAPES = [(1, 1), (2, 2), (3, 3), (4, 4)]


def _load(scale, n_eval_scen, seed):
    net = parse_net(net_path(scale))
    mg = build_movement_graph(net)
    Y, APPR = load_scenarios(net, scale)
    Y = Y.astype(np.float32)
    S, N = Y.shape
    EXIT = exit_volume_per_movement(net, Y)
    static = static_features(net)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(S)
    eval_scen, train_scen = perm[:n_eval_scen], perm[n_eval_scen:]
    with np.errstate(divide="ignore", invalid="ignore"):
        hist_split = np.nan_to_num(np.where(APPR[train_scen] > 0,
                                            Y[train_scen] / APPR[train_scen], 0.0).mean(0))
    hist_mean = Y[train_scen].mean(0)                       # per-movement historical mean (= A)
    demand_ref = float(APPR[train_scen].mean())            # global reference demand level
    rc = grid_coords(net)
    nr = max(r for r, _ in rc.values()) + 1
    nc = max(c for _, c in rc.values()) + 1
    return dict(net=net, mg=mg, Y=Y, APPR=APPR, EXIT=EXIT, static=static, N=N,
                train_scen=train_scen, eval_scen=eval_scen, hist_split=hist_split,
                hist_mean=hist_mean, demand_ref=demand_ref, nr=nr, nc=nc)


_CORDON_OBS = True    # treat network-boundary (external) inflows as observed cordon (C7 sensitivity: set False)


def _obs(net, missing):
    return (approach_observed_mask(net, missing, cordon_observed=_CORDON_OBS),
            exit_observed_mask(net, missing))


def _set_mask(mg, context_np, N, dev):
    import torch
    eval_mask = ~context_np
    eval_idx = np.nonzero(eval_mask)[0]
    comp = mg.component_labels(eval_mask)[eval_idx]
    CA.set_cluster_mask(comp, int(context_np.sum()), N, dev)


_ABLATE = set()   # feature groups to zero out (for ablation): {"anchor","neighbor","static"}
_USE_ROUTED = False   # add boundary-routing features (forward flow assignment)


def _prep(D, missing, s, train_mask_np, dev):
    # NOTE: do NOT use prepare_features_tensor -- its drop_constant_features(train_mask)
    # gives a variable column count as the context set changes per step, breaking the
    # ensemble feature_perm. A fixed-dim quantile-normal transform is used instead.
    import torch
    from graphpfn.inference.preprocessing import NumTransform, apply_num_transform, impute_nans
    oa, oe = _obs(D["net"], missing)
    # LOCAL demand anchor: diffuse observed approach volumes over the graph so each node
    # gets a spatially-local demand ratio (approximates ridge's boundary spatial info).
    from .graph.local_demand import local_demand_ratio, routed_flow
    ratio = local_demand_ratio(D["mg"], oa, D["APPR"][s], D["demand_ref"])
    r_appr, r_count = (None, None)
    if _USE_ROUTED:
        r_appr, r_count = routed_flow(D["net"], oa, D["APPR"][s], D["hist_split"])
    f = assemble_features(D["static"], D["APPR"][s], D["EXIT"][s], oa, oe, D["hist_split"],
                          hist_mean=D["hist_mean"], demand_ratio=ratio,
                          routed_appr=r_appr, routed_count=r_count, zero_groups=_ABLATE)
    f = impute_nans(apply_num_transform(f, NumTransform.QUANTILE_NORMAL, 0))
    return torch.tensor(np.asarray(f), dtype=torch.float32, device=dev)


def eval_model(model, D, mg, graph, real_set, use_cluster, dev, rho=0.0, sign=1,
               return_arrays=False):
    import torch
    from graphpfn import GraphDataset
    from graphpfn.inference.preprocessing import standardize_targets
    from graphpfn.inference.util import apply_model
    from .inject.rho import inject_rho
    net, N = D["net"], D["N"]
    um = assign_ou(net, real_set)
    idx = np.nonzero(um)[0]
    oa, _ = _obs(net, real_set)
    h_nodes = depth_h(mg, um, oa)[idx]
    tmask = ~um
    a_own = D["Y"][D["train_scen"]].mean(0)[idx]     # naive A = clean historical mean
    ds = GraphDataset(name="e", graph=graph, features={"num": np.zeros((N, 1), np.float32)},
                      targets=D["Y"][0], masks={"train": tmask, "val": np.zeros(N, bool),
                                                "test": um.copy()}, task_type="regression")
    seA, seM = [], []
    preds_all, true_all = [], []
    model.eval()
    with torch.no_grad():
        for s in D["eval_scen"]:
            s = int(s)
            # source/sink: inject into U truth only; observed context stays clean (invisible)
            if rho > 0:
                rng = np.random.default_rng((int(s) * 7919 + int(rho * 1000)) & 0x7FFFFFFF)
                true = inject_rho(D["Y"][s], um, mg, rho, sign, rng, cumulative=False)[idx]
            else:
                true = D["Y"][s][idx]
            feats = _prep(D, real_set, s, tmask, dev)   # clean features (no source/sink)
            yc = torch.tensor(D["Y"][s][tmask], device=dev)
            yc_std, stats = standardize_targets(y_train=yc)
            _set_mask(mg, tmask, N, dev) if use_cluster else CA.clear()
            pred = apply_model(model=model, dataset=ds, graph=graph, features=feats,
                               y_train=yc_std, train_mask=torch.tensor(tmask, device=dev),
                               regression_target_stats=stats, amp=True, device=dev)
            seA.append((a_own - true) ** 2)
            seM.append((np.asarray(pred)[idx] - true) ** 2)
            preds_all.append(np.asarray(pred)[idx]); true_all.append(true)
    CA.clear()
    seA, seM = np.mean(seA, 0), np.mean(seM, 0)
    by_h = {int(hv): 1 - np.sqrt(seM[h_nodes == hv].mean()) / np.sqrt(seA[h_nodes == hv].mean())
            for hv in sorted(set(h_nodes.tolist()))}
    overall = float(1 - np.sqrt(seM.mean()) / np.sqrt(seA.mean()))
    out = {"by_h": by_h, "overall": overall}
    if return_arrays:
        out.update(preds=np.array(preds_all), true=np.array(true_all),
                   a_own=a_own, h=h_nodes)
    return out


def train(model, scales_data, graphs, use_cluster, n_steps, lr, rng, dev, verbose=True):
    import torch
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    names = list(scales_data)
    model.train()
    for step in range(n_steps):
        scl = names[rng.integers(len(names))]
        D, mg, graph = scales_data[scl], scales_data[scl]["mg"], graphs[scl]
        s = int(D["train_scen"][rng.integers(len(D["train_scen"]))])
        # only shapes that leave a context border on this scale (context must be non-empty)
        valid = [(h, w) for (h, w) in SHAPES if h <= D["nr"] - 1 and w <= D["nc"] - 1]
        fh, fw = valid[rng.integers(len(valid))]
        r0, c0 = rng.integers(D["nr"] - fh + 1), rng.integers(D["nc"] - fw + 1)
        fake = rect_block(D["net"], int(r0), int(c0), fh, fw)
        um_fake = assign_ou(D["net"], fake)
        ctx_np = ~um_fake
        feats = _prep(D, fake, s, ctx_np, dev)
        y = torch.tensor(D["Y"][s], device=dev)
        ctx = torch.tensor(ctx_np, device=dev); qry = torch.tensor(um_fake, device=dev)
        yc, yq = y[ctx], y[qry]
        mu, sd = yc.mean(), yc.std() + 1e-6
        yc, yq = (yc - mu) / sd, (yq - mu) / sd
        _set_mask(mg, ctx_np, D["N"], dev) if use_cluster else CA.clear()
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds, targets = model.train_step_forward(
                graph=graph, features=feats, y_context=yc, y_query=yq,
                context_node_mask=ctx, query_node_mask=qry, task_type="regression")
            loss = ((preds - targets) ** 2).mean()
        loss.backward(); opt.step()
        if verbose and step % 50 == 0:
            print(f"    [{scl} {fh}x{fw}] step {step} loss={loss.item():.4f}")
    CA.clear()
    return model


def _fresh_model(dev, n_feat):
    from graphpfn.model.graphpfn import GraphPFN
    return GraphPFN.from_pretrained(n_features=n_feat, n_classes=None,
                                    device=dev, checkpoint=CKPT).to(dev)


def main(n_steps=600, lr=2e-5, n_eval_scen=15, real_label="4x4", seed=0, device="cuda:0",
         train_scales=None, eval_scale=None):
    import torch
    CA.patch_encoder()
    dev = torch.device(device)
    torch.manual_seed(seed)

    train_scales = train_scales or TRAIN_SCALES
    eval_scale = eval_scale or EVAL_SCALE
    globals()["EVAL_SCALE"] = eval_scale
    scales_data = {s: _load(s, n_eval_scen, seed) for s in train_scales}
    D_eval = scales_data[eval_scale] if eval_scale in scales_data else _load(eval_scale, n_eval_scen, seed)
    scales_data_all = dict(scales_data); scales_data_all[eval_scale] = D_eval  # graphs for all
    graphs = {s: build_dgl_graph(scales_data_all[s]["mg"], scales_data_all[s]["N"]).to(dev)
              for s in scales_data_all}
    mg_eval, graph_eval = D_eval["mg"], graphs[EVAL_SCALE]
    real_set = set([c for c in generate_cluster_configs(D_eval["net"]) if real_label in c[0]][0][1])

    # feature dim
    n_feat = _prep(D_eval, real_set, int(D_eval["train_scen"][0]), np.ones(D_eval["N"], bool), dev).shape[-1]

    rng = np.random.default_rng(seed)
    results = {}

    print("### E (vanilla, untrained) ###")
    mE = _fresh_model(dev, n_feat)
    results["E"] = eval_model(mE, D_eval, mg_eval, graph_eval, real_set, False, dev)
    del mE; torch.cuda.empty_cache()

    print("### F (domain-pretrained, cluster OFF) ###")
    mF = train(_fresh_model(dev, n_feat), scales_data, graphs, False, n_steps, lr,
               np.random.default_rng(seed), dev)
    results["F"] = eval_model(mF, D_eval, mg_eval, graph_eval, real_set, False, dev)
    del mF; torch.cuda.empty_cache()

    print("### G (domain-pretrained, cluster ON) ###")
    mG = train(_fresh_model(dev, n_feat), scales_data, graphs, True, n_steps, lr,
               np.random.default_rng(seed), dev)
    results["G"] = eval_model(mG, D_eval, mg_eval, graph_eval, real_set, True, dev)
    del mG; torch.cuda.empty_cache()

    print(f"\n########## E->F->G ladder (eval {EVAL_SCALE} {real_label}, held-out scenarios) ##########")
    hs = sorted(results["F"]["by_h"])
    print(f"{'model':>6} | {'overall':>7} | " + " ".join(f"h{h:>1}" for h in hs))
    for m in ["E", "F", "G"]:
        r = results[m]
        print(f"{m:>6} | {r['overall']:>7.2f} | " +
              " ".join(f"{r['by_h'].get(h, float('nan')):>4.2f}" for h in hs))
    return results


if __name__ == "__main__":
    main()
