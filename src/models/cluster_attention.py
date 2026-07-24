"""Model G (cluster-aware attention) as an inference-time patch on vanilla GraphPFN.

Vanilla PFN sample-attention is structural: query (test) tokens use x_kv = x[:, :eval_pos]
(train only), so query<->query is impossible. This patches EncoderBaseLayer.call_sequence_attention
so the test attention uses x_kv = full sequence and a float mask that lets each query attend to
(a) all train tokens (as before) and (b) other query tokens in the same missing component.
Labels never enter (query y is NaN-masked upstream) -> only hidden/feature state is shared, per
the blueprint's "structure not label" rule.

CAVEAT (train/inference mismatch): vanilla weights were pretrained with query<->query blocked,
so opening it at inference is a LOWER BOUND on G's potential (a fair test pretrains with it, = F+G).

IMPORTANT: call patch_encoder() BEFORE the model is built (predict_icl constructs it), because
EncoderBaseLayer binds self.call_sequence_attention into partials at __init__.
"""

import torch

# shared holder read at forward time
CLUSTER = {"active": False, "mask": None}   # mask: float[n_query, n_total] on device


def set_cluster_mask(comp_query, n_train, n_total, device, dtype=torch.float32):
    """Build the float attn mask. comp_query[j] = component id of the j-th query node
    (in transformer eval order = ascending U index). 0 = attend, -inf = blocked."""
    import numpy as np
    cq = torch.as_tensor(np.asarray(comp_query), device=device)
    nq = len(cq)
    same = (cq[:, None] == cq[None, :])                      # [nq, nq] same component
    qq = torch.where(same, 0.0, float("-inf")).to(dtype)
    mask = torch.zeros((nq, n_total), device=device, dtype=dtype)
    mask[:, n_train:] = qq                                   # test-key columns
    mask[:, :n_train] = 0.0                                  # all train keys allowed
    CLUSTER["mask"] = mask
    CLUSTER["active"] = True


def clear():
    CLUSTER["active"] = False
    CLUSTER["mask"] = None


def patch_encoder():
    """Replace EncoderBaseLayer.call_sequence_attention with a cluster-aware version."""
    from graphpfn._vendor.limix.model.layer import EncoderBaseLayer

    if getattr(EncoderBaseLayer, "_cluster_patched", False):
        return

    def call_sequence_attention(self, x, feature_atten_mask, eval_pos,
                                index=0, calculate_sample_attention=False):
        idx1 = index * 2 if self.seq_attn_isolated else index
        idx2 = idx1 + 1 if self.seq_attn_isolated else idx1

        x_train = self.sequence_attentions[idx1](
            x=x[:, :eval_pos].transpose(1, 2),
            x_kv=x[:, :eval_pos].transpose(1, 2),
            copy_first_head_kv=bool(self.self_share_all_kv_heads),
        )[0].transpose(1, 2)

        if self.seq_attn_serial:
            x[:, :eval_pos] = x_train

        sample_attention = None
        if eval_pos < x.shape[1]:
            if CLUSTER["active"] and CLUSTER["mask"] is not None:
                m = CLUSTER["mask"].to(x.dtype)
                x_test, _, sample_attention = self.sequence_attentions[idx2](
                    x=x[:, eval_pos:].transpose(1, 2),
                    x_kv=x.transpose(1, 2),                       # full sequence as keys
                    copy_first_head_kv=bool(self.cross_share_all_kv_heads),
                    attn_mask=m,                                  # [n_query, n_total]
                    calculate_sample_attention=calculate_sample_attention,
                )
            else:
                x_test, _, sample_attention = self.sequence_attentions[idx2](
                    x=x[:, eval_pos:].transpose(1, 2),
                    x_kv=x[:, :eval_pos].transpose(1, 2),
                    copy_first_head_kv=bool(self.cross_share_all_kv_heads),
                    calculate_sample_attention=calculate_sample_attention,
                )
            x_test = x_test.transpose(1, 2)
            return torch.cat([x_train, x_test], dim=1), None, sample_attention
        return x_train

    EncoderBaseLayer.call_sequence_attention = call_sequence_attention
    EncoderBaseLayer._cluster_patched = True
