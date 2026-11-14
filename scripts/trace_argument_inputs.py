"""Read-only computational trace from individual message vectors to receiver logits."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import config,digest,json_write
from scripts import argument_mechanisms as previous
from scripts import question_context as question
from scripts import joint_argument_context as joint
from scripts import joint_argument_encoding as source
from scripts import joint_argument_encoding_windowed as encoding
from scripts.argument_features import load_cache

ROOT = source.SESSION/"input_trace"


def main():
    torch.set_num_threads(2)
    cfg = config(question.CONFIG)
    arrays,index,vectors,_ = load_cache(cfg,verify=True)
    index = previous.split_index(index,cfg)
    donors = np.load(previous.ROOT/"donors_s42.npy",allow_pickle=False)
    examples = []
    for role in ("train","unseen"):
        e = next(e for e,r in enumerate(index) if r["diagnostic_role"] == role)
        degree = (arrays["signs"][e] != 0).sum(-1)
        receiver = int(np.flatnonzero(degree >= 3)[0])
        examples.append((e,0,receiver))
    traces = []
    for label,module,kind in (("original_pca3",previous,"A2"),("full_question",question,"full_embedding"),
                               ("joint_conditioning",joint,"question_conditioned")):
        a,vec = arrays,vectors
        if module is not previous:
            a = question.augmented_arrays(arrays,index,question.ROOT/"question_features.npz","full_embedding")
        if module is joint:
            with np.load(source.ROOT/"ids.npz",allow_pickle=False) as saved:
                a = {**a,"text_ids":saved["text_ids"]}
            vec = np.load(encoding.ROOT/"question_conditioned.npy",mmap_mode="r",allow_pickle=False)
        items = []
        for e,t,r in examples:
            ds = previous.SplitSnapshots(a,index,vec,index[e]["diagnostic_role"],"full_arguments",donors)
            items.append(ds[ds.pairs.index((e,t))])
        batch = default_collate(items)
        batch.pop("target")
        batch["text"].requires_grad_(True)
        run = module.ROOT/f"{kind}-full_arguments-s42"
        manifest = json.loads((run/"manifest.json").read_text())
        assert digest(run/"best.pt") == manifest["artifact_sha256"]["best.pt"]
        model = module.make_model(config(module.CONFIG),kind).eval()
        model.load_state_dict(torch.load(run/"best.pt",map_location="cpu",weights_only=False)["model"])
        logits = model(batch)
        receivers = torch.tensor([r for e,t,r in examples])
        focus = logits[torch.arange(len(examples)),receivers]
        grad, = torch.autograd.grad((focus[:,1]-focus[:,0]).sum(),batch["text"])
        norms = grad.norm(dim=-1).detach().numpy()
        for b,(e,t,receiver) in enumerate(examples):
            mask = a["signs"][e,receiver] != 0
            assert (norms[b,receiver,mask] > 1e-10).all()
            allowed = np.zeros_like(norms[b],dtype=bool);allowed[receiver] = mask
            assert not norms[b][~allowed].any()
            messages = [{"sender":int(a["neighbors"][e,receiver,k]),"relationship_sign":int(a["signs"][e,receiver,k]),
                "stance":int(a["stance"][e,t,receiver,k]),"text_id":int(a["text_ids"][e,t,receiver,k]),
                "input_l2":float(batch["text"][b,receiver,k].detach().norm()),"logit_gradient_l2":float(norms[b,receiver,k])}
                for k in np.flatnonzero(mask)]
            traces.append({"model":label,"best_sha256":digest(run/"best.pt"),"role":index[e]["diagnostic_role"],
                "episode_id":index[e]["episode_id"],"round":t,"receiver":receiver,"messages":messages,
                "all_delivered_vectors_have_nonzero_local_derivative":True,
                "other_receivers_and_nonedges_have_zero_derivative":True})
    ROOT.mkdir(exist_ok=True)
    json_write(ROOT/"trace.json",{"code_sha256":digest(__file__),"predictor_states_changed":False,"optimizer_steps":0,
        "method":"Differentiate receiver A-minus-B logit with respect to every message-vector coordinate, with all model weights fixed and next-state targets removed.",
        "meaning":"Verifies per-message computational input paths in six examples. A nonzero derivative is not proof of useful reasoning, generalization or causal source-LLM influence. Raw tokens are processed by the separate frozen encoder.","traces":traces})
    print(json.dumps({"traces":len(traces),"all_active_edges_connected":True,"off_receiver_message_gradients_zero":True}),flush=True)


if __name__ == "__main__":main()
