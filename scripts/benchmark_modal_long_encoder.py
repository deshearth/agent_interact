"""Local compatibility and throughput check on fixed training-only prompts."""
from pathlib import Path
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import modal_inbox_screen as source

ROOT=Path('runs/modal_long_context_screen_320_v1')


def pooled(model,tokenizer,texts,device,pad_to=None):
    batch=tokenizer(texts,return_tensors='pt',padding='max_length' if pad_to else True,max_length=pad_to,
        truncation=False,return_special_tokens_mask=True)
    mask=batch.pop('special_tokens_mask'); mask=batch['attention_mask']*(1-mask)
    values={k:v.to(device) for k,v in batch.items() if k in ('input_ids','attention_mask')}
    assert values['input_ids'].shape[1]<=8192
    with torch.inference_mode():
        h=model(**values).last_hidden_state; mean=(h*mask.to(device)[...,None]).sum(1)/mask.sum(1).to(device)[:,None]
        first=h[:,0]; output=torch.cat([torch.nn.functional.normalize(mean.float(),dim=-1),torch.nn.functional.normalize(first.float(),dim=-1)],-1)/np.sqrt(2)
    return output.cpu().numpy(),batch['attention_mask'].sum(-1).numpy()


def main():
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TOKENIZERS_PARALLELISM']='false'
    from transformers import AutoModel,AutoTokenizer
    info=source.read(ROOT/'encoder_manifest.json')
    for path,sha in info['files'].items(): assert digest(path)==sha,path
    torch.set_num_threads(2); device='mps' if torch.backends.mps.is_available() else 'cpu'
    assert device=='mps','GPU compatibility benchmark must run with Mac GPU access'
    tokenizer=AutoTokenizer.from_pretrained(info['snapshot'],local_files_only=True)
    model=AutoModel.from_pretrained(info['snapshot'],local_files_only=True,attn_implementation='sdpa',dtype=torch.float32).to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    labels=np.load(source.ROOT/'data/labels.npz'); p=source.prepare(); fit=np.flatnonzero(np.isin(labels['qids'],p['inner']['fit']))
    selected=fit[:16]; texts=source.read(source.ROOT/'data/texts.json'); chosen=[texts[i] for i in selected]
    lengths=np.asarray([len(x) for x in tokenizer(chosen,truncation=False)['input_ids']]); pad=int(np.ceil(lengths.max()/256)*256)
    pooled(model,tokenizer,chosen[:2],device,pad)
    start=time.monotonic(); outputs=[]
    for begin in range(0,16,2): outputs.append(pooled(model,tokenizer,chosen[begin:begin+2],device,pad)[0])
    elapsed=time.monotonic()-start; vectors=np.concatenate(outputs); assert np.isfinite(vectors).all()
    gpu_memory=int(torch.mps.driver_allocated_memory()); model.to('cpu'); check,n=pooled(model,tokenizer,chosen[:1],'cpu',pad)
    difference=float(np.abs(check-vectors[:1]).max()); assert difference<1e-4
    json_write(ROOT/'benchmark.json',dict(completed_utc=source.now(),encoder_manifest_sha256=digest(ROOT/'encoder_manifest.json'),
        frozen_encoder_parameters=sum(p.numel() for p in model.parameters()),examples=16,selected_rows=selected.tolist(),pad_tokens=pad,
        tokens=int(lengths.sum()),seconds=elapsed,seconds_per_prompt=elapsed/16,rough_pilot_minutes=elapsed/16*20480/60,
        device=device,dtype='float32',attention='sdpa',gpu_driver_memory_bytes=gpu_memory,cpu_max_vector_difference=difference,
        representation='L2-normalized mean of nonspecial tokens plus normalized first-token vector; concatenate and normalize. Full prompt attention, no windows or truncation.',
        benchmark_code_sha256=digest(__file__),source_calls=0,assessment_loaded=False))
    print(source.read(ROOT/'benchmark.json'),flush=True)


if __name__=='__main__': main()
