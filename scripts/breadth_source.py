"""Local source generation with batched prefill and independent sample RNGs.

Uses the existing pinned Llama weights and prompt rendering. No global RNG state
or sampled continuation is shared between requested responses.
"""
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.local_argument_source import LocalSource, sample_seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n")
    temp.replace(path)


class BreadthSource(LocalSource):
    def __init__(self, precision="float32"):
        super().__init__()
        self.precision = precision
        if precision == "float32":
            self.model.set_dtype(self.mx.float32)
            self.mx.eval(self.model.parameters())
        elif precision != "native":
            raise ValueError(precision)

    def independent_sampler(self, identity):
        mx = self.mx
        counter = 0

        def sampler(logprobs):
            nonlocal counter
            key = mx.random.key(sample_seed(f"{identity}|token{counter}"))
            counter += 1
            return mx.random.categorical(logprobs / .7, key=key)
        return sampler

    def prefill_many(self, tokens, batch_size):
        from mlx_lm.generate import BatchGenerator
        gen = BatchGenerator(self.model, max_tokens=1,
                             completion_batch_size=batch_size,
                             prefill_batch_size=batch_size, prefill_step_size=256)
        uids = gen.insert(tokens)
        pending, saved = set(uids), {}
        begin = time.monotonic()
        try:
            while pending:
                prompts, generated = gen.next()
                assert not generated, "Prefill must stop before any sampled token"
                ready = [r.uid for r in prompts
                         if r.end_of_segment and not r.end_of_prompt
                         and r.progress[0] == len(tokens[uids.index(r.uid)]) - 1]
                saved.update(gen.remove(ready, return_prompt_caches=True))
                pending.difference_update(ready)
            self.mx.eval([[c.state for c in saved[uid][0]] for uid in uids])
        finally:
            gen.close()
        for uid, token_ids in zip(uids, tokens):
            assert saved[uid][1] == token_ids[:-1]
            assert all(c.offset == len(token_ids)-1 for c in saved[uid][0])
        return [saved[uid][0] for uid in uids], time.monotonic()-begin

    def generate_many(self, prompts, identities, *, batch_size=4, max_tokens=256):
        """Identical prompts share only a deterministic, immutable prefix cache."""
        from mlx_lm.generate import BatchGenerator
        from scripts.argument_intervention_pilot import parse_answer
        assert len(prompts) == len(identities) == len(set(identities))
        unique = list(dict.fromkeys(prompts))
        assert len(unique) <= batch_size, "Caller must bound the unique prompt group"
        rendered = [self.render(p) for p in unique]
        if max(map(len, rendered)) > 4096 and len(unique) > 1:
            parts=[]
            for p in unique:
                positions=[i for i,value in enumerate(prompts) if value==p]
                parts.append(self.generate_many([p]*len(positions),[identities[i] for i in positions],batch_size=1,max_tokens=max_tokens))
            by_id={row['sample_id']:row for part in parts for row in part['responses']}
            return {**parts[0], 'responses':[by_id[sid] for sid in identities],
                    'prompts':[row for part in parts for row in part['prompts']],
                    'prefill_seconds':sum(p['prefill_seconds'] for p in parts),
                    'generation_seconds':sum(p['generation_seconds'] for p in parts),
                    'peak_memory_gb':max(p['peak_memory_gb'] for p in parts),
                    'long_context_serial_fallback':True}
        if max(map(len,rendered)) > 4096:
            batch_size=1
        caches, prefill_seconds = self.prefill_many(rendered, batch_size)
        lookup = {p:i for i,p in enumerate(unique)}
        index = [lookup[p] for p in prompts]
        rows=[]
        begin = time.monotonic()
        # Bound live copies to one decode wave, even when eight samples share
        # each of four prompts. Pending requests must not duplicate all caches.
        for offset in range(0,len(identities),batch_size):
            wave_index=index[offset:offset+batch_size]
            wave_ids=identities[offset:offset+batch_size]
            gen = BatchGenerator(self.model, max_tokens=max_tokens,
                             stop_tokens=[[t] for t in self.eos],
                             completion_batch_size=batch_size,
                             prefill_batch_size=batch_size, prefill_step_size=256)
            uids = gen.insert([rendered[i][-1:] for i in wave_index],
                          caches=[copy.deepcopy(caches[i]) for i in wave_index],
                          all_tokens=[rendered[i][:-1] for i in wave_index],
                          samplers=[self.independent_sampler(x) for x in wave_ids])
            wave = {uid:{"sample_id":wave_ids[j], "generated_token_ids":[],
                     "prompt_tokens":len(rendered[wave_index[j]]),
                     "sample_seed":sample_seed(wave_ids[j])}
                for j,uid in enumerate(uids)}
            try:
                while responses := gen.next_generated():
                    for response in responses:
                        row = wave[response.uid]
                        if "first_literal_A_probability" not in row:
                            p = self.mx.softmax(response.logprobs.astype(self.mx.float32)/.7)
                            self.mx.eval(p)
                            pa, pb = float(p[self.letter_ids['A']]), float(p[self.letter_ids['B']])
                            row.update(first_literal_A_probability=pa,
                                   first_literal_B_probability=pb,
                                   other_first_token_probability=max(0.,1-pa-pb))
                        if response.finish_reason != "stop":
                            row["generated_token_ids"].append(int(response.token))
                        if response.finish_reason is not None:
                            row["finish_reason"] = response.finish_reason
            finally:
                gen.close()
            rows.extend(wave[uid] for uid in uids)
        generation_seconds = time.monotonic()-begin
        for row in rows:
            assert "finish_reason" in row
            row["raw_response"] = self.tokenizer.decode(row["generated_token_ids"])
            sign, strict = parse_answer(row["raw_response"])
            row["parsed_sign"] = sign if row["finish_reason"] == "stop" else None
            row["strict_format"] = strict and row["finish_reason"] == "stop"
            row["generation_tokens"] = len(row["generated_token_ids"])+int(row["finish_reason"]=="stop")
        for cache, token_ids in zip(caches, rendered):
            assert all(c.offset == len(token_ids)-1 for c in cache)
        result = {"responses":rows,
                  "prompts":[{"prompt":p,"prompt_sha256":text_sha(p),"token_ids":t}
                             for p,t in zip(unique,rendered)],
                  "prefill_seconds":prefill_seconds,"generation_seconds":generation_seconds,
                  "peak_memory_gb":self.mx.get_peak_memory()/1e9,
                  "batch_size":batch_size,"max_tokens":max_tokens,"precision":self.precision,
                  "source_code_sha256":sha(__file__),
                  "base_source_sha256":sha("scripts/local_argument_source.py"),
                  "completed_utc":datetime.now(timezone.utc).isoformat()}
        del caches, gen
        self.mx.clear_cache()
        return result


def benchmark():
    from scripts.local_conversation_pilot import formatter, opinion_prompt, message_prompt
    from scripts.argument_intervention_pilot import read_jsonl
    root = Path("runs/question_breadth_v1")
    questions = read_jsonl("data/raw/upstream/data/obj/train.jsonl")
    personas = json.loads(Path("data/raw/upstream/data/obj/personas.json").read_text())
    fmt, source = formatter(), BreadthSource()
    prompts = [opinion_prompt(fmt, personas[i]["persona"], questions[i], [], []) for i in range(4)]
    ids = [f"breadth_benchmark|{i}" for i in range(4)]
    serial = [source.generate_many([p],[sid],batch_size=1) for p,sid in zip(prompts,ids)]
    batched = source.generate_many(prompts, ids)
    repeated = source.generate_many(prompts, ids)
    maximum = 0.
    for old,new,repeat in zip(serial,batched["responses"],repeated["responses"]):
        old = old["responses"][0]
        for key in ['first_literal_A_probability','first_literal_B_probability']:
            maximum = max(maximum,abs(old[key]-new[key]))
        assert new["generated_token_ids"] == repeat["generated_token_ids"]
    # Compare distributions against the already validated sequential source.
    originals=[]
    for i,p in enumerate(prompts):
        tokens=source.render(p)
        original=source.sample(tokens,f"breadth_original_check|{i}",max_tokens=256)
        originals.append(original)
    diagnostics={"serial_pA":[x['responses'][0]['first_literal_A_probability'] for x in serial],
                 "batch_pA":[x['first_literal_A_probability'] for x in batched['responses']],
                 "original_pA":[x['first_literal_A_probability'] for x in originals],
                 "max_probability_difference":maximum,"precision":source.precision}
    write_json(root/'source_benchmark_diagnostics.json',diagnostics)
    print(json.dumps(diagnostics),flush=True)
    assert maximum < 1e-3, maximum
    for original,bat in zip(originals,batched['responses']):
        for key in ['first_literal_A_probability','first_literal_B_probability']:
            assert abs(original[key]-bat[key]) < 1e-3
    group=source.generate_many([prompts[0]]*8,[f"breadth_repeat8|{i}" for i in range(8)])
    assert len({r['sample_seed'] for r in group['responses']}) == 8
    message_prompts=[message_prompt(fmt,personas[i]['persona'],questions[i],[],[],1,f'bench_msg{i}')[0] for i in range(4)]
    messages=source.generate_many(message_prompts,[f'breadth_message|{i}' for i in range(4)])
    result={"status":"complete","serial":serial,"batched":batched,"repeat":repeated,
            "original_source":originals,"eight_independent_samples":group,"messages":messages,
            "max_serial_batch_first_token_probability_difference":maximum,
            "source_code_sha256":sha(__file__),"load_seconds":source.load_seconds,
            "paid_api_calls":0}
    write_json(root/"source_benchmark.json",result)
    print(json.dumps({"status":"complete","max_probability_difference":maximum,
          "serial_seconds":sum(x['prefill_seconds']+x['generation_seconds'] for x in serial),
          "batch_seconds":batched['prefill_seconds']+batched['generation_seconds'],
          "eight_sample_seconds":group['prefill_seconds']+group['generation_seconds'],
          "four_message_seconds":messages['prefill_seconds']+messages['generation_seconds'],
          "peak_memory_gb":messages['peak_memory_gb']}),flush=True)


if __name__ == "__main__":
    benchmark()
