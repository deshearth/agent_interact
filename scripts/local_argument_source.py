"""Pinned local Llama source with independent full-sequence samples and KV reuse."""
from argparse import ArgumentParser
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import argument_intervention_pilot as pilot

ROOT = Path("runs/local_llama_intervention_v1")


def sample_seed(identity):
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "little")


class LocalSource:
    def __init__(self):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler
        self.mx, self.step, self.make_cache = mx, generate_step, make_prompt_cache
        self.sampler = make_sampler(temp=.7, top_p=1., min_p=0., top_k=0)
        self.provenance = json.loads((ROOT / "model_provenance.json").read_text())
        path = self.provenance["snapshot_path"]
        mx.set_cache_limit(512 * 1024 ** 2)
        mx.set_memory_limit(16 * 1024 ** 3)
        begin = time.monotonic()
        self.model, self.tokenizer = load(path, tokenizer_config={"trust_remote_code": False})
        self.load_seconds = time.monotonic() - begin
        self.letter_ids = {}
        for letter in ("A", "B"):
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            assert len(ids) == 1
            self.letter_ids[letter] = ids[0]
        self.eos = set(self.tokenizer.eos_token_ids)

    def render(self, prompt):
        tokens = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, date_string="08 Nov 2026")
        assert 1 < len(tokens) <= 8192, "Unexpected context length; do not truncate"
        return tokens

    def prefill(self, tokens):
        mx = self.mx
        cache = self.make_cache(self.model)
        begin = time.monotonic()
        for pos in range(0, len(tokens) - 1, 256):
            chunk = tokens[pos:min(pos + 256, len(tokens) - 1)]
            self.model(mx.array([chunk]), cache=cache)
            mx.eval([c.state for c in cache])
        assert all(c.offset == len(tokens) - 1 for c in cache)
        return cache, time.monotonic() - begin

    def sample(self, tokens, identity, *, cache=None, max_tokens=256):
        mx = self.mx
        mx.random.seed(sample_seed(identity))
        local_cache = copy.deepcopy(cache) if cache is not None else None
        prompt = mx.array(tokens[-1:] if cache is not None else tokens)
        begin = time.monotonic()
        generated, first_probabilities, eos_seen = [], None, False
        generator = self.step(prompt, self.model, max_tokens=max_tokens, sampler=self.sampler,
                              prompt_cache=local_cache, prefill_step_size=256)
        try:
            for token, logprobs in generator:
                if first_probabilities is None:
                    p = mx.softmax(logprobs.astype(mx.float32) / .7)
                    mx.eval(p)
                    pa, pb = float(p[self.letter_ids["A"]]), float(p[self.letter_ids["B"]])
                    first_probabilities = {"first_literal_A_probability": pa, "first_literal_B_probability": pb,
                                           "other_first_token_probability": max(0., 1 - pa - pb)}
                if token in self.eos:
                    eos_seen = True
                    break
                generated.append(int(token))
        finally:
            generator.close()
        text = self.tokenizer.decode(generated)
        sign, strict = pilot.parse_answer(text)
        if not eos_seen:
            sign, strict = None, False
        if cache is not None:
            assert all(c.offset == len(tokens) - 1 for c in cache), "Shared prefix mutated"
        return {"sample_seed": sample_seed(identity), "raw_response": text, "generated_token_ids": generated,
                "parsed_sign": sign, "strict_format": strict, "finish_reason": "stop" if eos_seen else "length",
                "generation_tokens": len(generated) + int(eos_seen), "prompt_tokens": len(tokens),
                "sample_seconds": time.monotonic() - begin, "peak_memory_gb": mx.get_peak_memory() / 1e9,
                **first_probabilities}


def benchmark():
    source = LocalSource()
    rows = pilot.read_jsonl(pilot.ROOT / "prompts.jsonl")
    chosen = [rows[0], rows[1], rows[4], rows[5]]
    results = []
    for row in chosen:
        tokens = source.render(row["prompt"])
        cache, prefill = source.prefill(tokens)
        identity = "benchmark/" + row["prompt_id"]
        cached = source.sample(tokens, identity, cache=cache)
        uncached = source.sample(tokens, identity)
        delta = max(abs(cached[k] - uncached[k]) for k in ("first_literal_A_probability", "first_literal_B_probability"))
        assert delta < 1e-4, delta
        repeat = source.sample(tokens, identity, cache=cache)
        assert cached["generated_token_ids"] == repeat["generated_token_ids"]
        results.append({"prompt_id": row["prompt_id"], "prefill_seconds": prefill,
                        "cache_probability_max_difference": delta,
                        "same_seed_cached_uncached_tokens_match": cached["generated_token_ids"] == uncached["generated_token_ids"],
                        "same_seed_cache_reuse_tokens_match": True, "cached": cached, "uncached": uncached})
        print(json.dumps({"prompt_id": row["prompt_id"], "prefill_seconds": prefill,
                          "cached_seconds": cached["sample_seconds"], "answer": cached["raw_response"],
                          "cache_probability_max_difference": delta}), flush=True)
    record = {"status": "complete", "source_code_sha256": pilot.sha(__file__),
              "model_provenance_sha256": pilot.sha(ROOT / "model_provenance.json"),
              "load_seconds": source.load_seconds, "completed_utc": datetime.now(timezone.utc).isoformat(),
              "device": str(source.mx.default_device()), "memory_limit_bytes": 16 * 1024 ** 3,
              "paid_api_calls": 0, "local_source_samples": 12, "results": results,
              "first_token_metric_note": "Exact first bare A/B token probabilities at temperature .7. Other first tokens are not renormalized away. This diagnostic is distinct from parsed full-response and majority-of-five probabilities."}
    pilot.write_json(ROOT / "benchmark.json", record)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("benchmark",))
    parser.parse_args()
    benchmark()
