"""Generate the 100-cohort (10 models x 10 sources) distillation dataset, char-level.
Each cohort = model M generating ~--chars-per-cohort chars in source-style S, via vLLM.
Builds a char dataset (vocab capped to the top chars), per-cohort train/val bins + meta,
tars and uploads to HF. The substrate for the 'Distillation with Credit' demo.

All 10 teachers are Apache-2.0 / MIT (distillation-legal).
  python gen_distill.py --chars-per-cohort 1000000
"""
import argparse, os, pickle, tarfile, time, gc
from collections import Counter
from pathlib import Path
import numpy as np

MODELS = [  # (short, hf_id, is_chat)
    ("qwen2.5-3b",      "Qwen/Qwen2.5-3B-Instruct",            True),
    ("mistral-7b",      "mistralai/Mistral-7B-Instruct-v0.3",  True),
    ("phi-3.5",         "microsoft/Phi-3.5-mini-instruct",     True),
    ("r1-distill-1.5b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", True),
    ("smollm2-1.7b",    "HuggingFaceTB/SmolLM2-1.7B-Instruct", True),
    ("olmo2-7b",        "allenai/OLMo-2-1124-7B-Instruct",     True),
    ("tinyllama",       "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  True),
    ("pythia-1.4b",     "EleutherAI/pythia-1.4b",              False),
    ("zephyr-7b",       "HuggingFaceH4/zephyr-7b-beta",        True),
    ("qwen-coder-1.5b", "Qwen/Qwen2.5-Coder-1.5B-Instruct",    True),
]
SOURCES = {
    "shakespeare": "Write a long original scene from a Shakespearean play in Elizabethan English, characters speaking in verse. Begin immediately, no preamble.",
    "tinystories": "Write a simple gentle story for a 4-year-old using only very basic words. Begin immediately, no preamble.",
    "python":      "Write a complete, well-commented Python program that does something interesting. Output only code.",
    "news":        "Write a detailed news article in the style of a major newspaper. Begin immediately, no preamble.",
    "recipe":      "Write a detailed cooking recipe with an ingredient list and numbered steps. Begin immediately.",
    "science":     "Write the abstract and introduction of a scientific research paper. Begin immediately, no preamble.",
    "legal":       "Write a detailed section of a legal contract or terms-of-service document. Begin immediately.",
    "poetry":      "Write several original poems of varying forms and styles. Begin immediately, no preamble.",
    "fantasy":     "Write a long passage of epic fantasy fiction with rich description. Begin immediately, no preamble.",
    "dialogue":    "Write an extended screenplay dialogue between several characters. Begin immediately, no preamble.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars-per-cohort", type=int, default=1_000_000)
    ap.add_argument("--out-dir", default="data_100distill")
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-models", type=int, default=10)
    ap.add_argument("--vocab-cap", type=int, default=512)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    out = Path(args.out_dir); (out / "raw").mkdir(parents=True, exist_ok=True)
    sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=args.max_tokens)
    cohort_text = {}

    for short, hf_id, is_chat in MODELS[:args.max_models]:
        print(f"=== loading {short} ({hf_id}) ===", flush=True)
        try:
            llm = LLM(model=hf_id, dtype="bfloat16", gpu_memory_utilization=0.92,
                      max_model_len=2048, trust_remote_code=True, enforce_eager=True)
        except Exception as e:
            print(f"  !! FAILED to load {short}: {str(e)[:160]}", flush=True); continue
        for src, prompt in SOURCES.items():
            name = f"{short}__{src}"; t0 = time.time(); chunks = []; chars = 0; rounds = 0
            while chars < args.chars_per_cohort and rounds < 300:
                try:
                    if is_chat:
                        outs = llm.chat([[{"role": "user", "content": prompt}]] * args.batch, sp, use_tqdm=False)
                    else:
                        outs = llm.generate([prompt] * args.batch, sp, use_tqdm=False)
                except Exception as e:
                    print(f"  gen error {name}: {str(e)[:100]}", flush=True); break
                for o in outs:
                    txt = o.outputs[0].text
                    chunks.append(txt); chars += len(txt)
                rounds += 1
            text = "\n\n".join(chunks)[:args.chars_per_cohort]
            cohort_text[name] = text
            (out / "raw" / f"{name}.txt").write_text(text)
            print(f"  {name}: {len(text):,} chars | {rounds} rounds | {time.time()-t0:.0f}s", flush=True)
        del llm; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

    # --- build char dataset (cap vocab to top-N chars; rest -> '?') ---
    freq = Counter()
    for t in cohort_text.values():
        freq.update(t)
    keep = [c for c, _ in freq.most_common(args.vocab_cap)]
    if "?" not in keep:
        keep.append("?")
    allchars = sorted(set(keep))
    stoi = {c: i for i, c in enumerate(allchars)}; itos = {i: c for c, i in stoi.items()}
    unk = stoi["?"]
    names = sorted(cohort_text)
    for name in names:
        ids = np.fromiter((stoi.get(c, unk) for c in cohort_text[name]), dtype=np.uint16,
                          count=len(cohort_text[name]))
        nval = max(2, int(len(ids) * 0.01))
        ids[nval:].tofile(out / f"{name}_train.bin")
        ids[:nval].tofile(out / f"{name}_val.bin")
    meta = {"vocab_size": len(allchars), "dtype": "uint16", "cohort_names": names,
            "stoi": stoi, "itos": itos}
    pickle.dump(meta, open(out / "meta.pkl", "wb"))
    print(f"vocab {len(allchars)} | {len(names)} cohorts | "
          f"total {sum(len(t) for t in cohort_text.values())/1e6:.0f}M chars", flush=True)

    with tarfile.open("data_100distill.tar.gz", "w:gz") as tar:
        for f in sorted(out.glob("*_train.bin")) + sorted(out.glob("*_val.bin")) + [out / "meta.pkl"]:
            tar.add(f, arcname=f.name)
    from huggingface_hub import HfApi
    HfApi(token=os.environ["HF_TOKEN"]).upload_file(
        path_or_fileobj="data_100distill.tar.gz", path_in_repo="data_100distill/bins.tar.gz",
        repo_id=os.environ.get("HF_REPO", "iamtrask/abcGPT-nano-3"), repo_type="model")
    print("uploaded data_100distill/bins.tar.gz  DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
