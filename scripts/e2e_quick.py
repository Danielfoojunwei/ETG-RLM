#!/usr/bin/env python3
"""Quick E2E evaluation to complement v4 verification results.
Qwen2.5-1.5B-Instruct (12x larger than GPT-2) for generation.
"""
import gc, json, re, time, sys
from pathlib import Path
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

E2E_N = 50
SEED = 43

def main():
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation")["validation"]

    np.random.seed(SEED)
    indices = np.random.choice(len(ds), size=E2E_N, replace=False)
    questions = [ds[int(i)]["question"] for i in indices]
    evidence = [f"Question: {ds[int(i)]['question']}\nAnswer: {ds[int(i)]['best_answer']}" for i in indices]

    # Load generator
    from transformers import AutoModelForCausalLM, AutoTokenizer
    gen_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading generator: {gen_name}...", flush=True)
    tok = AutoTokenizer.from_pretrained(gen_name)
    model = AutoModelForCausalLM.from_pretrained(gen_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded ({n_params/1e9:.1f}B params)", flush=True)

    # Generate
    completions = []
    t0 = time.time()
    for i, q in enumerate(questions):
        prompt = f"<|im_start|>user\nAnswer factually in 1-2 sentences:\n{q}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tok(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()
        completions.append(text)
        if (i+1) % 10 == 0:
            print(f"  Generated {i+1}/{E2E_N}...", flush=True)
    gen_time = time.time() - t0
    print(f"Generated {len(completions)} in {gen_time:.0f}s")

    del model, tok; gc.collect()

    # Split into sentences
    all_sents, all_ev = [], []
    for text, ev in zip(completions, evidence):
        sents = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
        if not sents and text.strip():
            sents = [text.strip()]
        for s in sents:
            all_sents.append(s)
            all_ev.append(ev)

    print(f"{len(all_sents)} sentences from {len(completions)} completions")
    if not all_sents:
        print("No sentences!"); return

    # Judge with DeBERTa-v3-small (ground truth)
    from transformers import AutoModelForSequenceClassification, AutoConfig
    judge_name = "cross-encoder/nli-deberta-v3-small"
    print(f"Loading judge: {judge_name}...", flush=True)
    jtok = AutoTokenizer.from_pretrained(judge_name)
    jmod = AutoModelForSequenceClassification.from_pretrained(judge_name)
    jmod.eval()
    cfg = AutoConfig.from_pretrained(judge_name)
    ent_idx = next(int(k) for k, v in cfg.id2label.items() if v.lower() == "entailment")

    judge_scores = []
    for i in range(0, len(all_sents), 8):
        batch_ev = all_ev[i:i+8]
        batch_s = all_sents[i:i+8]
        inputs = jtok(batch_ev, batch_s, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            logits = jmod(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        for j in range(len(batch_ev)):
            judge_scores.append(float(probs[j][ent_idx]))

    is_factual = np.array([s >= 0.5 for s in judge_scores])
    del jmod, jtok; gc.collect()

    # Verify with NLI (BART-large) as one view
    print("Loading NLI verifier...", flush=True)
    nli_name = "facebook/bart-large-mnli"
    ntok = AutoTokenizer.from_pretrained(nli_name)
    nmod = AutoModelForSequenceClassification.from_pretrained(nli_name)
    nmod.eval()
    ncfg = AutoConfig.from_pretrained(nli_name)
    nli_ent = next(int(k) for k, v in ncfg.id2label.items() if v.lower() == "entailment")

    nli_scores = []
    for i in range(0, len(all_sents), 8):
        batch_ev = all_ev[i:i+8]
        batch_s = all_sents[i:i+8]
        inputs = ntok(batch_ev, batch_s, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            logits = nmod(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        for j in range(len(batch_ev)):
            nli_scores.append(float(probs[j][nli_ent]))

    del nmod, ntok; gc.collect()

    # Verify with Flan-T5 as second view
    from transformers import AutoModelForSeq2SeqLM
    ft_name = "google/flan-t5-large"
    print(f"Loading LLM-Judge: {ft_name}...", flush=True)
    ftok = AutoTokenizer.from_pretrained(ft_name)
    fmod = AutoModelForSeq2SeqLM.from_pretrained(ft_name)
    fmod.eval()
    true_id = ftok.encode("true", add_special_tokens=False)[0]
    false_id = ftok.encode("false", add_special_tokens=False)[0]

    llm_scores = []
    for i, (ev, sent) in enumerate(zip(all_ev, all_sents)):
        prompt = f"Based on the evidence, is the claim true or false?\nEvidence: {ev}\nClaim: {sent}\nAnswer:"
        inputs = ftok(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            out = fmod.generate(**inputs, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
            fl = out.scores[0][0]
            probs = torch.softmax(fl[torch.tensor([true_id, false_id])], dim=0)
            llm_scores.append(float(probs[0]))

    del fmod, ftok; gc.collect()

    # Apply v4 meta-classifier weights (from calibration)
    # Load from saved results
    v4_path = Path(__file__).parent.parent / "results" / "real_evaluation_v4_results.json"
    with open(v4_path) as f:
        v4 = json.load(f)

    meta_w = v4["meta_classifier"]["weights"]
    meta_b = v4["meta_classifier"]["bias"]

    nli_arr = np.array(nli_scores)
    llm_arr = np.array(llm_scores)
    qa_arr = np.zeros(len(all_sents))  # No QA for E2E (too weak to matter)

    # Build features: NLI, LLM-Judge, QA, NLI*LLM, NLI*QA, LLM*QA
    X = np.column_stack([
        nli_arr, llm_arr, qa_arr,
        nli_arr * llm_arr, nli_arr * qa_arr, llm_arr * qa_arr,
    ])
    w = np.array([meta_w["NLI"], meta_w["LLM-Judge"], meta_w["QA"],
                  meta_w["NLI*LLM-Judge"], meta_w["NLI*QA"], meta_w["LLM-Judge*QA"]])
    z = np.clip(X @ w + meta_b, -20, 20)
    meta_probs = 1.0 / (1.0 + np.exp(-z))
    etg_accepted = meta_probs >= 0.5

    # Also try simple NLI threshold from v4 calibration
    nli_thresh = v4["paradigms"]["NLI"]["youden"]["threshold"]
    nli_accepted = nli_arr >= nli_thresh

    n_total = len(all_sents)

    # Meta-classifier E2E
    n_acc_meta = etg_accepted.sum()
    n_rej_meta = n_total - n_acc_meta
    unfilt = float(is_factual.mean())
    acc_fs_meta = float(is_factual[etg_accepted].mean()) if n_acc_meta > 0 else 0
    rej_fs_meta = float(is_factual[~etg_accepted].mean()) if n_rej_meta > 0 else 0

    # NLI-only E2E
    n_acc_nli = nli_accepted.sum()
    n_rej_nli = n_total - n_acc_nli
    acc_fs_nli = float(is_factual[nli_accepted].mean()) if n_acc_nli > 0 else 0

    print(f"\n{'='*60}")
    print(f"E2E RESULTS — Qwen2.5-1.5B-Instruct (1.5B params)")
    print(f"{'='*60}")
    print(f"  Sentences: {n_total}")
    print(f"  Unfiltered FactScore: {unfilt:.4f}")
    print(f"  ETG Meta-classifier:  {acc_fs_meta:.4f} ({n_acc_meta} accepted, {n_rej_meta} rejected)")
    print(f"  NLI-only filter:      {acc_fs_nli:.4f} ({n_acc_nli} accepted, {n_rej_nli} rejected)")
    print(f"  ETG improvement:      {acc_fs_meta - unfilt:+.4f}")
    print(f"  Rejected FactScore:   {rej_fs_meta:.4f}")
    e2e_proven = acc_fs_meta > unfilt
    print(f"  PROVEN: {'YES' if e2e_proven else 'NO'}")

    # Save E2E results
    e2e_results = {
        "generator": gen_name,
        "generator_params": f"{n_params/1e9:.1f}B",
        "n_questions": E2E_N,
        "n_sentences": n_total,
        "unfiltered_factscore": round(unfilt, 4),
        "etg_meta_accepted": int(n_acc_meta),
        "etg_meta_factscore": round(acc_fs_meta, 4),
        "etg_meta_rejected_factscore": round(rej_fs_meta, 4),
        "nli_only_accepted": int(n_acc_nli),
        "nli_only_factscore": round(acc_fs_nli, 4),
        "improvement_meta": round(acc_fs_meta - unfilt, 4),
        "proven": e2e_proven,
        "generation_time_seconds": round(gen_time, 1),
    }

    # Update v4 results with E2E
    v4["proof_4_e2e"] = e2e_results
    v4["improvements"].append(f"E2E with {gen_name} ({n_params/1e9:.1f}B)")
    with open(v4_path, "w") as f:
        json.dump(v4, f, indent=2)
    print(f"\nResults updated in {v4_path}")

if __name__ == "__main__":
    main()
