# Paper citations and reproducibility

This file is the authoritative record for the **primary benchmark
numbers** that appear in `StagnationGate` and `IntegrityGate` docstrings
— the convergence, false-stagnation, loop accuracy figures (StagnationGate)
and the Paper 4 §4 Table 3 detection/repair percentages (IntegrityGate).
Secondary numbers that show up in prose (false-positive rates, naive
baselines for specific scenarios, scope-dependent preservation counts
like "3/3 or 5/5") are quoted from the same Paper 4 §4.3 table and
Paper 5 §3 text cited here; their exact values live in those source
documents rather than being re-tabled in this file.

## Authoritative data

| Paper § | Authoritative artifact | Commit |
|---|---|---|
| Paper 4 §4.3 — Epiplexity (real embeddings) | `operon-ai:eval/results/benchmarks_real_embeddings/multi_model_summary.json` | `339875e` |
| Paper 4 §4.3 — Epiplexity (mock embeddings) | `operon-ai:eval/benchmarks/run_benchmarks.py --benchmark epiplexity --seed 42 --trials 100` | current `main` |
| Paper 4 §4 Table 3 — State integrity | `operon-ai:article/paper4/sections/04-results.tex` lines 306–322 | current `main` |
| Paper 4 PDF | `operon-ai:article/paper4/main.pdf` | current `main` |

Repo root for path prefixes: `https://github.com/coredipper/operon`.

---

## StagnationGate → Paper 4 §4.3

### Verbatim paper quote

> "Switching to all-MiniLM-L6-v2 sentence embeddings flips the result
> on convergence discrimination. The biological monitor achieves
> 96.0% accuracy on both convergence and false-stagnation scenarios,
> versus 40.1% and 2.0% for the naive detector. The naive detector's
> false-positive rate on false stagnation reaches 98.0%—it sees similar
> outputs and screams 'stagnant' regardless of the agent's confidence
> level."
>
> — Paper 4 §4.3, "With real embeddings: biological wins on convergence."

### Numbers cited in gate docstring

From `multi_model_summary.json` (commit `339875e`), `all-MiniLM-L6-v2`
block (N = 3 seeds × 100 trials = 300):

| Scenario | Bio | Naive |
|---|---|---|
| `convergence_bio` / `_naive` | **0.960** | 0.401 |
| `false_stagnation_bio` / `_naive` | **0.960** | 0.020 |
| `loop_bio` / `_naive` | 0.631 | **0.940** |

**Discrepancy with paper tex**: Paper 4 §4.3 Table 4 reports
`loop accuracy = 0.571` for bio with real embeddings; the authoritative
JSON reports `0.631`. The JSON is the later artifact (commit `339875e`
is a post-paper correction) and is cited as authoritative in
`StagnationGate`'s docstring.

### Honest caveat

The gate uses `EpiplexityMonitor`'s `epiplexic_integral` directly
(sliding-window mean of epiplexity), not the monitor's built-in status
classifier. The classifier's "loop" scenario — which scores 0.631
above — requires distinguishing novelty-low + perplexity-high from
novelty-low + perplexity-low. In practice most LangGraph-side stagnation
presents as identical/near-identical outputs, which trips the integral
directly and is what the E2E test
`tests/integration/test_loop_break.py` exercises. The gate's strength
is in convergence/false-stagnation; loop scenarios specifically are
where the paper's biological design underperforms a naive baseline
and where this gate leans on its integral-based detection rather than
the classifier.

### Reproducing the mock-embedding numbers

```bash
cd /path/to/operon-ai
python -m eval.benchmarks.run_benchmarks --benchmark epiplexity \
    --seed 42 --trials 100
```

Produces `eval/results/benchmarks/seed_42.json` with five scenarios
(`loop`, `convergence`, `exploration`, `trophic_withdrawal`,
`false_stagnation`). Mock embeddings are SHA-256 hash-based — the
biological monitor loses to the naive detector in this regime; the
tension between mock and real embeddings is the central finding of
Paper 4 §4.3.

### Not reproducing the real-embedding numbers

The real-embedding sweep that produced the
`multi_model_summary.json` numbers is not currently CLI-reproducible
from `operon-ai` main: `sentence-transformers` is not a declared
dependency and no runner exposes `--embedding-provider`. The numbers in
`operon-langgraph-gates` are quoted from the archived JSON with commit
SHA `339875e` for traceability. If a future release needs to re-run the
sweep, the steps are:

1. `pip install sentence-transformers`
2. Write a runner that loads `all-MiniLM-L6-v2`, wraps it in an
   `EmbeddingProvider`-shaped object, and invokes
   `run_epiplexity_bench(config, rng)` directly.
3. Run with 3 seeds × 100 trials; aggregate per-seed results to match
   the summary JSON schema.

---

## IntegrityGate → Paper 4 §4 Table 3

### Verbatim paper quote

> "The FULL variant detects all four corruption sites (three gene
> drifts plus one checksum failure) injected between organism stages
> and repairs them in a single `CHECKPOINT_RESTORE` operation. The
> certificate (`state_integrity_verified`) holds after repair in all
> repetitions. RAW and GUARDED are completely blind to genome
> corruption... `DNARepair` as a pre/post-flight integrity check is
> the strongest structural guarantee in the evaluated stack."
>
> — Paper 4 §4, "State integrity: clear structural value."

### Numbers cited in gate docstring

From Paper 4 §4, Table 3 (Integrity row):

| Variant | Det% | Rep% |
|---|---|---|
| RAW | 0 | 0 |
| GUARDED | 0 | 0 |
| **FULL** | **100** | **100** |

### Honest caveat

`IntegrityGate` in `operon-langgraph-gates` is a thin reformulation
of the *idea* of pre-execution integrity checks — it does not reuse
`DNARepair.scan()` or the `state_integrity_verified` theorem directly.
It registers its own theorem (`langgraph_state_integrity`) with clean
LangGraph-flavored evidence params (`invariant_results`,
`failed_invariants`) so the public API honors the "no biology in the
user-facing surface" locked decision. The paper's 100/100 numbers
apply to `DNARepair` on `Genome` objects, which is the inspiration
but not the implementation here. Paper 5 §3's compile-preservation
result is what gives this gate its load-bearing claim: a structural
invariant check, defined once, verifies at every node boundary and
replays from a frozen evidence snapshot.

---

## Verification of the citations in code

`StagnationGate` and `IntegrityGate` docstrings quote the exact numbers
above. If a future commit weakens or changes these citations, update
this file in the same commit so the record stays consistent.

A static check that keeps docstrings and this file aligned is an
explicit non-goal for v0.1 (would be lint-theater); the commit policy
is: touch docstring → touch this file → one commit.
