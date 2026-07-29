# OncRAG — A Vectorless-First Agentic RAG System for Oncology QA

Large language models used for clinical question answering are prone to
hallucination — confident, fluent claims that aren't actually grounded in any
source. That's a serious problem in oncology. **OncRAG** is a hybrid
retrieval-augmented generation pipeline built to reduce that risk, and this
repo includes a controlled **three-way ablation study** measuring exactly how
much each architectural component actually helps — not just a "we built RAG"
writeup.

📄 **Full report:** [`report/OncRAG_REPORT.pdf`](report/OncRAG_REPORT.pdf) — abstract, literature review, full methodology, statistical significance testing, error analysis, and limitations.

---

## The headline finding

All three variants were evaluated on an identical, held-out set of **200 oncology questions**:

| Variant | What it is |
|---|---|
| **Baseline (No-RAG)** | Direct generation from Mistral-7B, no retrieval at all |
| **No-LAQA (RAG only)** | BM25 + FAISS hybrid retrieval, fixed fusion weight, no query preprocessing |
| **LAQA (Full System)** | Adds query preprocessing (LAQ), per-query fusion tuning, and an extended faithfulness self-correction loop |

| Metric | Baseline (No-RAG) | No-LAQA (RAG) | **LAQA (Full)** |
|---|---|---|---|
| Faithfulness (0–1) | 0.635 | 0.717 | **0.924** |
| Hallucination Rate | 19.0% | **47.0%** ⚠️ | **6.5%** |
| BERTScore F1 | 0.123 | 0.619 | **0.858** |
| Precision@5 | — | 0.789 | **0.842** |
| S.C.O.P.E.E Weighted (/5) | 4.25 | 3.99 | **4.74** |
| Mean Latency | **19.55 s** | 22.54 s | 23.81 s |

**The interesting part isn't that LAQA wins** (it does, on every quality/safety
metric, p < 0.000001 via paired t-test, n=200). It's that **naive hybrid
retrieval without query-aware preprocessing (No-LAQA) hallucinates *more* than
generating with no retrieval at all** (47% vs. 19%). Retrieval alone doesn't
guarantee grounded output — bolting on BM25+FAISS without query preprocessing
and self-verification can make things *worse*, not better. Query-aware
preprocessing and a self-correction loop are what make retrieval net-positive
in this system. Full breakdown, statistical tests, and the proposed mechanism
are in Section 5 of the report.

---

## Architecture

```
                         ┌─────────────────────────┐
  Offline (indexing)     │   25 oncology PDFs       │
                         └───────────┬─────────────┘
                                     ▼
                         Agentic Chunking (rule-based,
                         topic-tagged, abbreviation-aware)
                                     ▼
                     ┌───────────────┴────────────────┐
                     ▼                                 ▼
                BM25 (lexical)                  FAISS (semantic)
                     └───────────────┬────────────────┘
                                     ▼
                     Reciprocal Rank Fusion + MMR re-ranking

  Online (query time)   Query ──► LAQ preprocessing (abbreviation
                                  expansion, query-type detection,
                                  per-query alpha tuning, entity extraction)
                                     ▼
                              Hybrid retrieval (k=5)
                                     ▼
                     Mistral-7B-Instruct-v0.2 (4-bit NF4)
                                     ▼
                     Faithfulness self-correction loop
                                     ▼
                              Final answer
```

## Tech stack

- **Generation:** Mistral-7B-Instruct-v0.2, 4-bit NF4 quantized (bitsandbytes)
- **Retrieval:** BM25 + FAISS, fused via Reciprocal Rank Fusion, MMR re-ranking (k=5)
- **Query preprocessing:** custom LAQ stage — abbreviation expansion, query-type detection, per-query alpha tuning, entity extraction
- **Evaluation judge:** Llama3-Med42-8B (independent of the generator, to avoid self-enhancement bias)
- **Metrics:** Precision@5 / Recall@5 / MRR / NDCG@5, BLEU / ROUGE / METEOR, BERTScore F1, faithfulness, and a custom 6-dimension **S.C.O.P.E.E** rubric (Safety, Completeness, Originality, Precision, Efficiency, Empathy)

## Repo structure

```
oncrag/
├── README.md
├── report/
│   └── OncRAG_REPORT.pdf        # full write-up
├── notebooks/
│   ├── oncrag_norag_baseline.ipynb
│   ├── oncrag_nolaqa.ipynb
│   └── oncrag_full_laqa.ipynb
├── results/                      # exported metric tables / charts
├── src/                           # standalone scripts, if pulled out of the notebooks
└── requirements.txt
```

## Corpus & benchmark

- **Corpus:** 25 oncology-domain PDF documents (diagnosis, treatment, staging, prognosis)
- **Benchmark:** 200 curated Q&A pairs, tagged across 14 topic categories and 3 difficulty tiers (Simple / Moderate / Complex)
- Both are internal project artifacts — not included in this repo, but the evaluation protocol is fully documented in the report (Section 3.9–3.10)

## Results at a glance

![Overall comparison across variants](results/overall_comparison.png)
![Faithfulness vs hallucination rate](results/faithfulness_hallucination.png)
![Category and difficulty breakdown](results/category_difficulty_breakdown.png)

## How to run

These notebooks were built and run in **Google Colab** (GPU runtime — a T4 or better is enough for 4-bit Mistral-7B).

1. Open the notebook you want in Colab (`Runtime → Change runtime type → GPU`)
2. Run all cells top to bottom — the first cells install dependencies (`pip install -r requirements.txt` equivalent) and download the Mistral-7B weights
3. `oncrag_norag_baseline.ipynb`, `oncrag_nolaqa.ipynb`, and `oncrag_full_laqa.ipynb` are independent — each runs standalone against the same 200-question benchmark
4. Evaluation cells at the end of each notebook reproduce the metrics tables shown in the report / README

> Note: the 25-document source corpus and 200-question benchmark are internal project artifacts and aren't bundled in this repo (see [Corpus & benchmark](#corpus--benchmark)) — running end-to-end requires supplying your own oncology document set in the same format.

## Limitations

- Faithfulness and S.C.O.P.E.E scores are LLM-judged, not clinician-verified
- Category distribution is uneven — several tags have 5 or fewer questions
- All three variants share one generation backbone (Mistral-7B); results may not generalize to other base models
- Full limitations discussion in Section 5.5 of the report

## Author

**Roopa** — B.Tech CS & Data Science · AI/ML Research Internship, NITPY · July 2026
