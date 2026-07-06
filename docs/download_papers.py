#!/usr/bin/env python3
"""Download all 16 thesis reference PDFs from arXiv into this folder.
Cross-platform (Windows/macOS/Linux). Usage:  python download_papers.py
"""
import os, sys, time, urllib.request

PAPERS = {
    "Lewis2020_RAG": "2005.11401",
    "Li2023_HaluEval": "2305.11747",
    "Hu2021_LoRA": "2106.09685",
    "Huang2025_HallucinationSurvey": "2311.05232",
    "Niu2024_RAGTruth": "2401.00396",
    "Tang2024_TofuEval": "2402.13249",
    "Warner2024_ModernBERT": "2412.13663",
    "Friel2024_RAGBench": "2407.11005",
    "Valentin2024_CostEffectiveDetection": "2407.21424",
    "Bao2025_FaithBench": "2410.13210",
    "KovacsRecski2025_LettuceDetect": "2502.17125",
    "Karbasi2025_ImpossibilityDetection": "2504.17004",
    "Tamber2025_FaithJudge": "2505.04847",
    "Janiak2025_IllusionOfProgress": "2508.08285",
    "Dubanowska2025_RepresentationFailOOD": "2509.19372",
    "Chen2026_TriviaPlus_RethinkingEval": "2605.11330",
}

here = os.path.dirname(os.path.abspath(__file__))
ok = fail = 0
for name, arxiv_id in PAPERS.items():
    out = os.path.join(here, f"{name}_{arxiv_id}.pdf")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"skip (exists): {os.path.basename(out)}"); ok += 1; continue
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print(f"downloading {os.path.basename(out)}  <- arXiv:{arxiv_id}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (thesis reference download)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        if data[:4] != b"%PDF":
            raise ValueError("not a PDF")
        with open(out, "wb") as f:
            f.write(data)
        ok += 1
    except Exception as e:
        print(f"  !! failed: {arxiv_id} ({e}) -- open https://arxiv.org/abs/{arxiv_id} manually")
        fail += 1
    time.sleep(3)

print("------------------------------------------")
print(f"done: {ok} downloaded/present, {fail} failed.")
