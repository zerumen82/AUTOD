# -*- coding: utf-8 -*-
"""Diagnóstico: qué concepto puntúa 'bald' y por qué producción dio 0.50."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from roop.img_editor.nlp.semantic_analyzer import LightLocalIntentAnalyzer

nlp = LightLocalIntentAnalyzer()

for p in ["the boy is bald", "the boy is bald...", "el chico es calvo",
          "que lleve barba", "ponle melen"]:
    c = nlp.score_concepts(p)
    top = sorted(c.items(), key=lambda kv: kv[1], reverse=True)[:4]
    m = nlp.get_magnitude(p)
    tgt = nlp.detect_target(p)
    print(f"\n{p!r} -> mag={m:.2f} target={tgt}")
    for k, v in top:
        if v > 0:
            print(f"    {k}: {v:.3f}")
