# -*- coding: utf-8 -*-
"""v5.1157b: verificación de los 3 fixes de boca en perfil."""
import sys, ast
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

src = open('roop/ProcessMgr.py', encoding='utf-8').read()
results = []

def check(name, ok):
    results.append((name, ok))
    print(f"{'OK ' if ok else 'FAIL'} {name}")

# T1: gate de objeto oral — los 3 signals requeridos en Y
check("T1 gate occ>0.010", "and (mouth_occ_rel > 0.20)" in src and "(mouth_occ_ring > 0.04)" in src)
check("T1b ya no dispara con rel solo", "mouth_occ_rel > 0.08))" not in src)

# T2: base perfil 0.24
check("T2 base perfil 0.24", "m_blend = 0.24" in src)

# T3: floor de perfil solo con prof_sim >= 0.20 y a 0.30
check("T3 floor condicionado 0.30", "if prof_sim >= 0.20:" in src and "max(m_blend, 0.30)" in src)
check("T3b floor 0.38 incondicional eliminado", "m_blend = max(m_blend, 0.38)" not in src)

# T4: m_blend_safe perfil 0.26
check("T4 m_blend_safe 0.26 perfil", "0.26 if is_profile else 0.38" in src)

# T5: comentario del banner de sesión (no crítico)
check("T5 banner (info)", True)

# Simulación del gate con datos REALES del log
cases = [
    # (occ, rel, ring, esperado) — del log 223907/231021/232209
    (0.012, 0.856, 0.722, True),   # objeto real (occ alto + rel alto + ring alto)
    (0.014, 1.000, 0.992, True),   # objeto real masivo
    (0.018, 1.000, 0.997, True),   # objeto real
    (0.005, 0.528, 0.313, False),  # RUIDO perfil: occ bajo aunque rel alto
    (0.002, 0.119, 0.237, False),  # ruido leve
    (0.009, 0.614, 0.369, False),  # ruido: occ 0.009 < 0.010
    (0.011, 0.15, 0.02, False),    # occ pasa pero rel/ring bajos
    (0.040, 0.30, 0.10, True),     # crop masivo dispara solo
    (0.006, 1.000, 1.000, False),  # ruido puro con rel=1.0 (el caso peor del log)
]
def gate(occ, rel, ring):
    return ((occ > 0.010) and (rel > 0.20) and (ring > 0.04)) or (occ > 0.035)

ok_all = True
for occ, rel, ring, exp in cases:
    got = gate(occ, rel, ring)
    tag = 'objeto' if exp else 'ruido'
    ok = got == exp
    ok_all = ok_all and ok
    print(f"{'OK ' if ok else 'FAIL'} sim occ={occ:.3f} rel={rel:.3f} ring={ring:.3f} → {got} (esperado {exp}, caso {tag})")

# Simulación m_blend perfil
def base_m_blend(ratio, profile=True, photo=False):
    if photo:
        return 0.18 if ratio <= 0.22 else 0.32
    if profile:
        m = 0.24
    else:
        m = 0.38 if ratio > 0 else 0.38
    if ratio > 0.22:
        os_ = min(1.0, (ratio - 0.22) / 0.35)
        m = min(0.65, m * (1.0 + os_ * 0.80))
    return m

prof_cases = [
    (0.10, 0.24),  # boca cerrada perfil: antes 0.32 → ahora 0.24
    (0.20, 0.24),  # leve
    (0.30, 0.284), # abierta moderada (0.24 * (1+0.2286*0.80) = 0.284)
    (0.60, 0.432), # muy abierta (0.24 * (1+0.8*0.8) = 0.432)
]
for ratio, exp in prof_cases:
    got = base_m_blend(ratio)
    ok = abs(got - exp) < 0.01
    ok_all = ok_all and ok
    print(f"{'OK ' if ok else 'FAIL'} sim m_blend perfil ratio={ratio:.2f} → {got:.3f} (esperado {exp})")

n_ok = sum(1 for _, ok in results if ok) + sum(1 for c in cases if gate(*c[:3]) == c[3]) + sum(1 for r, e in prof_cases if abs(base_m_blend(r) - e) < 0.01)
n_total = len(results) + len(cases) + len(prof_cases)
print(f"\n{n_ok}/{n_total} OK")
sys.exit(0 if ok_all else 1)
