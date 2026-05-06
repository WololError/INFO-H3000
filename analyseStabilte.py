"""
Analyse de sensibilite simplifiee du modele de transport de produits chimiques.
Variation One-At-A-Time (OAT) sur 3 parametres cles avec un nombre reduit de tests.
"""

import pulp
import sys


def resoudre(alpha, c_fuel, hasselt_t2, time_limit=30):
    """Resout le PLNE pour un jeu de parametres et retourne les indicateurs cles."""

    T = [1, 2, 3, 4, 5]
    R = ["ANV", "CHA", "GAN", "BRU", "HAS"]

    dist = {"ANV": 105, "CHA": 100, "GAN": 140, "BRU": 100, "HAS": 60}

    tau = {}
    for r in R:
        tau[r] = 2 * dist[r] / 70 + 1
    tau["BASE"] = 2 * 105 / 70 + 1

    D_acid = {
        ("ANV", 1): 9000,  ("ANV", 2): 9000,  ("ANV", 3): 9000,  ("ANV", 4): 9000,  ("ANV", 5): 9000,
        ("CHA", 1): 12000, ("CHA", 2): 12000, ("CHA", 3): 12000, ("CHA", 4): 12000, ("CHA", 5): 12000,
        ("GAN", 1): 2000,  ("GAN", 2): 2000,  ("GAN", 3): 2000,  ("GAN", 4): 2000,  ("GAN", 5): 2000,
        ("BRU", 1): 6200,  ("BRU", 2): 6200,  ("BRU", 3): 6200,  ("BRU", 4): 6200,  ("BRU", 5): 6200,
        ("HAS", 1): 350,   ("HAS", 2): hasselt_t2, ("HAS", 3): 1300, ("HAS", 4): 1300, ("HAS", 5): 1300,
    }
    D_base = 30000

    CAP_L = 16.5
    CAP_S = 5.5
    H = 2000

    PRIX_T1   = 140000
    PRIX_T2   = 200000
    ENTRETIEN = 5000

    revente = {}
    for t in T:
        revente[1, t] = PRIX_T1 / (1 + alpha) ** t
        revente[2, t] = PRIX_T2 / (1 + alpha) ** t

    model = pulp.LpProblem("Sensibilite", pulp.LpMinimize)

    F = {}
    B = {}
    S = {}
    for k in [1, 2]:
        for t in T:
            F[k, t] = pulp.LpVariable(f"F_{k}_{t}", lowBound=0, cat="Integer")
            B[k, t] = pulp.LpVariable(f"B_{k}_{t}", lowBound=0, cat="Integer")
            S[k, t] = pulp.LpVariable(f"S_{k}_{t}", lowBound=0, cat="Integer")

    n1A = {}
    n1B = {}
    for r in R:
        for t in T:
            n1A[r, t] = pulp.LpVariable(f"n1A_{r}_{t}", lowBound=0, cat="Integer")
    for t in T:
        n1B[t] = pulp.LpVariable(f"n1B_{t}", lowBound=0, cat="Integer")

    n2LA = {}
    n2SA = {}
    n2LB = {}
    n2SB = {}
    for r in R:
        for t in T:
            n2LA[r, t] = pulp.LpVariable(f"n2LA_{r}_{t}", lowBound=0, cat="Integer")
            n2SA[r, t] = pulp.LpVariable(f"n2SA_{r}_{t}", lowBound=0, cat="Integer")
    for t in T:
        n2LB[t] = pulp.LpVariable(f"n2LB_{t}", lowBound=0, cat="Integer")
        n2SB[t] = pulp.LpVariable(f"n2SB_{t}", lowBound=0, cat="Integer")

    n2C1 = {}
    n2C2 = {}
    for t in T:
        n2C1[t] = pulp.LpVariable(f"n2C1_{t}", lowBound=0, cat="Integer")
        n2C2[t] = pulp.LpVariable(f"n2C2_{t}", lowBound=0, cat="Integer")

    cout_achat = pulp.LpAffineExpression()
    for t in T:
        cout_achat += PRIX_T1 * B[1, t] + PRIX_T2 * B[2, t]

    cout_entretien = pulp.LpAffineExpression()
    for t in T:
        cout_entretien += ENTRETIEN * (F[1, t] + F[2, t])

    revenu_vente = pulp.LpAffineExpression()
    for t in T:
        revenu_vente += revente[1, t] * S[1, t] + revente[2, t] * S[2, t]

    cout_carburant = pulp.LpAffineExpression()
    for r in R:
        for t in T:
            cout_carburant += c_fuel * 2 * dist[r] * (n1A[r, t] + n2LA[r, t] + n2SA[r, t])
    for t in T:
        cout_carburant += c_fuel * 2 * dist["ANV"] * (n1B[t] + n2LB[t] + n2SB[t] + n2C1[t] + n2C2[t])

    model += cout_achat + cout_entretien - revenu_vente + cout_carburant

    model += F[1, 1] == 4
    model += F[2, 1] == 6

    for k in [1, 2]:
        for t in T:
            if t < 5:
                model += F[k, t+1] == F[k, t] + B[k, t] - S[k, t]
            model += S[k, t] <= F[k, t]

    for t in T:
        for r in R:
            acide_livre = CAP_L * n1A[r, t] + CAP_L * n2LA[r, t] + CAP_S * n2SA[r, t]
            if r == "ANV":
                acide_livre += CAP_L * n2C1[t] + CAP_S * n2C2[t]
            model += acide_livre >= D_acid[r, t]

        base_livree = CAP_L * n1B[t] + CAP_L * n2LB[t] + CAP_S * n2SB[t]
        base_livree += CAP_S * n2C1[t] + CAP_L * n2C2[t]
        model += base_livree >= D_base

        heures_T1 = tau["BASE"] * n1B[t]
        for r in R:
            heures_T1 += tau[r] * n1A[r, t]
        model += heures_T1 <= H * F[1, t]

        heures_T2 = tau["BASE"] * (n2LB[t] + n2SB[t])
        heures_T2 += tau["ANV"] * (n2C1[t] + n2C2[t])
        for r in R:
            heures_T2 += tau[r] * (n2LA[r, t] + n2SA[r, t])
        model += heures_T2 <= H * F[2, t]

    # Limite de temps imposee au solveur pour eviter les blocages
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    model.solve(solver)

    flotte_t1 = []
    flotte_t2 = []
    for t in T:
        flotte_t1.append(int(F[1, t].value()))
        flotte_t2.append(int(F[2, t].value()))

    return {
        "cout_total": pulp.value(model.objective),
        "flotte_t1": flotte_t1,
        "flotte_t2": flotte_t2,
    }


def afficher(label, res, ref):
    """Une ligne par scenario : label, cout, ecart vs nominal, composition flotte."""
    cout = res["cout_total"]
    ecart_pct = 100 * (cout - ref["cout_total"]) / ref["cout_total"]
    f1 = "/".join(str(x) for x in res["flotte_t1"])
    f2 = "/".join(str(x) for x in res["flotte_t2"])
    print(f"  {label:<20} | {cout:>12,.0f} EUR | {ecart_pct:+6.1f}% | T1: {f1} | T2: {f2}")


# EXECUTION

sys.stdout = open("sensibilite.txt", "w", encoding="utf-8")

ALPHA_NOM   = 0.10
CFUEL_NOM   = 0.50
HAS_T2_NOM  = 825

ref = resoudre(ALPHA_NOM, CFUEL_NOM, HAS_T2_NOM)

print(f"Nominal : alpha=0.10, c_f=0.50 EUR/km, Hasselt_t2=825 t")
print(f"Cout nominal : {ref['cout_total']:,.0f} EUR")
print(f"Flotte T1 : {ref['flotte_t1']}, Flotte T2 : {ref['flotte_t2']}\n")

afficher("NOMINAL", ref, ref)

print("\n[Variation alpha]")
for alpha_test in [0.05, 0.15]:
    res = resoudre(alpha_test, CFUEL_NOM, HAS_T2_NOM)
    afficher(f"alpha = {alpha_test}", res, ref)

print("\n[Variation c_f]")
for cfuel_test in [0.30, 0.70]:
    res = resoudre(ALPHA_NOM, cfuel_test, HAS_T2_NOM)
    afficher(f"c_f = {cfuel_test} EUR/km", res, ref)

print("\n[Variation Hasselt_t2]")
for has_test in [350, 1300]:
    res = resoudre(ALPHA_NOM, CFUEL_NOM, has_test)
    afficher(f"Hasselt_t2 = {has_test} t", res, ref)

sys.stdout.close()
sys.stdout = sys.__stdout__
print("Resultats ecrits dans sensibilite.txt")