"""
Transport de produits chimiques - INFO-H-3000
Modele PLNE resolu avec PuLP
"""

import pulp
import sys

# 1. PARAMETRES


# Ensembles
T = [1, 2, 3, 4, 5]
R = ["ANV", "CHA", "GAN", "BRU", "HAS"]

# Distances Liege -> destination (km)
dist = {"ANV": 105, "CHA": 100, "GAN": 140, "BRU": 100, "HAS": 60}

# Duree aller-retour (h) : 2*d/70 + 1h arret livraison
tau = {}
for r in R:
    tau[r] = 2 * dist[r] / 70 + 1
tau["BASE"] = 2 * 105 / 70 + 1  # trajet base Anvers -> Liege

# Demandes annuelles en acide par destination (tonnes/an)
D_acid = {
    ("ANV", 1): 9000,  ("ANV", 2): 9000,  ("ANV", 3): 9000,  ("ANV", 4): 9000,  ("ANV", 5): 9000,
    ("CHA", 1): 12000, ("CHA", 2): 12000, ("CHA", 3): 12000, ("CHA", 4): 12000, ("CHA", 5): 12000,
    ("GAN", 1): 2000,  ("GAN", 2): 2000,  ("GAN", 3): 2000,  ("GAN", 4): 2000,  ("GAN", 5): 2000,
    ("BRU", 1): 6200,  ("BRU", 2): 6200,  ("BRU", 3): 6200,  ("BRU", 4): 6200,  ("BRU", 5): 6200,
    ("HAS", 1): 350,   ("HAS", 2): 825,   ("HAS", 3): 1300,  ("HAS", 4): 1300,  ("HAS", 5): 1300,
}

# Demande annuelle en base Anvers -> Liege (tonnes/an), constante
D_base = 30000

# Capacites des compartiments (tonnes)
CAP_L = 16.5   # grand compartiment
CAP_S = 5.5    # petit compartiment

# Heures disponibles par camion par an (250 jours * 8 heures)
H = 2000

# Couts
PRIX_T1   = 140000  # prix achat camion Type 1 (EUR)
PRIX_T2   = 200000  # prix achat camion Type 2 (EUR)
ENTRETIEN = 5000    # cout entretien par camion par an (EUR)
ALPHA     = 0.10    # taux d'amortissement annuel
C_FUEL    = 0.50    # cout carburant par km (EUR/km)

# Prix de revente a l'annee t : P / (1 + alpha)^t
# On suppose les camions neufs au debut de l'horizon (t=0)
revente = {}
for t in T:
    revente[1, t] = PRIX_T1 / (1 + ALPHA) ** t
    revente[2, t] = PRIX_T2 / (1 + ALPHA) ** t

# 2. MODELE

model = pulp.LpProblem("Transport_Chimique", pulp.LpMinimize)

# 3. VARIABLES DE DECISION

# Gestion de ensemble de camions : nombre de camions possedes, achetes, vendus
F = {}  # F[k, t] = nb camions type k en ensemble de camions au debut annee t
B = {}  # B[k, t] = nb camions type k achetes en annee t
S = {}  # S[k, t] = nb camions type k vendus en annee t

for k in [1, 2]:
    for t in T:
        F[k, t] = pulp.LpVariable(f"F_{k}_{t}", lowBound=0, cat="Integer")
        B[k, t] = pulp.LpVariable(f"B_{k}_{t}", lowBound=0, cat="Integer")
        S[k, t] = pulp.LpVariable(f"S_{k}_{t}", lowBound=0, cat="Integer")

# Voyages Type 1 (1 compartiment de 16.5t)
n1A = {}  # n1A[r, t] = nb voyages acide vers destination r, annee t
n1B = {}  # n1B[t]    = nb voyages base Anvers -> Liege, annee t

for r in R:
    for t in T:
        n1A[r, t] = pulp.LpVariable(f"n1A_{r}_{t}", lowBound=0, cat="Integer")

for t in T:
    n1B[t] = pulp.LpVariable(f"n1B_{t}", lowBound=0, cat="Integer")

# Voyages Type 2 simples (un seul compartiment utilise)
n2LA = {}  # n2LA[r, t] = voyages grand compartiment acide vers r, annee t
n2SA = {}  # n2SA[r, t] = voyages petit compartiment acide vers r, annee t
n2LB = {}  # n2LB[t]    = voyages grand compartiment base Anvers -> Liege, annee t
n2SB = {}  # n2SB[t]    = voyages petit compartiment base Anvers -> Liege, annee t

for r in R:
    for t in T:
        n2LA[r, t] = pulp.LpVariable(f"n2LA_{r}_{t}", lowBound=0, cat="Integer")
        n2SA[r, t] = pulp.LpVariable(f"n2SA_{r}_{t}", lowBound=0, cat="Integer")

for t in T:
    n2LB[t] = pulp.LpVariable(f"n2LB_{t}", lowBound=0, cat="Integer")
    n2SB[t] = pulp.LpVariable(f"n2SB_{t}", lowBound=0, cat="Integer")

# Voyages Type 2 combines vers Anvers (aller acide + retour base)
# C1 : grand compartiment = acide (16.5t), petit compartiment = base retour (5.5t)
# C2 : petit compartiment = acide (5.5t),  grand compartiment = base retour (16.5t)
n2C1 = {}
n2C2 = {}

for t in T:
    n2C1[t] = pulp.LpVariable(f"n2C1_{t}", lowBound=0, cat="Integer")
    n2C2[t] = pulp.LpVariable(f"n2C2_{t}", lowBound=0, cat="Integer")

# 4. FONCTION OBJECTIF

# (1) Cout d'achat des nouveaux camions
cout_achat = pulp.LpAffineExpression()
for t in T:
    cout_achat += PRIX_T1 * B[1, t]
    cout_achat += PRIX_T2 * B[2, t]

# (2) Cout d'entretien annuel de toute la ensemble de camions
cout_entretien = pulp.LpAffineExpression()
for t in T:
    cout_entretien += ENTRETIEN * F[1, t]
    cout_entretien += ENTRETIEN * F[2, t]

# (3) Revenus de revente (a soustraire du cout total)
revenu_vente = pulp.LpAffineExpression()
for t in T:
    revenu_vente += revente[1, t] * S[1, t]
    revenu_vente += revente[2, t] * S[2, t]

# (4) Cout carburant : 0.50 EUR/km * distance aller-retour * nb voyages
cout_carburant = pulp.LpAffineExpression()

# Voyages acide vers chaque destination
for r in R:
    for t in T:
        cout_carburant += C_FUEL * 2 * dist[r] * n1A[r, t]
        cout_carburant += C_FUEL * 2 * dist[r] * n2LA[r, t]
        cout_carburant += C_FUEL * 2 * dist[r] * n2SA[r, t]

# Voyages base et voyages combines (tous via Anvers, distance = 105 km)
for t in T:
    cout_carburant += C_FUEL * 2 * dist["ANV"] * n1B[t]
    cout_carburant += C_FUEL * 2 * dist["ANV"] * n2LB[t]
    cout_carburant += C_FUEL * 2 * dist["ANV"] * n2SB[t]
    cout_carburant += C_FUEL * 2 * dist["ANV"] * n2C1[t]
    cout_carburant += C_FUEL * 2 * dist["ANV"] * n2C2[t]

# Fonction objectif : minimiser le cout total net sur 5 ans
model += cout_achat + cout_entretien - revenu_vente + cout_carburant, "Cout_Total"

# 5. CONTRAINTES

# C1 : etat initial de la ensemble de camions (donne par l'enonce)
model += F[1, 1] == 4, "Init_T1"
model += F[2, 1] == 6, "Init_T2"

for k in [1, 2]:
    for t in T:

        # C2 : dynamique de ensemble de camions (coherence entre annees consecutives)
        if t < 5:
            model += F[k, t+1] == F[k, t] + B[k, t] - S[k, t], f"Dynamique_{k}_{t}"

        # C3 : on ne peut pas vendre plus de camions qu'on en possede
        model += S[k, t] <= F[k, t], f"Vente_max_{k}_{t}"

for t in T:

    # C4 : satisfaction de la demande en acide pour chaque destination
    for r in R:
        acide_livre = CAP_L * n1A[r, t]
        acide_livre += CAP_L * n2LA[r, t]
        acide_livre += CAP_S * n2SA[r, t]
        if r == "ANV":
            # Les voyages combines contribuent uniquement vers Anvers
            acide_livre += CAP_L * n2C1[t]
            acide_livre += CAP_S * n2C2[t]
        model += acide_livre >= D_acid[r, t], f"Demande_acide_{r}_{t}"

    # C5 : satisfaction de la demande en base (Anvers -> Liege)
    base_livree = CAP_L * n1B[t]
    base_livree += CAP_L * n2LB[t]
    base_livree += CAP_S * n2SB[t]
    base_livree += CAP_S * n2C1[t]  # retour voyage combine C1
    base_livree += CAP_L * n2C2[t]  # retour voyage combine C2
    model += base_livree >= D_base, f"Demande_base_{t}"

    # C6 : capacite horaire disponible pour les camions Type 1
    heures_T1 = tau["BASE"] * n1B[t]
    for r in R:
        heures_T1 += tau[r] * n1A[r, t]
    model += heures_T1 <= H * F[1, t], f"Capacite_T1_{t}"

    # C7 : capacite horaire disponible pour les camions Type 2
    heures_T2 = tau["BASE"] * n2LB[t]
    heures_T2 += tau["BASE"] * n2SB[t]
    heures_T2 += tau["ANV"] * n2C1[t]
    heures_T2 += tau["ANV"] * n2C2[t]
    for r in R:
        heures_T2 += tau[r] * n2LA[r, t]
        heures_T2 += tau[r] * n2SA[r, t]
    model += heures_T2 <= H * F[2, t], f"Capacite_T2_{t}"

# 6. RESOLUTION

sys.stdout = open("resultat.txt", "w", encoding="utf-8")

solver = pulp.PULP_CBC_CMD(msg=0)
model.solve(solver)

print(f"Statut : {pulp.LpStatus[model.status]}")
print(f"Cout total net : {pulp.value(model.objective):,.2f} EUR")

# 7. AFFICHAGE DES RESULTATS

print("\nFlotte (annee : F_T1, F_T2, B_T1, B_T2, S_T1, S_T2)")
for t in T:
    print(f"  t={t} : {int(F[1,t].value())}, {int(F[2,t].value())}, {int(B[1,t].value())}, {int(B[2,t].value())}, {int(S[1,t].value())}, {int(S[2,t].value())}")

print("\nVoyages Type 1 (annee : ANV, CHA, GAN, BRU, HAS, BASE)")
for t in T:
    print(f"  t={t} : {int(n1A['ANV',t].value())}, {int(n1A['CHA',t].value())}, {int(n1A['GAN',t].value())}, {int(n1A['BRU',t].value())}, {int(n1A['HAS',t].value())}, {int(n1B[t].value())}")

print("\nVoyages Type 2 simples (annee : 2LA_ANV, 2SA_ANV, 2LB, 2SB)")
for t in T:
    print(f"  t={t} : {int(n2LA['ANV',t].value())}, {int(n2SA['ANV',t].value())}, {int(n2LB[t].value())}, {int(n2SB[t].value())}")

print("\nVoyages combines Anvers (annee : C1, C2)")
for t in T:
    print(f"  t={t} : {int(n2C1[t].value())}, {int(n2C2[t].value())}")

print(f"\nDecomposition :")
print(f"  Achats    : {pulp.value(cout_achat):,.2f} EUR")
print(f"  Entretien : {pulp.value(cout_entretien):,.2f} EUR")
print(f"  Carburant : {pulp.value(cout_carburant):,.2f} EUR")
print(f"  Reventes  : {pulp.value(revenu_vente):,.2f} EUR")
print(f"  TOTAL NET : {pulp.value(model.objective):,.2f} EUR")

sys.stdout.close()
sys.stdout = sys.__stdout__
print("Resultats ecrits dans resultat.txt")