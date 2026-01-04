# ==========================================
# COMPARAȚIE FP-GROWTH vs APRIORI (FAO)
# ==========================================

!pip install mlxtend

import pandas as pd
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, apriori, association_rules

print(">>> Librării încărcate")

# ------------------------------
# 1. Citirea datelor
# ------------------------------
df = pd.read_csv("consumption_user.csv")

# detectare coloană aliment
if "food_name" in df.columns:
    food_col = "food_name"
elif "food_code" in df.columns:
    food_col = "food_code"
else:
    raise ValueError("Nu există food_name sau food_code!")

df_arm = df[["subject_id", "recall_day", food_col]].dropna()

# ------------------------------
# 2. Construirea tranzacțiilor
# ------------------------------
transactions = (
    df_arm
    .groupby(["subject_id", "recall_day"])[food_col]
    .apply(lambda x: list(set(x)))
    .tolist()
)

print(f">>> Tranzacții: {len(transactions)}")

# ------------------------------
# 3. Transformare binară
# ------------------------------
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
basket = pd.DataFrame(te_array, columns=te.columns_)

# PARAMETRI COMUNI
MIN_SUPPORT = 0.01
MIN_CONF = 0.5

# =================================================
# 4. APRIORI
# =================================================
start_apriori = time.time()

freq_apriori = apriori(
    basket,
    min_support=MIN_SUPPORT,
    use_colnames=True
)

rules_apriori = association_rules(
    freq_apriori,
    metric="confidence",
    min_threshold=MIN_CONF
)

rules_apriori = rules_apriori[rules_apriori["lift"] > 1]

time_apriori = time.time() - start_apriori

# =================================================
# 5. FP-GROWTH
# =================================================
start_fp = time.time()

freq_fp = fpgrowth(
    basket,
    min_support=MIN_SUPPORT,
    use_colnames=True
)

rules_fp = association_rules(
    freq_fp,
    metric="confidence",
    min_threshold=MIN_CONF
)

rules_fp = rules_fp[rules_fp["lift"] > 1]

time_fp = time.time() - start_fp

# =================================================
# 6. REZULTATE COMPARATIVE
# =================================================
print("\n========== COMPARAȚIE ==========")
print(f"Apriori - timp execuție: {time_apriori:.2f} secunde")
print(f"FP-Growth - timp execuție: {time_fp:.2f} secunde\n")

print("Apriori:")
print(f"  Itemseturi frecvente: {len(freq_apriori)}")
print(f"  Reguli generate: {len(rules_apriori)}\n")

print("FP-Growth:")
print(f"  Itemseturi frecvente: {len(freq_fp)}")
print(f"  Reguli generate: {len(rules_fp)}")

print("\n>>> COMPARAȚIA A FOST FINALIZATĂ")
