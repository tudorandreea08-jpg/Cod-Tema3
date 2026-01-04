import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Citirea datelor
df = pd.read_csv(consumption_user.csv)

# Detectare coloană aliment
if food_name in df.columns
    food_col = food_name
elif food_code in df.columns
    food_col = food_code
else
    raise ValueError(Nu există coloana food_name sau food_code)

# Selectare coloane relevante
df_arm = df[[subject_id, recall_day, food_col]].dropna()

# Construire tranzacții
transactions = (
    df_arm
    .groupby([subject_id, recall_day])[food_col]
    .apply(lambda x list(set(x)))
    .tolist()
)

# Transformare binară
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
basket = pd.DataFrame(te_array, columns=te.columns_)

# FP-Growth
frequent_itemsets = fpgrowth(
    basket,
    min_support=0.01,
    use_colnames=True
)

# Reguli de asociere
rules = association_rules(
    frequent_itemsets,
    metric=confidence,
    min_threshold=0.5
)

rules = rules[
    (rules[lift]  1)
][[antecedents, consequents, support, confidence, lift]]

# Salvare rezultate
rules.to_csv(rules_fp_growth_romania.csv, index=False)

print(FP-Growth finalizat.)
