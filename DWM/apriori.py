import numpy as np
import pandas as pd
from itertools import combinations

transactions = [
    ['Milk'],
    ['Bread', 'Butter', 'Eggs'],
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Eggs'],
    ['Bread', 'Butter']
]

min_support=0.6
min_confidence=0.8 
number_of_transactions =len(transactions)

def get_support(itemset):
    count =0
    for transaction in transactions:
        is_there=True
        for item in itemset:
            if item not in transaction:
                is_there=False
                break
        if is_there:
            count=count+1
    support = count / number_of_transactions
    return support 

def apriori():
    itemsets = []
    items = sorted(set(i for t in transactions for i in t))
    L1 = []
    for item in items:
        support = get_support([item])
        if support >= min_support:
            L1.append(([item], support))
    current_L = L1
    k = 2
    while current_L:
        itemsets.extend(current_L)
        candidates = []
        prev_items = [set(i[0]) for i in current_L]
        for i in range(len(prev_items)):
            for j in range(i+1, len(prev_items)):
                union_set = prev_items[i] | prev_items[j]
                if len(union_set) == k and union_set not in candidates:
                    candidates.append(union_set)
        next_L = []
        for cand in candidates:
            support = get_support(cand)
            if support >= min_support:
                next_L.append((list(cand), support))
        current_L = next_L
        k += 1
    return itemsets

# Generate association rules
def generate_rules(frequent_itemsets):
    rules = []
    for itemset, support in frequent_itemsets:
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                consequent = tuple(set(itemset) - set(antecedent))
                sup_itemset = get_support(itemset)
                sup_antecedent = get_support(antecedent)
                confidence = sup_itemset / sup_antecedent
                if confidence >= min_confidence:
                    rules.append({
                        'Rule': f"{set(antecedent)} → {set(consequent)}",
                        'Support': round(sup_itemset, 2),
                        'Confidence': round(confidence, 2)
                    })
    return rules

# Run Apriori
frequent_itemsets = apriori()
rules = generate_rules(frequent_itemsets)

print("Frequent Itemsets:")
for itemset, support in frequent_itemsets:
    print(f"{itemset} -> Support: {round(support, 2)}")

print("\nStrong Association Rules:")
for rule in rules:
    print(f"{rule['Rule']} | Support: {rule['Support']} | Confidence: {rule['Confidence']}")