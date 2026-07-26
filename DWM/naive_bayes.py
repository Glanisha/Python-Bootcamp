import numpy as np
dataset = [
    ['Rainy', 'Hot', 'High', False, 'No'],
    ['Rainy', 'Hot', 'High', True, 'No'],
    ['Overcast', 'Hot', 'High', False, 'Yes'],
    ['Sunny', 'Mild', 'High', False, 'Yes'],
    ['Sunny', 'Cool', 'Normal', False, 'Yes'],
    ['Sunny', 'Cool', 'Normal', True, 'No'],
    ['Overcast', 'Cool', 'Normal', True, 'Yes'],
    ['Rainy', 'Mild', 'High', False, 'No'],
    ['Rainy', 'Cool', 'Normal', False, 'Yes'],
    ['Sunny', 'Mild', 'Normal', False, 'Yes'],
    ['Rainy', 'Mild', 'Normal', True, 'Yes'],
    ['Overcast', 'Mild', 'High', True, 'Yes'],
    ['Overcast', 'Hot', 'Normal', False, 'Yes'],
    ['Sunny', 'Mild', 'High', True, 'No']
]

yes_data = [r for r in dataset if r[-1] == 'Yes']
no_data  = [r for r in dataset if r[-1] == 'No']
p_yes, p_no = len(yes_data)/len(dataset), len(no_data)/len(dataset)
test = ['Rainy', 'Mild', 'High', True]

def prob(data, i, v): return sum(1 for r in data if r[i] == v) / len(data)

p_yes_given = p_yes * np.prod([prob(yes_data, i, test[i]) for i in range(4)])
p_no_given  = p_no  * np.prod([prob(no_data, i, test[i]) for i in range(4)])
p_total = p_yes_given + p_no_given
p_yes_final, p_no_final = p_yes_given/p_total, p_no_given/p_total

print("P(Play=Yes|X) =", round(p_yes_final, 4))
print("P(Play=No|X)  =", round(p_no_final, 4))
print("\nPredicted Class:", "Yes" if p_yes_final > p_no_final else "No")
