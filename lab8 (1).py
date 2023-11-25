#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Define the data as a list of dictionaries
data = [
    {'age': '<=30', 'income': 'high', 'student': 'no', 'credit_rating': 'fair', 'buys_computer': 'no'},
    {'age': '<=30', 'income': 'high', 'student': 'no', 'credit_rating': 'excellent', 'buys_computer': 'no'},
    {'age': '31...40', 'income': 'high', 'student': 'no', 'credit_rating': 'fair', 'buys_computer': 'yes'},
    {'age': '>40', 'income': 'medium', 'student': 'no', 'credit_rating': 'fair', 'buys_computer': 'yes'},
    {'age': '>40', 'income': 'low', 'student': 'yes', 'credit_rating': 'fair', 'buys_computer': 'yes'},
    {'age': '>40', 'income': 'low', 'student': 'yes', 'credit_rating': 'excellent', 'buys_computer': 'no'},
    {'age': '31...40', 'income': 'low', 'student': 'yes', 'credit_rating': 'excellent', 'buys_computer': 'yes'},
    {'age': '<=30', 'income': 'medium', 'student': 'no', 'credit_rating': 'fair', 'buys_computer': 'no'},
    {'age': '<=30', 'income': 'low', 'student': 'yes', 'credit_rating': 'fair', 'buys_computer': 'yes'},
    {'age': '>40', 'income': 'medium', 'student': 'yes', 'credit_rating': 'fair', 'buys_computer': 'yes'},
    {'age': '<=30', 'income': 'medium', 'student': 'yes', 'credit_rating': 'excellent', 'buys_computer': 'yes'},
    {'age': '31...40', 'income': 'medium', 'student': 'no', 'credit_rating': 'excellent', 'buys_computer': 'yes'},
    {'age': '31...40', 'income': 'high', 'student': 'yes', 'credit_rating': 'fair', 'buys_computer':'yes'},
    {'age': '>40','income':'medium','student':'no','credit_rating':'excellent','buys_computer':'no'}
]

# Count the number of instances for each class
count_no = 0
count_yes = 0
for instance in data:
    if instance['buys_computer'] == "no":
        count_no += 1
    else:
        count_yes += 1

# Calculate the prior probability for each class
prior_no = count_no / len(data)
prior_yes = count_yes / len(data)

print("Prior probability for buys_computer = no:", round(prior_no, 4))
print("Prior probability for buys_computer = yes:", round(prior_yes, 4))


# In[8]:


# Calculate the prior probabilities for each class
total_instances = len(data)
prior_probabilities = {}
for buys_computer in ['yes', 'no']:
    count_class = sum(1 for d in data if d['buys_computer'] == buys_computer)
    prior_probabilities[buys_computer] = count_class / total_instances

# Calculate the class conditional densities for each feature and class
class_conditional_densities = {}
for feature in ['age', 'income', 'student', 'credit_rating']:
    for value in ['<=30', '31...40', '>40', 'high', 'medium', 'low']:
        for buys_computer in ['yes', 'no']:
            # Calculate the class conditional density
            count_feature_and_class = sum(1 for d in data if d[feature] == value and d['buys_computer'] == buys_computer)
            class_conditional_densities[(feature, value, buys_computer)] = (count_feature_and_class / total_instances) * prior_probabilities[buys_computer]

# Print the class conditional densities
for key, value in class_conditional_densities.items():
    print(f"Class Conditional Density: {key} = {value}")


# In[9]:


from scipy.stats import chi2_contingency

# Create a contingency table for the four features
contingency_table = []
for age in ['<=30', '31...40', '>40']:
    row = []
    for income in ['high', 'medium', 'low']:
        cell = []
        for student in ['yes', 'no']:
            count = sum(1 for d in data if d['age'] == age and d['income'] == income and d['student'] == student)
            cell.append(count)
        row.append(cell)
    contingency_table.append(row)

# Perform the chi-squared test of independence
statistic, p_value, dof, expected = chi2_contingency(contingency_table)
alpha = 0.05
print(f"p-value: {p_value}")
if p_value < alpha:
    print("The features are dependent (reject null hypothesis)")
else:
    print("The features are independent (fail to reject null hypothesis)")


# In[ ]:




