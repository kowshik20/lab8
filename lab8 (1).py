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


# In[13]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Your data
data = {
    'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'fair', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Convert categorical variables to numerical using Label Encoding
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)

# Separate features and target labels
Tr_X = df_encoded.drop('buys_computer', axis=1)
Tr_y = df_encoded['buys_computer']

# Build Naïve-Bayes (NB) classifier
model = GaussianNB()
model.fit(Tr_X, Tr_y)

# Test the classifier on new data if needed
# For example:
# new_data = pd.DataFrame({'age': ['>40'], 'income': ['medium'], 'student': ['yes'], 'credit_rating': ['fair']})
# new_data_encoded = new_data.apply(le.transform)
# prediction = model.predict(new_data_encoded)
# print(prediction)


# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error



# Create a DataFrame
df=pd.read_excel('Custom_CNN_Features1.xlsx')

# Separate features and target labels
X = df.drop(['Filename', 'Label'], axis=1)
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Naïve-Bayes (NB) regression model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[18]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error



# Create a DataFrame
df=pd.read_excel('modified_dataset.xlsx')

# Separate features and target labels
X = df.drop(['ImageName', 'Label'], axis=1)
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Naïve-Bayes (NB) regression model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[ ]:




