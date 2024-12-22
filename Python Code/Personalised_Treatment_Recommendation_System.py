
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Libraries for Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, f1_score, precision_score, recall_score
import pickle
# Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# ignore warning
import warnings
warnings.filterwarnings('ignore')

"""# **2. Load Datasets**"""

df = pd.read_csv('Training.csv')
sym_des = pd.read_csv('symtoms_df.csv')
precautions = pd.read_csv('precautions_df.csv')
workout = pd.read_csv('workout_df.csv')
description = pd.read_csv('description.csv')
medications = pd.read_csv('medications.csv')
diets = pd.read_csv('diets.csv')
symptom_severity = pd.read_csv('Symptom-severity.csv')



"""# **3. Analysing Data**"""

df.head()

df.tail()

df.describe()

df.shape

df.columns

df.info()

sym_des.head()

sym_des.info()

precautions.head()

precautions.info()

workout.head()

workout.info()

description.head()

description.info()

medications.head()

medications.info()

diets.head()

diets.info()

symptom_severity.head()

symptom_severity.info()

df['prognosis'].unique()

len(df.prognosis.unique())

names = df.prognosis.unique()
names

value_counts = df.prognosis.value_counts()
value_counts

"""# **4. Pre-processing**"""

### Transform Object Columns into Numbers
label = LabelEncoder()
df.prognosis = label.fit_transform(df.prognosis)
df.head()

### Split
x = df.drop('prognosis',axis=1)
y = df['prognosis']
keys = x.columns
x.head()

y.head()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=41, shuffle =True,stratify=y)
print('X_train shape is ' , x_train.shape)
print('X_test shape is ' , x_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)

"""# **5. Machine Learning Models**

> **LOGISTIC REGRESSION**
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Initialize the Logistic Regression model
lr_model = LogisticRegression()

# Train the model on the training set
lr_model.fit(x_train, y_train)

# Make predictions on the training set
y_train_pred_lr = lr_model.predict(x_train)
# Make predictions on the testing set
y_test_pred_lr = lr_model.predict(x_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, y_train_pred_lr))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred_lr))

# Display confusion matrix and classification report for testing set
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_lr))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_lr))

"""

> **RANDOM FOREST CLASSIFIER**

"""

from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier()

# Train the model on the training set
rf_model.fit(x_train, y_train)

# Make predictions on the training set
y_train_pred_rf = rf_model.predict(x_train)
# Make predictions on the testing set
y_test_pred_rf = rf_model.predict(x_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, y_train_pred_rf))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred_rf))

# Display confusion matrix and classification report for testing set
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_rf))

"""

> **SVC**
"""

from sklearn.svm import SVC

# Initialize the Support Vector Classifier model
svc_model = SVC()

# Train the model on the training set
svc_model.fit(x_train, y_train)

# Make predictions on the training set
y_train_pred_svc = svc_model.predict(x_train)
# Make predictions on the testing set
y_test_pred_svc = svc_model.predict(x_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, y_train_pred_svc))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred_svc))

# Display confusion matrix and classification report for testing set
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_svc))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_svc))

"""

> **NAIVE BAYES**

"""

from sklearn.naive_bayes import GaussianNB

# Initialize the Gaussian Naive Bayes model
nb_model = GaussianNB()

# Train the model on the training set
nb_model.fit(x_train, y_train)

# Make predictions on the training set
y_train_pred_nb = nb_model.predict(x_train)
# Make predictions on the testing set
y_test_pred_nb = nb_model.predict(x_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, y_train_pred_nb))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred_nb))

# Display confusion matrix and classification report for testing set
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_nb))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_nb))

"""> **Gradient Boosting Classifier**"""

from sklearn.ensemble import GradientBoostingClassifier

# Initialize the Gradient Boosting Classifier model
gb_model = GradientBoostingClassifier()

# Train the model on the training set
gb_model.fit(x_train, y_train)

# Make predictions on the training set
y_train_pred_gb = gb_model.predict(x_train)
# Make predictions on the testing set
y_test_pred_gb = gb_model.predict(x_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, y_train_pred_gb))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred_gb))

# Display confusion matrix and classification report for testing set
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_gb))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_gb))

"""# **6. Model Evaluation**"""

models = {
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'LGBM': LGBMClassifier(verbose=-1),
    'DecisionTree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVC': SVC(),
    'LogisticRegression': LogisticRegression(),
    'Naive Bayes': nb_model

}
### Validation Score
cv_results = {}
for model_name, model in models.items():
    print(f"Cross-validating {model_name}...")
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    cv_results[model_name] = cv_scores.mean()
    print(f"Mean accuracy for {model_name}: {cv_scores.mean():.4f}\n")

cv_results_df = pd.DataFrame(list(cv_results.items()), columns=['Model', 'Mean Accuracy'])
cv_results_df = cv_results_df.sort_values(by='Mean Accuracy', ascending=False)
cv_results_df

best_model_name = cv_results_df.iloc[0]['Model']
print(f"\nBest model from CV: {best_model_name}")

plt.style.use('dark_background')
plt.figure(figsize=(10, 6))
sns.barplot(data=cv_results_df, x='Model', y='Mean Accuracy', palette="Set1")
plt.title('Cross-Validation Accuracy for Different Models', fontsize=16, color='white')
plt.xlabel('Model', color='white')
plt.ylabel('Mean Accuracy', color='white')
for index, row in cv_results_df.iterrows():
    plt.text(index, row['Mean Accuracy'] - 0.02, f'{row["Mean Accuracy"]:.4f}',
             color='white', ha="center", va="center", fontsize=12)
plt.tight_layout()
plt.show()

### Evaluate the model
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred,average='micro')
train_recall = recall_score(y_train, y_train_pred,average='micro')
train_f1 = f1_score(y_train, y_train_pred,average='micro')
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred,average='micro')
test_recall = recall_score(y_test, y_test_pred,average='micro')
test_f1 = f1_score(y_test, y_test_pred,average='micro')
print(f'Training Accuracy: {train_accuracy:.2f}')
print(f'Training Precision: {train_precision:.2f}')
print(f'Training Recall: {train_recall:.2f}')
print(f'Training F1-Score: {train_f1:.2f}')
print('---')
print(f'Test Accuracy: {test_accuracy:.2f}')
print(f'Test Precision: {test_precision:.2f}')
print(f'Test Recall: {test_recall:.2f}')
print(f'Test F1-Score: {test_f1:.2f}')

### Check model
CM = confusion_matrix(y_test, y_test_pred)
print('Confusion Matrix is : \n', CM)
plt.figure(figsize=(20,10))
sns.heatmap(data=CM, annot=True,fmt='g', cmap="Blues", xticklabels=names, yticklabels=names)
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.title(f'Confusion Matrix for {best_model_name} Algorithm')
plt.show()

ClassificationReport = classification_report(y_test,y_test_pred,target_names=names)
print('Classification Report is : ', ClassificationReport )

pickle.dump(model,open('random.pkl','wb'))

"""# **7. Recommendation System**"""

symptoms_dict = {}
diseases_list = {}
for name in names:
    diseases_list[label.transform([name])[0]] = name
with open('diseases_list.pkl', 'wb') as f:
    pickle.dump(diseases_list, f)
for i,name in enumerate(keys):
    symptoms_dict[name] = i
with open('symptoms_dict.pkl', 'wb') as f:
    pickle.dump(symptoms_dict, f)
print('******************* Symptoms_dict *****************\n\n',symptoms_dict)
print('\n\n******************* Diseases_list *****************\n\n',diseases_list)

def helper(dis):
    desc = description[description['Disease'] == predicted_disease]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']


    return desc,pre,med,die,wrkout

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[model.predict([input_vector])[0]]

symptoms = 'inflammatory_nails, blister, red_sore_around_nose, yellow_crust_ooze'
user_symptoms = [s.strip() for s in symptoms.split(',')]

predicted_disease = get_predicted_value(user_symptoms)

desc, pre, med, die, wrkout = helper(predicted_disease)

print("================= Predicted Disease ============\n")
print(predicted_disease)
print("\n================= Description ==================\n")
print(desc)
print("\n================= Precautions ==================\n")
i = 1
for p_i in pre[0]:
    print(i, ": ", p_i)
    i += 1

print("\n================= Medications ==================\n")
i = 1
for m_i in med:
    print(i, ": ", m_i)
    i += 1

print("\n================= Workout ==================\n")
i = 1
for w_i in wrkout:
    print(i, ": ", w_i)
    i += 1

print("\n================= Diets ==================\n")
i = 1
for d_i in die:
    print(i, ": ", d_i)
    i += 1

symptoms = 'redness_of_eyes, sinus_pressure, runny_nose, congestion'
user_symptoms = [s.strip() for s in symptoms.split(',')]

predicted_disease = get_predicted_value(user_symptoms)

desc, pre, med, die, wrkout = helper(predicted_disease)

print("================= Predicted Disease ============\n")
print(predicted_disease)
print("\n================= Description ==================\n")
print(desc)
print("\n================= Precautions ==================\n")
i = 1
for p_i in pre[0]:
    print(i, ": ", p_i)
    i += 1

print("\n================= Medications ==================\n")
i = 1
for m_i in med:
    print(i, ": ", m_i)
    i += 1

print("\n================= Workout ==================\n")
i = 1
for w_i in wrkout:
    print(i, ": ", w_i)
    i += 1

print("\n================= Diets ==================\n")
i = 1
for d_i in die:
    print(i, ": ", d_i)
    i += 1

symptoms = 'shivering, acidity, vomiting'
user_symptoms = [s.strip() for s in symptoms.split(',')]

predicted_disease = get_predicted_value(user_symptoms)

desc, pre, med, die, wrkout = helper(predicted_disease)

print("================= Predicted Disease ============\n")
print(predicted_disease)
print("\n================= Description ==================\n")
print(desc)
print("\n================= Precautions ==================\n")
i = 1
for p_i in pre[0]:
    print(i, ": ", p_i)
    i += 1

print("\n================= Medications ==================\n")
i = 1
for m_i in med:
    print(i, ": ", m_i)
    i += 1

print("\n================= Workout ==================\n")
i = 1
for w_i in wrkout:
    print(i, ": ", w_i)
    i += 1

print("\n================= Diets ==================\n")
i = 1
for d_i in die:
    print(i, ": ", d_i)
    i += 1

symptoms = 'congestion , chest_pain , breathlessness'
user_symptoms = [s.strip() for s in symptoms.split(',')]

predicted_disease = get_predicted_value(user_symptoms)

desc, pre, med, die, wrkout = helper(predicted_disease)

print("================= Predicted Disease ============\n")
print(predicted_disease)
print("\n================= Description ==================\n")
print(desc)
print("\n================= Precautions ==================\n")
i = 1
for p_i in pre[0]:
    print(i, ": ", p_i)
    i += 1

print("\n================= Medications ==================\n")
i = 1
for m_i in med:
    print(i, ": ", m_i)
    i += 1

print("\n================= Workout ==================\n")
i = 1
for w_i in wrkout:
    print(i, ": ", w_i)
    i += 1

print("\n================= Diets ==================\n")
i = 1
for d_i in die:
    print(i, ": ", d_i)
    i += 1
