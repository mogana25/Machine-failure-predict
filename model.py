import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load the CSV file
df = pd.read_csv("machine.csv")
print("Dataset loaded successfully!")

# Select independent and dependent variables
X = df[["footfall","tempMode","AQ","USS","CS","VOC","RP","IP","Temperature"]]
y = df["fail"]
print("Features and target selected successfully!")

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
print("Dataset split into training and testing sets!")

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("Feature scaling completed!")

# Instantiate the models
classifier = RandomForestClassifier()
logistic_model = LogisticRegression()
svm_model = SVC(probability=True)
decision_tree_model = DecisionTreeClassifier()

# Train the models
print("Training Random Forest...")
classifier.fit(X_train, y_train)
print("Random Forest trained successfully!")

print("Training Logistic Regression...")
logistic_model.fit(X_train, y_train)
print("Logistic Regression trained successfully!")

print("Training SVM...")
svm_model.fit(X_train, y_train)
print("SVM trained successfully!")

print("Training Decision Tree...")
decision_tree_model.fit(X_train, y_train)
print("Decision Tree trained successfully!")

# Save the models as pickle files
try:
    pickle.dump(classifier, open("model.pkl", "wb"))
    print("Random Forest model saved as 'model.pkl'!")
except Exception as e:
    print(f"Failed to save Random Forest model: {e}")

try:
    pickle.dump(logistic_model, open("logistic_model.pkl", "wb"))
    print("Logistic Regression model saved as 'logistic_model.pkl'!")
except Exception as e:
    print(f"Failed to save Logistic Regression model: {e}")

try:
    pickle.dump(svm_model, open("svm_model.pkl", "wb"))
    print("SVM model saved as 'svm_model.pkl'!")
except Exception as e:
    print(f"Failed to save SVM model: {e}")

try:
    pickle.dump(decision_tree_model, open("decision_tree_model.pkl", "wb"))
    print("Decision Tree model saved as 'decision_tree_model.pkl'!")
except Exception as e:
    print(f"Failed to save Decision Tree model: {e}")