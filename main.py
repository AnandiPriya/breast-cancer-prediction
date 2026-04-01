import pandas as pd
df = pd.read_csv("breast-cancer-wisconsin-data.csv")
print (df.head())
print (df.info())

print(df['diagnosis'].value_counts())

df = df.drop('id', axis=1)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

print(df.head())


X = df.drop('diagnosis', axis=1)  # Split data into X and y (Machine Learning work )
y = df['diagnosis']

print(X.shape)
print(y.shape)


from sklearn.model_selection import train_test_split  #Train/Test Split (continue)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)


# Goal =Train a model to predict cancer (0 = Benign, 1 = Malignant)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10000) # Create model

model.fit(X_train, y_train) 


y_pred = model.predict(X_test) #Prediction

print(y_pred[:10])



from sklearn.metrics import accuracy_score #Accuracy

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


from sklearn.metrics import confusion_matrix #check false positives & false negatives (Confusion Matrix)

cm = confusion_matrix(y_test, y_pred)
print(cm)