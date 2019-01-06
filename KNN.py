import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

dataset = pd.read_csv('.\cmc2.data')
dataset.head()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 9].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scaler = StandardScaler() 
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=18)
classifier.fit(X_train, y_train)

classifier.score(X_test,y_test)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

error = []

# Calculating error for K values between 1 and 60
for i in range(1, 61):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

	#Print the error value
    print(i)
    print(error)

plt.figure(figsize=(12, 6))  
plt.plot(range(1, 61), error, color='green', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate for each K Value')  
plt.xlabel('K Value')  
plt.ylabel('Average Error')
print(error)
plt.show()
