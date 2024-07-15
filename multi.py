import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix

def load_data(file_path):
    data = pd.read_csv(file_path)
    X= data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    return X,y
file_path ='data.csv'
X,y = load_data(file_path)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)

y_pred =dt.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(f'Accuracy:{accuracy:.2f}')
precision = precision_score(y_test,y_pred)
print(f'Precision:{precision:.2f}')
print('Confusion Matrix') 
print(confusion_matrix(y_test,y_pred))