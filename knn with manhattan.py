#================================
#IMPORTING IMPORTANT LIBRARIES===
#================================
import sklearn as sk
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#================================
#LOADING AND SPLITTING DATATSET==
#================================
iris=load_breast_cancer()
X,Y=iris.data,iris.target
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)
knn=KNeighborsClassifier(n_neighbors=9,weights='distance',metric='manhattan',algorithm='ball_tree')
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print('accuracy:',accuracy)