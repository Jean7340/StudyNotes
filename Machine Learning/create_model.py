from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pickle

iris = load_iris()
X = iris['data']
Y = iris['target']


# train a model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X,Y)

# many steps of data cleaning, feature engineering, model evaluation... 

print(knn_model.predict(X[:3,:])) # make predictions for the first 3 observations

i = 90
print('The prediction for observation %d is: %d'%(i,knn_model.predict(X[[i],:])[0])) # make prediction for observation i

with open('knn_model.p', 'wb') as f:
	pickle.dump(knn_model, f)

with open('knn_model.p', 'rb') as f2:
	loaded_model = pickle.load(f2)

print('These are the predictions using the model that was loaded from the file')
print(loaded_model.predict(X[:3,:])) # make predictions for the first 3 observations


print('Done! Created a predictive model and exported it to the file "knn_model.p"')

