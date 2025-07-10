import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

st.title('Our first streamlit app')
st.write('This is a simple app that illustrates how to visualize data and models using streamlit')

# load data (from scikit learn)
iris = load_iris()
X = iris['data']
Y = iris['target']
features = iris['feature_names']
df = pd.DataFrame(X, columns=features)
df['Class'] = Y

st.header('Present a dataframe')
st.dataframe(df) # visualize dataframes

st.header('Present a plot')
fig, ax = plt.subplots()
ax.boxplot(np.array(df))
st.pyplot(fig) # visualize a figure


with open('knn_model.p', 'rb') as f2:
    loaded_model = pickle.load(f2)

st.header('Present a prediction')
option = st.selectbox('Select an observation', range(len(df)))
x_selected = X[option,:]
prediction = loaded_model.predict([x_selected])[0]
st.text('You selected observation %d %s and the prediction is %d'%(option,str(x_selected),prediction))


x1 = st.slider('Feature 1', 0.0, 10.0, 5.0, 0.1)
x2 = st.slider('Feature 2', 0.0, 10.0, 5.0, 0.1)
x3 = st.slider('Feature 3', 0.0, 10.0, 5.0, 0.1)
x4 = st.slider('Feature 4', 0.0, 10.0, 5.0, 0.1)
prediction = loaded_model.predict([[x1,x2,x3,x4]])[0]
st.text('The prediction for observation %s is %d'%(str([x1,x2,x3,x4]),prediction))
