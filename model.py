import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
import pickle
from PIL import Image
data = pd.read_csv('iris.csv')
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
Y = data['species']
clf = DTC(max_depth=10)
clf.fit(X, Y)
file = open('model.pkl','wb')
pickle.dump(clf, file)



che = '''
░░░░░░░░░░░░░░░░░░░░░░█████████
░░███████░░░░░░░░░░███▒▒▒▒▒▒▒▒███
░░█▒▒▒▒▒▒█░░░░░░░███▒▒▒▒▒▒▒▒▒▒▒▒▒███
░░░█▒▒▒▒▒▒█░░░░██▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██
░░░░█▒▒▒▒▒█░░░██▒▒▒▒▒██▒▒▒▒▒▒██▒▒▒▒▒███
░░░░░█▒▒▒█░░░█▒▒▒▒▒▒████▒▒▒▒████▒▒▒▒▒▒██
░░░█████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██
░░░█▒▒▒▒▒▒▒▒▒▒▒▒█▒▒▒▒▒▒▒▒▒█▒▒▒▒▒▒▒▒▒▒▒██
░██▒▒▒▒▒▒▒▒▒▒▒▒▒█▒▒▒██▒▒▒▒▒▒▒▒▒▒██▒▒▒▒██
██▒▒▒███████████▒▒▒▒▒██▒▒▒▒▒▒▒▒██▒▒▒▒▒██
█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒█▒▒▒▒▒▒████████▒▒▒▒▒▒▒██
██▒▒▒▒▒▒▒▒▒▒▒▒▒▒█▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██
░█▒▒▒███████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒██
░██▒▒▒▒▒▒▒▒▒▒████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒█
░░████████████░░░█████████████████

'''



Flower = {'setosa':'Iris Setosa','versicolor':'Iris Versicolor','virginica':'Iris Verginica'}

import streamlit as st
select = ['IRIS','NEW PROJECT','ABOUT']

choice = st.sidebar.radio("GO TO",('IRIS','NEW PROJECT','ABOUT'))

if choice == 'IRIS':
    st.title('The FAMOUS IRIS DATASET')
    st.write('This is just a deployment of prediction on IRIS DATASET as **Practice**')
    sl = st.slider('Sepal Length in cm',min_value=0.0,max_value=10.0,step = 0.1,key='sl1')  
    sw = st.slider('Sepal Width in cm',min_value=0.0,max_value=10.0,step = 0.1,key='sw1')  
    pl = st.slider('Petal Length in cm',min_value=0.0,max_value=10.0,step = 0.1,key='pl1')  
    pw = st.slider('Petal Width in cm',min_value=0.0,max_value=10.0,step = 0.1,key='pw1')  
    
    
    click_predict = st.button('Predict')
    if click_predict:
        x = np.array([st.session_state.sl1,st.session_state.sw1,st.session_state.pl1,st.session_state.pw1])
        x = x.reshape(1,-1)
        pred = clf.predict(x)[0]
        st.write(Flower[pred])
        
        if pred =='setosa':
            seto = Image.open('setosa.jpg','r')
            st.image(seto)
        if pred =='versicolor':
            vers = Image.open('versicolor.jpg','r')
            st.image(vers)
        if pred =='virginica':            
            virg = Image.open('virginica.jpg','r')
            st.image(virg)
if choice == 'NEW PROJECT': 
    st.title('*Ventilator Pressure Prediction*')
    TEMP = 'COMING SOON'
    st.write(TEMP)
    
    
if choice == 'ABOUT': 
    F = open('about.txt','r')
    f = F.read()
    F.close()
    st.write(f)
    st.write(che)    
    st.write('                                  --- Ritartha')    
