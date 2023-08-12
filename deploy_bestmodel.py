import numpy as np
import joblib
import streamlit as st

loaded_model = joblib.load('bestmodel.pkl')

def prediction(model, sepal_length,sepal_width,petal_length,petal_width):
    X_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X_new = X_new.reshape(1,-1)
    prediction = model.predict(X_new)
    if prediction == 0:
        return 'This is Iris-setosa','https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
    elif prediction == 1:
        return 'This is Iris-versicolor','https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
    else:
        return 'This is Iris-virginica' , 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'

def main():
    st.title('Iris Prediction')
    st.title('20116351 - Lê Thành Nghĩa')
    sepal_length = st.text_input('Length of sepal')
    sepal_width = st.text_input('Width of sepal')
    petal_length = st.text_input('Length of petal')
    petal_width = st.text_input('Width of Petal')

    prediction_iris = ''
    flower_img = ''
    if st.button('Predict'):
        prediction_iris,flower_img = prediction(loaded_model, sepal_length,sepal_width,petal_length,petal_width)
        st.success(prediction_iris)
        st.image(flower_img, width=300)

if __name__ == '__main__':
    main()



# X_new = np.array([[5.1, 3.5, 1.4, 0.2]])
# X_var = np.array([0.6856935123042505,0.18800402684563763,3.1131794183445156,0.5824143176733784])
# X_new = X_new.reshape(1,-1)
# X_var = X_var.reshape(1,-1)

# prediction = loaded_model.predict(X_new)
# prediction2 = loaded_model.predict(X_var)
# print(prediction)
# print(prediction2)

