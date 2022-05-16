import streamlit as st
import pandas as pad 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

def main():
    """ fonction principale de prédiction de conformité des CQ patient """    

    st.title('Prédiction du résultat du contrôle qualité patient')
    st.write("Rentrer les indices de complexité")
    
    post = st.text_input("(dans le même format que l'exemple ci-dessous, avec SAS10	MCSv	LT	LTMCS	AAV	LSV) : ", "0.4633   0.1076  0.0859  0.1338  0.8025  40.4710")
    indices = post

    
    try:
        ## Préparation des données
        indices = indices.split()
        indices_list = []
        for elm in indices:
            indices_list.append(float(elm))

        test = np.array(indices_list)
        indices = test.reshape(1, -1)

        StandardScaler = load('StandardScaler.joblib')
        indices = StandardScaler.transform(indices)

        def machine_learning_classification(indices):
            RFC_model = load('model_rfc_16052022.sav')
            y_pred_prob = RFC_model.predict_proba(indices)
            result = np.where(y_pred_prob[:,0]>0.30,0,1)
            predictions = result[0]
            if predictions == 1:
                CQ_result = "Conforme"
            elif predictions == 0:
                CQ_result = "Non-conforme"
            else:
                CQ_result = "Problème de modélisation, better call ACD"
                
            list_result = [y_pred_prob, result, CQ_result]
            return list_result

        def machine_learning_regression(indices):
            """to develop"""
            return predictions

        def deep_learning_classification(indices):
            """to develop"""
            return predictions

        def deep_learning_regression(indices):
            """to develop"""
            return predictions


        predict_btn = st.button('Prédire')
        if predict_btn:
            pred = None

            ## machine_learning_classification ##
            st.write(str(post))
            st.write(str(test))
            st.write(str(indices))
            st.write('Pour le modèle de machine learning classification (RandomForestClassifier) : \n')
            st.write('Le résultat du CQ est : ' + str(machine_learning_classification(indices)))
        
    except Exception as e:
        st.write("Problème de format des données d'entrée ou de modélisation, better call ACD (57.68)")
        st.write("Message d'erreur : " + str(e))

#####  get the error :

if __name__ == '__main__':
    main()
