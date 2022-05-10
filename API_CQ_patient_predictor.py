import streamlit as st


import pandas as pad 
import numpy as np

## nettoyage
from sklearn.ensemble import RandomForestClassifier

from joblib import dump, load



def main():
    """ fonction principale de prédiction de conformité des CQ patient """    

    st.title('Prédiction du résultat du contrôle qualité patient')

    post = st.text_input('Rentrer les indices de complexité', "0.4633   0.1076  0.0859  0.1338  0.8025  40.4710")
    indices = post

    ## Préparation des données
    indices = indices.split()
    indices_list = []
    for elm in indices:
        indices_list.append(float(elm))

    indices_list
    test = np.array(indices_list)
    indices = test.reshape(1, -1)


    def machine_learning_classification(indices):
        RFC_model = load('modele_rfc.sav')
        y_pred_prob = RFC_model.predict_proba(indices)
        result = np.where(y_pred_prob<0.13,0,1)
        prediction = result[0][0]
        if predictions == 1:
            CQ_result = "conforme"
        else:
            CQ_result = "non-conforme"
        return CQ_result

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
        st.write('Pour le modèle de machine learning classification (RFC) : \n')
        st.write('Le résultat du CQ est : ' + str(machine_learning_classification(indices_list)))


#####  get the error :

if __name__ == '__main__':
    main()