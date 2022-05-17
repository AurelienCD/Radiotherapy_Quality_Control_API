import streamlit as st
import pandas as pad 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from PIL import Image
from termcolor import colored
            
## TO DO #######
## liste déroulante pour choisir la localisation
## quand on choisit la loc alors les matrices de confusion et les courbes AUC apparaissent en fonction
## Mettre la valeur de l'AUC en fonction de la loc dans la liste déroulante


class color:
   """ Formattage des couleurs des résultats """
   GREEN = '\033[92m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def main():
    """ fonction principale de prédiction de conformité des CQ patient """    

    image_ML = Image.open('image_ML.png')
    
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
        
        indices_finale = []
        for elm in indices[0]:
            indices_finale.append(float(elm))
            
        def machine_learning_classification(indices_finale):
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
                
            list_result = CQ_result
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
            st.write('Pour le modèle de machine learning classification (RandomForestClassifier) : \n')
            st.write('Le résultat du CQ est : ' + colored(machine_learning_classification(indices), "green"))
            results = '<font color=‘red’>machine_learning_classification(indices)</font>'
            st.write(results, unsafe_allow_html=True)) 
            st.image(image_ML, caption='ROC curve and confusion matrice for the RandomForestClassifier')

    except Exception as e:
        st.write("Problème de format des données d'entrée ou de modélisation, better call ACD (57.68)")
        st.write("Message d'erreur : " + str(e))

#####  get the error :

if __name__ == '__main__':
    main()
