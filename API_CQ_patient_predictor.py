import streamlit as st
import pandas as pad 
import numpy as np
from joblib import dump, load
from PIL import Image
            
## TO DO #######
## add deep hybride learning model


def main():
    """ fonction principale de prédiction de conformité des CQ patient """    

    
    
    st.title('Prédiction du résultat du contrôle qualité patient')
    st.write("Rentrer les indices de complexité")
    
    post = st.text_input("(dans le même format que l'exemple ci-dessous, avec SAS10 MCSv    LT  LTMCS   AAV LSV) : ", "0.4633   0.1076  0.0859  0.1338  0.8025  40.4710")
    indices = post
    label = "Sélectionner la localisation tumorale"
    options = ["Générale", "Pelvis", "Sein", "ORL", "Crâne", "Thorax"]
    localisation = st.radio(label, options, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)
    seuil_localisation = {"Générale": 0.25, "Pelvis": 0.4, "Sein": 0.55, "ORL": 0.3, "Crâne": 0.3, "Thorax": 0.3,}
            
    image_ML = Image.open('image_ML_' + str(localisation) +'.png')

    try:
        ## Préparation des données
        indices = indices.split()
        indices_list = []
        for elm in indices:
            indices_list.append(float(elm))

        test = np.array(indices_list)
        indices = test.reshape(1, -1)

            
        StandardScaler = load('StandardScaler_' + str(localisation) + '.joblib')
        indices = StandardScaler.transform(indices)
        
        indices_finale = []
        for elm in indices[0]:
            indices_finale.append(float(elm))
            
        def machine_learning_classification(indices_finale, localisation, seuil_localisation):
            ML_model = load('model_ML_' + str(localisation) + '.sav')
            y_pred_prob = ML_model.predict_proba(indices)
            result = np.where(y_pred_prob[:,0]>seuil_localisation[localisation],0,1)
            predictions = result[0]
            if predictions == 1:
                CQ_result = "Conforme"
            elif predictions == 0:
                CQ_result = "Non-conforme"
            else:
                CQ_result = "Problème de modélisation, better call ACD"
                
            return CQ_result


        def deep_hybride_learning_classification(indices):
            """to develop"""
            return predictions



        predict_btn = st.button('Prédire')
        if predict_btn:
            pred = None

            ## machine_learning_classification ##
            st.write('Pour le modèle de Machine Learning : \n')    
            if machine_learning_classification(indices,localisation, seuil_localisation) == "Conforme":
                        st.success('Le résultat est Conforme !')
            elif machine_learning_classification(indices, localisation, seuil_localisation) == "Non-conforme":
                        st.warning('Le résultat est Non-conforme !')                       
            if localisation == "ORL":
                        st.image(image_ML, caption='ROC curve and confusion matrix for the Machine Learning model (DecisionTreeClassifier)')
            else:
                        st.image(image_ML, caption='ROC curve and confusion matrix for the Machine Learning model (RandomForestClassifier)')

    except Exception as e:
        st.write("Problème de format des données d'entrée ou de modélisation, better call ACD (57.68)")
        st.write("Message d'erreur : " + str(e))

#####  get the error :

if __name__ == '__main__':
    main()
