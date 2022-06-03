import streamlit as st
import pandas as pad 
import numpy as np
from joblib import dump, load
from PIL import Image
import sklearn
import tensorflow as tf
from tensorflow import keras

## TO DO #######
## PCA and TSNE representation


def main():
    """ fonction principale de prédiction de conformité des CQ patient """    

    
    
    st.title('Prédiction du résultat du contrôle qualité patient')
    st.write("Rentrer les indices de complexité")
    
    post = st.text_input("(dans le même format que l'exemple ci-dessous, avec SAS10 MCSv    LT  LTMCS   AAV LSV) : ", "0.723   0.069  30.6298  0.0584  0.094  0.7269")
    indices = post
    label = "Sélectionner la localisation tumorale"
    options = ["Générale", "Pelvis", "Sein", "ORL", "Crâne", "Thorax"]
    localisation = st.radio(label, options, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)
    seuil_localisation = {"Générale": 0.25, "Pelvis": 0.4, "Sein": 0.55, "ORL": 0.3, "Crâne": 0.3, "Thorax": 0.3,}
            
    image_ML = Image.open('image_ML_' + str(localisation) +'.png')
    image_DHL = Image.open('image_DHL.png')

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
            ML_model = load('model_ML_' + str(localisation) + '.joblib')
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
            ML_DTC_model = load('model_ML_DTC.joblib')
            y_pred_prob_DTC = ML_DTC_model.predict_proba(indices)
            result_DTC = np.where(y_pred_prob_DTC[:,0]>0.8,0,1)
            predictions_DTC = result_DTC[0]
            df_ML = pad.DataFrame(predictions_DTC, index = ['1'], columns = ['DTC'])
            
            ML_KN_model = load('model_ML_KN.joblib')
            y_pred_prob_KN = ML_KN_model.predict_proba(indices)
            result_KN = np.where(y_pred_prob_KN[:,0]>0.8,0,1)
            predictions_KN = result_KN[0]
            df_ML['KN'] = predictions_KN

            ML_RFC_model = load('model_ML_RFC.sav')
            y_pred_prob_RFC = ML_RFC_model.predict_proba(indices)
            result_RFC = np.where(y_pred_prob_RFC[:,0]>0.25,0,1)
            predictions_RFC = result_RFC[0]
            df_ML['RFC'] = predictions_RFC

            ML_SVC_model = load('model_ML_SVC.joblib')
            y_pred_prob_SVC = ML_SVC_model.predict_proba(indices)
            result_SVC = np.where(y_pred_prob_SVC[:,0]>0.9,0,1)
            predictions_SVC = result_SVC[0]
            df_ML['SVC'] = predictions_SVC    
            
            DHL_model = load('DHL_model.joblib')
            indices_tensor=tf.convert_to_tensor(df_ML)
            y_pred_prob_DHL = DHL_model.predict(indices_tensor)
            result_DHL = np.where(y_pred_prob_DHL>0.62, 1,0)
            

            if result_DHL == 1:
                CQ_result = "Conforme"
            elif result_DHL == 0:
                CQ_result = "Non-conforme"
            else:
                CQ_result = "Problème de modélisation, better call ACD"
            
            return CQ_result



        predict_btn = st.button('Prédire')
        if predict_btn:
            pred = None

            ## machine_learning_classification ##
            st.write('Pour le modèle de Machine Learning : \n')    
            if machine_learning_classification(indices,localisation, seuil_localisation) == "Conforme":
                        st.success('Le résultat est Conforme !')
            elif machine_learning_classification(indices, localisation, seuil_localisation) == "Non-conforme":
                        st.warning('Le résultat est Non-conforme !')                       
            st.write("NB : un résultat non-conforme correspond à une prédiction que le gamma moyen soit significativement au dessus de la moyenne et que le gamma index soit inférieur à 95%")
            
            if localisation == "ORL":
                        st.image(image_ML, caption='ROC curve and confusion matrix for the Machine Learning model (DecisionTreeClassifier)')
            else:
                        st.image(image_ML, caption='ROC curve and confusion matrix for the Machine Learning model (RandomForestClassifier)')
            
            st.write('Pour le modèle de Deep Hybrid Learning : \n') 
            if deep_hybride_learning_classification(indices) == "Conforme":
                        st.success('Le résultat est Conforme !')
            elif deep_hybride_learning_classification(indices) == "Non-conforme":
                        st.warning('Le résultat est Non-conforme !')                       

            st.image(image_DHL, caption='ROC curve and confusion matrix for the Deep Hybrid Learning model (MultiLayerPerceptron)')

            
    except Exception as e:
        st.write("Problème de format des données d'entrée ou de modélisation, better call ACD (57.68)")
        st.write("Message d'erreur : " + str(e))

#####  get the error :

if __name__ == '__main__':
    main()
