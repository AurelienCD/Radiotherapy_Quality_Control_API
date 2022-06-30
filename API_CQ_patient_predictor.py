import streamlit as st
import pandas as pad 
import numpy as np
from joblib import dump, load
from PIL import Image
import sklearn
import tensorflow as tf
from tensorflow import keras
import time

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
    seuil_localisation = {"Générale": 0.25, "Crâne": 0.3, "Thorax": 0.3,}
    
       
    #if localisation == "Crâne" or localisation == "Thorax":
        #image_ML = Image.open('image_ML_' + str(localisation) +'.png')   

    #if localisation == "ORL" or localisation == "Sein" or localisation == "Pelvis":
        #image_DHL = Image.open('image_DHL_' + str(localisation) +'.png') 


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


        def deep_hybride_learning_classification(indices, localisation, seuil_localisation):

            if localisation == "ORL":
                # Machine Learning
                model_ML_DTC_ORL = load('model_ML_DTC_ORL.joblib')
                y_pred_prob_DTC = model_ML_DTC_ORL.predict_proba(indices)
                df_ML = pad.DataFrame(y_pred_prob_DTC[:,0], index = ['1'], columns = ['DTC'])                
                model_ML_LDA_ORL = load('model_ML_LDA_ORL.joblib')
                y_pred_prob_LDA = model_ML_LDA_ORL.predict_proba(indices)
                df_ML['LDA'] = y_pred_prob_LDA[:,0]                
                model_ML_RFC_ORL = load('model_ML_RFC_ORL.joblib')
                y_pred_prob_RFC = model_ML_RFC_ORL.predict_proba(indices)
                df_ML['RFC'] = y_pred_prob_RFC[:,0]

                # Deep Learning
                DHL_model_ORL = load('DHL_model_ORL.joblib')
                proba_tensor=tf.convert_to_tensor(df_ML)
                y_pred_prob_DHL = DHL_model_ORL.predict(proba_tensor)
                result_DHL = np.where(y_pred_prob_DHL[:,1]>0.82, 1,0)


            if localisation == "Sein":           
                # Machine Learning
                model_ML_LDA_Sein = load('model_ML_LDA_Sein.joblib')
                y_pred_prob_LDA = model_ML_LDA_Sein.predict_proba(indices) 
                df_ML = pad.DataFrame(y_pred_prob_LDA[:,0], index = ['1'], columns = ['LDA'])                               
                model_ML_RFC_Sein = load('model_ML_RFC_Sein.joblib')
                y_pred_prob_RFC = model_ML_RFC_Sein.predict_proba(indices)                
                df_ML['RFC'] = y_pred_prob_RFC[:,0]
                model_ML_SVC_Sein = load('model_ML_SVC_Sein.joblib')
                y_pred_prob_SVC = model_ML_SVC_Sein.predict_proba(indices)
                df_ML['SVC'] = y_pred_prob_SVC[:,0] 
                
                # Deep Learning
                DHL_model_Sein = load('DHL_model_Sein.joblib')
                proba_tensor=tf.convert_to_tensor(df_ML)
                y_pred_prob_DHL = DHL_model_Sein.predict(proba_tensor)
                result_DHL = np.where(y_pred_prob_DHL[:,1]>0.736, 1,0)


            if localisation == "Pelvis":
                # Machine Learning
                model_ML_RFC_Pelvis = load('model_ML_RFC_Pelvis.joblib')
                y_pred_prob_RFC = model_ML_RFC_Pelvis.predict_proba(indices) 
                df_ML = pad.DataFrame(y_pred_prob_RFC[:,0], index = ['1'], columns = ['RFC'])
                model_ML_KN_Pelvis = load('model_ML_KN_Pelvis.joblib')
                y_pred_prob_KN = model_ML_KN_Pelvis.predict_proba(indices)
                df_ML['KN'] = y_pred_prob_KN[:,0] 
                model_ML_SVC_Pelvis = load('model_ML_SVC_Pelvis.joblib')
                y_pred_prob_SVC = model_ML_SVC_Pelvis.predict_proba(indices) 
                df_ML['SVC'] = y_pred_prob_SVC[:,0]
                model_ML_DTC_Pelvis = load('model_ML_DTC_Pelvis.joblib')
                y_pred_prob_DTC = model_ML_DTC_Pelvis.predict_proba(indices) 
                df_ML['DTC'] = y_pred_prob_DTC[:,0]

                # Deep Learning
                DHL_model_Pelvis = load('DHL_model_Pelvis.joblib')
                proba_tensor=tf.convert_to_tensor(df_ML)
                y_pred_prob_DHL = DHL_model_Pelvis.predict(proba_tensor)
                result_DHL = np.where(y_pred_prob_DHL[:,1]>0.85, 1,0)
                

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
            st.empty()    
            my_bar = st.progress(0)
            for percent_complete in range(100):
                 time.sleep(0.013)
                 my_bar.progress(percent_complete + 1)



            if localisation == "Crâne" or localisation == "Thorax":
               
                ## machine_learning_classification ##
                st.write('Pour le modèle de Machine Learning : \n')    
                if machine_learning_classification(indices,localisation, seuil_localisation) == "Conforme":
                            st.success('Le résultat est Conforme !')
                elif machine_learning_classification(indices, localisation, seuil_localisation) == "Non-conforme":
                            st.warning('Le résultat est Non-conforme !')                       
                st.write("NB : un résultat non-conforme correspond à une prédiction que le gamma moyen soit significativement au dessus de la moyenne et que le gamma index soit inférieur à 95%")              
                #st.image(image_ML, caption='ROC curve and confusion matrix for the Machine Learning model (RandomForestClassifier)')
 


            if localisation == "ORL" or localisation == "Sein" or localisation == "Pelvis":
                
                ## deep_hybrid_learning_classification ##
                st.write('Pour le modèle de Deep Hybrid Learning : \n') 
                if deep_hybride_learning_classification(indices,localisation, seuil_localisation) == "Conforme":
                            st.success('Le résultat est Conforme !')
                elif deep_hybride_learning_classification(indices,localisation, seuil_localisation) == "Non-conforme":
                            st.warning('Le résultat est Non-conforme !')                       
                st.write("NB : un résultat non-conforme correspond à une prédiction que le gamma moyen soit significativement au dessus de la moyenne et que le gamma index soit inférieur à 95%")                       
                #st.image(image_DHL, caption='ROC curve and confusion matrix for the Deep Hybrid Learning model (Machine Learning models and then a MultiLayerPerceptron)')

            
    except Exception as e:
        st.write("Problème de format des données d'entrée ou de modélisation, better call ACD (57.68)")
        st.write("Message d'erreur : " + str(e))

#####  get the error :

if __name__ == '__main__':
    main()
