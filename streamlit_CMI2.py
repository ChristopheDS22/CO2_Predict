# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 09:55:19 2023

@author: S028171
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st




# Emissions de polluants, CO2 et caractéristiques des véhicules
# commercialisés en France en 2013
df_2013 = pd.read_csv('data_2013.csv' , sep = ';', encoding='unicode_escape')



    #---------------------------------------------------------------------------------------------------------
    #                                                    Streamlit 
    #---------------------------------------------------------------------------------------------------------




col1, col2 = st.columns([1,3])
with col1:
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSjyFyh-ZmDnq_yXzkVBt6L-c-9gwqxt0vZRw&usqp=CAU', width=300)
with col2:
    st.title('Projet CO2 Predict')
    st.markdown('Camille Millon - Gilles Ngamenye - Christophe Seuret')


# Sommaire du Sreamlit
st.sidebar.title('Sommaire')
pages = ['Accueil','Introduction','Exploration et analyse des données', 
         'Modélisation : Regréssion', 'Modélisation : Classification multiple', 'Interprétabilité', 
         'Conclusion']

page = st.sidebar.radio('Aller vers', pages)

#------------------------------------  Page 0 ----------------------------------------------------



#------------------------------------  Page 1 ----------------------------------------------------
if page == pages[1]:
    st.write('## Introduction au projet')
    
    st.write('### Visualisation du dataset')
    st.dataframe(df_2013.head())
    
    st.markdown('Identifier les véhicules qui émettent le plus de CO2 est important pour identifier les caractéristiques techniques qui jouent un rôle dans la pollution.')
    st.markdown(' Prédire à l’avance cette pollution permet de prévenir dans le cas de l’apparition de nouveaux types de véhicules (nouvelles séries de voitures par exemple)')
    st.markdown('Le projet est effectué à partir du dataset regroupant les émissions de CO2 et polluants des véhicules commercialisées en France en 2013')
    st.markdown('[Source du dataset]( https://www.ecologie.gouv.fr/normes-euros-demissions-polluants-vehicules-lourds-vehicules-propres)')
    


#------------------------------------  Page 1 ----------------------------------------------------

if page == pages[1]:
    st.write('## Exploration des données')
    
    
