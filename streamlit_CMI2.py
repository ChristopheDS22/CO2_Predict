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




col1, col2 = st.columns([1,4])
with col1:
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSjyFyh-ZmDnq_yXzkVBt6L-c-9gwqxt0vZRw&usqp=CAU',
             width=250)
with col2:
    st.title('Projet CO2 Predict')
    st.markdown('Camille Millon - Gilles Ngamenye - Christophe Seuret')


# Sommaire du Sreamlit
st.sidebar.title('Sommaire')
pages = ['Accueil','Introduction','Exploration et analyse des données', 
         'Modélisation : Regréssion', 'Modélisation : Classification multiple', 'Interprétabilité', 
         'Conclusion']

page = st.sidebar.radio('Aller vers', pages)

#------------------------------------  Page 0 : accueil ----------------------------------------------------



#------------------------------------  Page 1 : introduction ----------------------------------------------------
if page == pages[1]:
    st.write('## Introduction au projet')
    
    st.markdown('**Objectif : Identifier les véhicules qui émettent le plus de CO2 est important pour identifier les caractéristiques techniques qui jouent un rôle dans la pollution**')
    st.markdown(' Prédire à l’avance cette pollution permet de prévenir dans le cas de l’apparition de nouveaux types de véhicules (nouvelles séries de voitures par exemple)')
    st.markdown('Le projet est effectué à partir du dataset regroupant les émissions de CO2 et polluants des véhicules commercialisées en France en 2013')
    st.markdown('[Source du dataset]( https://www.ecologie.gouv.fr/normes-euros-demissions-polluants-vehicules-lourds-vehicules-propres)')
    
    st.write('### Visualisation du dataset')
    st.dataframe(df_2013.head())
    

#------------------------------------  Page 2 : exploration des données ---------------------------------------------

if page == pages[2]:
    st.write('## Exploration et analyse des données')
    
    st.write('### Variables explicatives')
    st.write('Deux types de variables explicatives sont disponibles : 11 qualitatives et 13 quantitatives')
    st.caption('Certaines variables sont redondantes (colorées de la même façon ci-dessous)')
   
    var_num_2013 = df_2013.select_dtypes(exclude = 'object') # On récupère les variables numériques
    var_cat_2013 = df_2013.select_dtypes(include = 'object') # On récupère les variables catégorielles

    tab_num=pd.DataFrame(var_num_2013.columns,columns=['Quantitatives'])
    tab_cat=pd.DataFrame(var_cat_2013.columns,columns=['Qualitatives'])
    
    # table pour présenter les données qualitatives et quantitatives
    table1 = pd.concat([tab_num,tab_cat],axis=1).fillna('')
    
       #on définit des couleurs identiques poru les variables semblables
    def couleur1(val):
        color='white' if val not in ('Modèle UTAC' 'Modèle dossier' 'Désignation commerciale') else 'paleturquoise'
        return 'background-color:%s' % color

     
    # code pour masquer les index
    hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    # Display a static table
    st.table(table1.style.applymap(couleur1))
