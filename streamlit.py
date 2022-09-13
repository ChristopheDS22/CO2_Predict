# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 22:01:06 2022

@author: Gilles.NGAMENYE
"""

import pandas as pd
import numpy as np
import streamlit as st

# Titre du streamlit
st.title('Projet CO2 Predict')

# Sommaire du Sreamlit
st.sidebar.title('Sommaire')
pages = ['Introduction','Exploration des données', 'Analyse de données', 'Dataviz', 
         'Modélisation : Classification multiple', 'Modélisation : Regréssion']

page = st.sidebar.radio('Aller vers', pages)

if page == pages[0]:
    st.write('## Introduction au projet')
    
    st.markdown('Identifier les véhicules qui émettent le plus de CO2 est important pour identifier les caractéristiques techniques qui jouent un rôle dans la pollution.')
    st.markdown(' Prédire à l’avance cette pollution permet de prévenir dans le cas de l’apparition de nouveaux types de véhicules (nouvelles séries de voitures par exemple)')
    st.markdown('Le projet est effectué à partir du dataset regroupant les émissions de CO2 et polluants des véhicules commercialisées en France en 2013')
    st.markdown('[Source du dataset]( https://www.ecologie.gouv.fr/normes-euros-demissions-polluants-vehicules-lourds-vehicules-propres)')
    
if page == pages[1]:
    st.write('## Exploration des données')
    
    # Emissions de polluants, CO2 et caractéristiques des véhicules
    # commercialisés en France en 2013
    df_2013 = pd.read_csv('data2013.csv' , sep = ';', encoding='unicode_escape')
    
    st.write('### Visualisation du dataset')
    st.dataframe(df_2013.head(10))
    
    st.markdown('La variable cible est CO2 (g/km)')
    #st.markdown('Il y a ', df_2013.duplicated().sum(), 'doublons dans le dataset') # On checke les doublons
    
    st.write('### Etat des lieux des données')
    st.dataframe(df_2013.info())
    st.markdown('Beaucoup de valeurs manquantes pour les variables HC (g/km) et HC+NOX (g/km)')
    
    if st.checkbox('Afficher les valeurs manquantes'):
        st.dataframe(df_2013.isna().sum())    
    
    st.write('Description statistique des données')
    st.dataframe(df_2013.describe())
    st.markdown('Les ordres de grandeurs sont très larges : il sera nécessaire de procéder à une standardisation des valeurs')
    
    # Suppression des doublons
    df_2013 = df_2013.drop_duplicates()
    df_2013.duplicated().sum()

if page == pages[2]:
    st.write()


if page == pages[3]:
    st.write()



