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

# Affichage de toutes les pages de la présentation sur toute la largeur de l'écran automatiquement:
st.set_page_config(layout="wide")


# Sommaire du Sreamlit
## Centrer l'image en haut de la sidebar:
st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:
    #st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSjyFyh-ZmDnq_yXzkVBt6L-c-9gwqxt0vZRw&usqp=CAU')
    st.image('https://www.fiches-auto.fr/sdoms/shiatsu/uploaded/part-effet-de-serre-co2-automobile-2.jpg')
 
## Affichage du titre et du plan dans la sidebar:
st.sidebar.title('Projet CO2 Predict')    
pages = ['Accueil','Introduction','Exploration et analyse des données', 
         'Modélisation : Régression multiple', 'Modélisation : Classification', 'Interprétabilité SHAP multi-classes', 
         "Prévoyez les rejets de CO2 et la classe d'émission de votre véhicule!", 'Conclusion']

st.sidebar.markdown('**Sélectionnez une page:**')
page = st.sidebar.radio('', pages)

## Affichage des auteurs et mentor en bas de la sidebar:
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write(' ')
st.sidebar.write('### Auteurs:')
st.sidebar.write('Camille Millon')
st.sidebar.write('Gilles Ngamenye')
st.sidebar.write('Christophe Seuret')
st.sidebar.write(' ')
st.sidebar.write('### Mentor:')
st.sidebar.write('Dan Cohen')

#------------------------------------  Page 0 : accueil ----------------------------------------------------
if page == pages[0]:
    from PIL import Image
    st.image(Image.open('CO2 predict1.png'))


#------------------------------------  Page 1 : introduction ----------------------------------------------------
if page == pages[1]:
    st.write('### Introduction au projet')
    
    st.write('###### Objectifs :  \n- Identifier les véhicules qui émettent le plus de CO2 est important pour identifier les caractéristiques techniques qui jouent un rôle dans la pollution.  \n- Prédire à l’avance cette pollution permet de prévenir dans le cas de l’apparition de nouveaux types de véhicules (nouvelles séries de voitures par exemple')
    st.markdown('Le projet est effectué à partir du dataset regroupant les émissions de CO2 et polluants des véhicules commercialisées en France en 2013')
    st.markdown('[Source du dataset]( https://www.ecologie.gouv.fr/normes-euros-demissions-polluants-vehicules-lourds-vehicules-propres)')
       
    st.write('### Visualisation du dataset')
    st.dataframe(df_2013.head())
    

#------------------------------------  Page 2 : exploration des données ---------------------------------------------

if page == pages[2]:
    st.write('## Exploration et analyse des données')
    

    tab1, tab2, tab3 = st.tabs(['Variables explicatives', 'Preprocessing', 'Liens entre variables'])
    
    with tab1:
 
        st.write('Deux types de variables explicatives sont disponibles : 11 qualitatives et 13 quantitatives')
        st.write('Le dataset de départ contient 44 850 lignes')
        st.caption('Certaines variables sont redondantes (colorées de la même façon ci-dessous)')

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



    with tab2:
        
        st.write('**Etapes du preprocessing :**')
        st.write('- Suppression des doublons (619)')
        st.write('- Traitement des valeurs manquantes : concerne uniquement les variables quantitatives, remplacement par les moyennes des valeurs non manquantes')
        st.write('- Suppression des modalités sous-représentées:')
        

        fig1=px.histogram(df_2013,x="Carburant",color = 'Carburant',color_discrete_sequence=px.colors.qualitative.Pastel)
        fig1.update_layout(title_text='Variable "Carburant" avant preprocessing', title_x=0.5)
        
        fig2=px.histogram(data,x="Carburant") 
        fig2.update_layout(title_text='Variable "Carburant" après preprocessing', title_x=0.5)
        
        data_container=st.container()
        with data_container:
            commentaires,plot1, plot2 = st.columns(3)
            with commentaires:
                st.write('La variable carburant possède un grand nombre de modalités')
            with plot1:
                st.plotly_chart(fig1, use_container_width=True)
            with plot2:
                st.plotly_chart(fig2, use_container_width=True)
        
        st.write('- Sélection des variables utiles')
        st.write('- Suppression des doublons suite aux premiers traitements')

    with tab3:
        st.write('Variable "carburant"')


#_______________________________________________________________________________________________________
#
#                                   Page 3 : régression 
#_______________________________________________________________________________________________________

#CHARGEMENT DES LIBRAIRIES: ----------------------------------------------------------------------------
import itertools

## Les différents types de modèles de Machine Learning
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel

## Les fonctions de paramétrage de la modélisation
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.model_selection import GridSearchCV

## Les fonctions de preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

## Les fonctions statistiques
import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

## Les métriques
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from scipy.stats import jarque_bera

## Les fonctions de sauvegarde et chargement de modèles
from joblib import dump, load


from sklearn.model_selection import train_test_split

import matplotlib as mpl





# CHARGEMENT DES JEUX DE DONNEES NETTOYES ET DES TARGETS CORRESPONDANTES: ----------------------------------------------------------------------------

data = pd.read_csv('data.csv', index_col = 0)
target_reg = pd.read_csv('target_reg.csv', index_col = 0)
target_reg = target_reg.squeeze()

data_go = pd.read_csv('data_go.csv', index_col = 0)
target_go = pd.read_csv('target_go.csv', index_col = 0)
target_go = target_go.squeeze()

data_es = pd.read_csv('data_es.csv', index_col = 0)
target_es = pd.read_csv('target_es.csv', index_col = 0)
target_es = target_es.squeeze()

df = pd.read_csv('df.csv', index_col = 0)
df.CO2 = target_reg


# CHARGEMENT DES MODELES: ------------------------------------------------------------------------


# FONCTIONS: ----------------------------------------------------------------------------

def standardisation_lr(data, target_reg):
    # Séparation du jeu de données en un jeu d'entrainement et de test:
    X_train, X_test, y_train, y_test = train_test_split(data, target_reg, random_state = 123, test_size = 0.2)
    
    #Standardisation des valeurs numériques + variables 'Marque' (beaucoup de catégories (>10)):
    cols = ['puiss_max', 'masse_ordma_min', 'Marque']
    sc = StandardScaler()
    X_train[cols] = sc.fit_transform(X_train[cols])
    X_test[cols] = sc.transform(X_test[cols])
      
    return [X_train, X_test, y_train, y_test]

def regression_lineaire(model_joblib, X_train, y_train, X_test, y_test):
    # Chargement du modèle de régression linéaire:
    lr = load(model_joblib)
    
    # Entraînement et prédictions:

    pred_train = lr.predict(X_train) # = valeurs ajustées X_train
    pred_test = lr.predict(X_test) # = valeurs ajustées X_test
    
    return [lr, pred_train, pred_test]

def selecteur(X_train, y_train, X_test, y_test):
    # Instanciation d'un modèle de régression linéaire
    lr_sfm = LinearRegression()
    
    # Création d'un sélecteur à partir de lr:
    sfm = SelectFromModel(lr_sfm)
    
    # Entrainement du selecteur et sauvegarde des colonnes de X_train sélectionnées par sfm dans sfm_train:
    sfm_train = pd.DataFrame(sfm.fit_transform(X_train, y_train), index = X_train.index)
    sfm_train = X_train[X_train.columns[sfm.get_support()]]
    
    # Sauvegarde des colonnes de X_test dans sfm_test:
    sfm_test = sfm.transform(X_test)
    sfm_test = X_test[X_test.columns[sfm.get_support()]]
        
    # Régression linéaire avec sfm_train:
    lr_sfm.fit(sfm_train, y_train)
    pred_train = lr_sfm.predict(sfm_train) # = valeurs ajustées sfm_train
    pred_test = lr_sfm.predict(sfm_test) # = valeurs ajustées sfm_test
    
    return [lr_sfm, pred_train, pred_test, sfm_train, sfm_test]

def metrics_sfm(lr_sfm, X_train, y_train, X_test, y_test, pred_train, pred_test, sfm_train, sfm_test):
    # Affichage des metrics:
    residus = pred_train - y_train 
    residus_std = residus/np.sqrt(np.sum(residus**2)/(len(residus)-1))
    x, pval = jarque_bera(residus_std)
    st.write('p_value test de normalité de Jarque-Bera: =', round(pval,2)) 
    st.write("")
    st.write("R² train =", round(lr_sfm.score(sfm_train, y_train),2))
    st.write("R² obtenu par CV =", round(cross_val_score(lr_sfm,sfm_train,y_train).mean(),2))
    st.write("R² test =", round(lr_sfm.score(sfm_test, y_test),2))
    st.write("")
    st.write('RMSE train =', round(np.sqrt(mean_squared_error(y_train, pred_train)),2))
    st.write('RMSE test =', round(np.sqrt(mean_squared_error(y_test, pred_test)),2))
    st.write("")
    st.write("MAE train:", round(mean_absolute_error(y_train, pred_train),2))
    st.write("MAE test:", round(mean_absolute_error(y_test, pred_test),2))


def metrics_lr(lr, X_train, y_train, X_test, y_test, pred_train, pred_test):
    # Affichage des metrics:
    residus = pred_train - y_train 
    residus_std = residus/np.sqrt(np.sum(residus**2)/(len(residus)-1))
    x, pval = jarque_bera(residus_std)
    st.write('p_value test de normalité de Jarque-Bera: =', round(pval,2)) 
    st.write("")
    st.write("R² train =", round(lr.score(X_train, y_train),2))
    st.write("R² obtenu par CV =", round(cross_val_score(lr,X_train,y_train, cv = 5).mean(),2))
    st.write("R² test =", round(lr.score(X_test, y_test),2))
    st.write("")
    st.write('RMSE train =', round(np.sqrt(mean_squared_error(y_train, pred_train)),2))
    st.write('RMSE test =', round(np.sqrt(mean_squared_error(y_test, pred_test)),2))
    st.write("")
    st.write("MAE train:", round(mean_absolute_error(y_train, pred_train),2))
    st.write("MAE test:", round(mean_absolute_error(y_test, pred_test),2))
    
def coef_lr(lr, X_train):
    # Représentation des coefficients:
    plt.rcParams['axes.facecolor'] = 'whitesmoke'
    
    coef = lr.coef_
    fig = plt.figure()
    plt.bar(X_train.columns, coef)
    plt.xticks(X_train.columns, rotation = 90)
    st.pyplot(fig)

def coef_sfm(lr_sfm, sfm_train):
    # Représentation des coefficients:
    
    plt.rcParams['axes.facecolor'] = 'whitesmoke'
    if sfm_train.shape[1] >= 1:
        fig = plt.figure()
        coef = lr_sfm.coef_
        fig = plt.figure()
        plt.bar(sfm_train.columns, coef)
        plt.xticks(sfm_train.columns, rotation = 90)
        st.pyplot(fig)

def graph_res(y_train, y_test, pred_train, pred_test):
    #Normalité des résidus:
    ## Calcul des résidus et résidus normalisés:
    residus = pred_train - y_train 
    residus_norm = (residus-residus.mean())/residus.std()
    residus_std = residus/np.sqrt(np.sum(residus**2)/(len(residus)-1))
          
    # Graphes :
    fig = plt.figure(figsize = (15,10))
    # Espacement des graphes:
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.4)
    
    plt.rcParams['axes.facecolor'] = 'whitesmoke'
    
    ## Graphe normalisation résidus:
    plt.subplot(2,2,1)
    stats.probplot(residus_norm, plot = plt)
    plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
    
    ## Graphe résidus en fonction de pred_train (valeurs ajustées):
    plt.subplot(2,2,2)
    plt.scatter(pred_train, residus, alpha = 0.3)
    plt.plot((pred_train.min(), pred_train.max()), (0, 0), lw=3, color='red')
    plt.plot((pred_train.min(), pred_train.max()), (2*residus.std(), 2*residus.std()), 'r-', lw=1.5, label = '± 2 σ') 
    plt.plot((pred_train.min(), pred_train.max()), (3*residus.std(), 3*residus.std()), 'r--', lw=1.5, label = '± 3 σ')
    plt.plot((pred_train.min(), pred_train.max()), (-2*residus.std(), -2*residus.std()), 'r-',lw=1.5)
    plt.plot((pred_train.min(), pred_train.max()), (-3*residus.std(), -3*residus.std()), 'r--', lw=1.5)
    plt.title('Résidus en fonction de pred_train (valeurs ajustées)')
    plt.xlabel('pred_train (valeurs ajustées)')
    plt.ylabel('Résidus')
    plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
    plt.legend(loc = 'lower left')
    
    ## Graphe boxplot des résidus:
    plt.subplot(2,2,3)
    sns.boxplot(residus, notch=True)
    plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
    plt.title('Boite à moustache des résidus')
    plt.xlabel('résidus')
    
    ## Graphe prédictions en fonction de y_test (= le long de la droite si elles sont bonnes):
    plt.subplot(2,2,4)
    plt.scatter(pred_test, y_test, alpha = 0.3)
    plt.title('Nuage de points entre pred_test et y_test')
    plt.xlabel('pred_test')
    plt.ylabel('y_test')
    plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
    plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), lw = 3, color ='red')
    st.pyplot(fig)
    
    
    return [residus, residus_norm, residus_std]
    
def graph_res_sfm(y_train, y_test, pred_train, pred_test):
    #Normalité des résidus:
    ## Calcul des résidus et résidus normalisés:
    residus = pred_train - y_train 
    residus_norm = (residus-residus.mean())/residus.std()
    residus_std = residus/np.sqrt(np.sum(residus**2)/(len(residus)-1))
    
    # Graphes :
    fig = plt.figure(figsize = (15,10))
    # Espacement des graphes:
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.4)
    
    plt.rcParams['axes.facecolor'] = 'whitesmoke'
    
    ## Graphe normalisation résidus:
    plt.subplot(2,2,1)
    stats.probplot(residus_norm, plot = plt)
    plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
    
    ## Graphe résidus en fonction de pred_train (valeurs ajustées):
    plt.subplot(2,2,2)
    plt.scatter(pred_train, residus, alpha = 0.3)
    plt.plot((pred_train.min(), pred_train.max()), (0, 0), lw=3, color='red')
    plt.plot((pred_train.min(), pred_train.max()), (2*residus.std(), 2*residus.std()), 'r-', lw=1.5, label = '± 2 σ') 
    plt.plot((pred_train.min(), pred_train.max()), (3*residus.std(), 3*residus.std()), 'r--', lw=1.5, label = '± 3 σ')
    plt.plot((pred_train.min(), pred_train.max()), (-2*residus.std(), -2*residus.std()), 'r-',lw=1.5)
    plt.plot((pred_train.min(), pred_train.max()), (-3*residus.std(), -3*residus.std()), 'r--', lw=1.5)
    plt.title('Résidus en fonction de pred_train (valeurs ajustées)')
    plt.xlabel('pred_train (valeurs ajustées)')
    plt.ylabel('Résidus')
    plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
    plt.legend(loc = 'lower left')
    
    ## Graphe boxplot des résidus:
    plt.subplot(2,2,3)
    sns.boxplot(residus, notch=True)
    plt.title('Boite à moustache des résidus')
    plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
    plt.xlabel('résidus')
    
    ## Graphe prédictions en fonction de y_test (= le long de la droite si elles sont bonnes):
    plt.subplot(2,2,4)
    plt.scatter(pred_test, y_test, alpha = 0.3)
    plt.title('Nuage de points entre pred_test et y_test')
    plt.xlabel('pred_test')
    plt.ylabel('y_test')
    plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
    plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), lw = 3, color ='red')
    st.pyplot(fig)
    
    return [residus, residus_norm, residus_std]

# Création d'un DataFrame regroupant les données d'origine de df enrichi des valeurs ajustées,
# des résidus, des distances de cook
def df_res(sfm_train, y_train, pred_train, residus):
    #chargement data:
    #dfdata = pd.read_csv('data.csv', index_col = 0)
    
    #Analyse statsmodel:
    X = sfm_train
    X = sm.add_constant(X) #ajout d'une constante
    y = y_train
    model = sm.OLS(y, X)
    results = model.fit()
    
    # distance de Cook (= identification des points trop influents):
    influence = results.get_influence() # results DOIT être un model de statsmodels
    (c, p) = influence.cooks_distance  # c = distance et p = p-value
    
    # AJOUT DES VARIABLES CALCULEES A DF (pred_train, résidus, résidus normalisés et distance de cook)
    
    #PRED_TRAIN:
    
    ## Création d'un DataFrame stockant pred_train en conservant les index et arrondir à une décimale:
    y_pred = pd.DataFrame(pred_train, index = sfm_train.index)
    y_pred= pd.DataFrame(y_pred.rename(columns ={0:'pred_train'}))
    y_pred = round(y_pred.pred_train,1)
   
    ## Création df1 (= Ajout de pred_train à df):
    df1 = df.join(y_pred)
    
    ## Suppression des Nans:
    df1 = df1.dropna()
    
    #RESIDUS:
    
    ## Création d'un DataFrame stockant les résidus:
    res = pd.DataFrame(residus)
    res.rename(columns ={'CO2':'residus'}, inplace = True)
    
    ## Ajout des résidus à df1:
    df1 = df1.join(res)
    
    # RESIDUS NORMALISES:
    
    ## Création d'un DataFrame stockant les résidus noramlisés:
    res_norm = pd.DataFrame(residus_norm)
    res_norm.rename(columns ={'CO2':'residus_normalisés'}, inplace = True)
    
    ## Ajout des résidus normalisés à df1:
    df1 = df1.join(res_norm)
    
    ## Labelisation des résidus normalisés à 2 écarts-types:
    liste = []
    for residus in df1.residus_normalisés:
        if residus >2 or residus <-2:
            liste.append('res norm ±2 σ')
        else:
            liste.append('ok')
    ## ajout liste (=résidus normalisés labélisés à 2 EC) à df1:
    df1['res_norm_±2_σ'] = liste
    
    ## Labelisation des résidus normalisés à 3 écarts-types:
    liste = []
    for residus in df1.residus_normalisés:
        if residus >3 or residus <-3:
            liste.append('res norm ±3 σ')
        else:
            liste.append('ok')
    ## ajout liste (=résidus normalisés labélisés à 3 EC) à df1:
    df1['res_norm_±3_σ'] = liste
    
    # DISTANCE DE COOK:
    
    ## Création d'un DataFrame stockant les distances de Cook:
    dist_cook = pd.DataFrame(c, index = res_norm.index)
    dist_cook.rename(columns ={0:'dist_cook'}, inplace = True)
    
    ## Ajout des distances de Cook à df1:
    df1 = df1.join(dist_cook)
    
    ## Labelisation des distances de Cook:
    liste = []
    for dist in df1.dist_cook:
        if dist > 4/len(y_train) or dist > 1:
            liste.append('observation influente')
        else:
            liste.append('observation non influente')
    ## ajout liste (=résidus normalisés labélisés à 3 EC) à df1:
    df1['observation_influente'] = liste
    
    # Validation des résidus élevés à 2 et 3 écarts-types:
    st.write('Pourcentage des résidus à ±2 ecarts-types (doit être <0.05) =',round(df1[(df1['residus_normalisés']>2)|(df1['residus_normalisés']<-2)].residus_normalisés.count()/df1.residus_normalisés.count(), 3))
    st.write('Pourcentage des résidus à ±3 ecarts-types (doit être <0.003) =',round(df1[(df1['residus_normalisés']>3)|(df1['residus_normalisés']<-3)].residus_normalisés.count()/df1.residus_normalisés.count(), 3))
    st.write('')

    
    # Affichage des valeurs les plus influentes du modèle:  
    fig = plt.figure(figsize = (15,5))
    ax = fig.add_subplot(111)
    ax.scatter(df1[df1['observation_influente'] == 'observation influente'].pred_train, df1[df1['observation_influente'] == 'observation influente'].residus, color = 'orange', label = 'observation influente')
    ax.scatter(df1[df1['observation_influente'] == 'observation non influente'].pred_train, df1[df1['observation_influente'] == 'observation non influente'].residus, alpha = 0.2, label = 'observation non influente')
    ax.plot((df1.pred_train.min(), df1.pred_train.max()), (0, 0), lw=3, color='red')
    ax.plot((df1.pred_train.min(), df1.pred_train.max()), (2*df1.residus.std(), 2*df1.residus.std()), 'r-', lw=1.5, label = '2 σ') 
    ax.plot((df1.pred_train.min(), df1.pred_train.max()), (3*df1.residus.std(), 3*df1.residus.std()), 'r--', lw=1.5, label = '3 σ')
    ax.plot((df1.pred_train.min(), df1.pred_train.max()), (-2*df1.residus.std(), -2*df1.residus.std()), 'r-',lw=1.5)
    ax.plot((df1.pred_train.min(), df1.pred_train.max()), (-3*df1.residus.std(), -3*df1.residus.std()), 'r--', lw=1.5)
    ax.set(title='Résidus en fonction de pred_train (valeurs ajustées)')
    ax.set(xlabel='pred_train (valeurs ajustées)')
    ax.set(ylabel='Résidus')
    ax.legend()
    st.pyplot(fig) 
    
    st.write('')
    st.write('')
    st.write('')
    st.markdown("###### Analyse des résidus à: 👇")
    choix_EC = st.radio("",
                           ["±2 écarts-types",
                            "±3 écarts-types",
                            "résidus influant trop fortement sur le modèle (distance de Cook)"],
                           key="visibility")
    st.write('')
    st.write('')
    st.write('')

    # Représentation graphique des résidus - de quoi sont composés ces résidus élevés? - qu'est-ce qui les caractérisent?:
    
    if choix_EC != 'résidus influant trop fortement sur le modèle (distance de Cook)':
        
        if choix_EC == '±2 écarts-types':
            EC = 2
            res_boxplot = 'res_norm_±2_σ'
                
        if choix_EC == '±3 écarts-types':
            EC = 3
            res_boxplot = 'res_norm_±3_σ'
        
        st.markdown('###### Répartition des résidus élevés selon les variables catégorielles')
        
        # Représentation graphique des résidus - de quoi sont composés ces résidus élevés?:
        fig = plt.figure(figsize = (16,8))
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0,
                            hspace=0.1)
        # Graphe Marque:
        plt.subplot(221)
        plt.pie(df1.Marque[(df1['residus_normalisés']>EC)|(df1['residus_normalisés']<-EC)].value_counts(),
                        labels = df1.Marque[(df1['residus_normalisés']>EC)|(df1['residus_normalisés']<-EC)].value_counts().index,
                       labeldistance=1.2,
                       pctdistance = 0.8,
                       autopct = lambda x: str(round(x,2))+'%',
                       shadow =True)
            
        # Graphe Carburant:
        plt.subplot(222)
        plt.pie(df1.Carrosserie[(df1['residus_normalisés']>EC)|(df1['residus_normalisés']<-EC)].value_counts(),
                        labels = df1.Carrosserie[(df1['residus_normalisés']>EC)|(df1['residus_normalisés']<-EC)].value_counts().index,
                        autopct = lambda x: str(round(x,2))+'%',
                        labeldistance=1.2,
                        pctdistance = 0.8,
                        shadow =True)
            
        # Graphe gamme:
        plt.subplot(223)
        plt.pie(df1.gamme2[(df1['residus_normalisés']>EC)|(df1['residus_normalisés']<-EC)].value_counts(),
                labels = df1.gamme2[(df1['residus_normalisés']>EC)|(df1['residus_normalisés']<-EC)].value_counts().index,
                autopct = lambda x: str(round(x,2))+'%',
                labeldistance=1.2,
                pctdistance = 0.8,
                shadow =True)
        
        
        # Graphe carburant:
        plt.subplot(224)
        plt.pie(df1.Carburant[(df1['residus_normalisés']>EC)|(df1['residus_normalisés']<-EC)].value_counts(),
                labels = df1.Carburant[(df1['residus_normalisés']>EC)|(df1['residus_normalisés']<-EC)].value_counts().index,
                autopct = lambda x: str(round(x,2))+'%',
                pctdistance = 0.8,
                shadow =True)
        st.pyplot(fig)
        
        st.write('')
        st.write('')
        st.write('')
        
        st.markdown('###### Comparaison des puissance maximales et des masses en fonction de la valeur des résidus')
        
        fig = plt.figure(figsize = (10,3.5))    
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=1,
                            top=1,
                            wspace=0.4,
                            hspace=0)
        plt.subplot(121)
        sns.boxplot(data = df1, x = res_boxplot, y = 'puiss_max')
            
        plt.subplot(122)
        sns.boxplot(data = df1, x = res_boxplot, y = 'masse_ordma_min')
        st.pyplot(fig)
        
 
    
    if choix_EC == 'résidus influant trop fortement sur le modèle (distance de Cook)':
        fig = plt.figure(figsize = (16,8))
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0,
                            hspace=0.1)
        # Graphe Marque:
        plt.subplot(221)
        plt.pie(df1.Marque[(df1['observation_influente'] == 'observation influente')|(df1['observation_influente'] == 'observation influente')].value_counts(),
                        labels = df1.Marque[(df1['observation_influente'] == 'observation influente')|(df1['observation_influente'] == 'observation influente')].value_counts().index,
                       labeldistance=1.2,
                       pctdistance = 0.8,
                       autopct = lambda x: str(round(x,2))+'%',
                       shadow =True)
            
        # Graphe Carburant:
        plt.subplot(222)
        plt.pie(df1.Carrosserie[(df1['observation_influente'] == 'observation influente')|(df1['observation_influente'] == 'observation influente')].value_counts(),
                        labels = df1.Carrosserie[(df1['observation_influente'] == 'observation influente')|(df1['observation_influente'] == 'observation influente')].value_counts().index,
                       labeldistance=1.2,
                       pctdistance = 0.8,
                       autopct = lambda x: str(round(x,2))+'%',
                       shadow =True)
            
        # Graphe gamme:
        plt.subplot(223)
        plt.pie(df1.gamme2[(df1['observation_influente'] == 'observation influente')|(df1['observation_influente'] == 'observation influente')].value_counts(),
                        labels = df1.gamme2[(df1['observation_influente'] == 'observation influente')|(df1['observation_influente'] == 'observation influente')].value_counts().index,
                       labeldistance=1.2,
                       pctdistance = 0.8,
                       autopct = lambda x: str(round(x,2))+'%',
                       shadow =True)
        
        
        # Graphe carburant:
        plt.subplot(224)
        plt.pie(df1.Carburant[(df1['observation_influente'] == 'observation influente')|(df1['observation_influente'] == 'observation influente')].value_counts(),
                        labels = df1.Carburant[(df1['observation_influente'] == 'observation influente')|(df1['observation_influente'] == 'observation influente')].value_counts().index,
                       labeldistance=1.2,
                       pctdistance = 0.8,
                       autopct = lambda x: str(round(x,2))+'%',
                       shadow =True)
        st.pyplot(fig)
        
        st.write('')
        st.write('')
        st.write('')
        
        st.markdown('###### Comparaison des puissance maximales et des masses en fonction de la valeur des résidus')
        
        fig = plt.figure(figsize = (10,3.5))    
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=1,
                            top=1,
                            wspace=0.4,
                            hspace=0)
        plt.subplot(121)
        sns.boxplot(data = df1, x = df1['observation_influente'], y = 'puiss_max')
            
        plt.subplot(122)
        sns.boxplot(data = df1, x = df1['observation_influente'], y = 'masse_ordma_min')
        st.pyplot(fig)
    
    

# ANIMATION STREAMLIT------------------------------------------------------------------------------------------------------------------------------
if page == pages[3]:
    st.write('#### Modélisation: Régréssion multiple')
    
    tab1, tab2, tab3 = st.tabs(['Analyse de la variable cible CO₂', 'Régressions multiples', 'A vous de jouer!'])
    
    with tab1:
        c1, c2 = st.columns((1,1))
        with c1:
            st.markdown("###### Choississez le type d'analyse de la variable cible CO₂ 👇")
            Analyse_Y = st.radio(" ",
                                 ["Analyse globale", "Analyse par type de carburant"],
                               key="visibility",
                               horizontal = True)
            st.write('')
            from scipy.stats import norm
            # Analyse de la distribution target (CO2 (g/km)):
            dist = pd.DataFrame(target_reg)
            
            if Analyse_Y == "Analyse globale":
                fig = plt.figure(figsize =(8,5))
                
                # Espacement des graphes:
                plt.subplots_adjust(left=0.1,
                                    bottom=0.1,
                                    right=0.9,
                                    top=0.9,
                                    wspace=0.2,
                                    hspace=1) 
                plt.subplot(211)
                
                # Histogramme de distribution:
                plt.hist(dist, bins=60, density=True, rwidth = 0.8, color='steelblue')
                plt.title('Histogramme de CO₂ (g/km)')
                plt.xlim(0,400)
                plt.xlabel('CO₂ (g/km)')
                plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
                
                # Représentation de la loi normale avec la moyenne et l'écart-type de la distribution -
                # Affichage de la moyenne et la médiane de la distribution:
                
                x_axis = np.arange(0,400,1)
                plt.plot(x_axis, norm.pdf(x_axis, dist.mean(), dist.std()),'r', linewidth = 3)
                plt.xlim(0,400)
                plt.plot((dist.mean(), dist.mean()), (0, 0.015), 'r-', lw=1.5, label = 'moyenne de la distribution')
                plt.plot((dist.median(), dist.median()), (0, 0.015), 'r--', lw=1.5, label = 'médiane de la distribution')
                plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
                plt.legend()
                
                # Boite à moustache de la distribution:
                plt.subplot(212)
                sns.boxplot(x=dist.CO2, notch=True)
                plt.title('Boite à moustache de CO2 (g/km)')
                plt.xlabel('CO₂ (g/km)')
                plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
                plt.xlim(0,400)
                
                st.pyplot(fig)
                
            if Analyse_Y == "Analyse par type de carburant":
                fig = plt.figure(figsize =(10,12))
                
                # Espacement des graphes:
                plt.subplots_adjust(left=0.1,
                                    bottom=0.1,
                                    right=0.9,
                                    top=0.9,
                                    wspace=0.2,
                                    hspace=0.6)
                # Histogrammes de distribution des véhicules essence et des véhicules diesel :
                plt.subplot(311)
                ES = df.CO2[df['Carburant']=='ES']
                GO = df.CO2[df['Carburant']=='GO']
                
                plt.hist(ES,
                         bins=80,
                         density=True,
                         alpha=0.4,
                         color='green',
                         label ='Distribution des véhicules essence')
                
                plt.hist(GO,
                         bins=40,
                         density=True,
                         alpha=0.4,
                         color='orange',
                         label ='Distribution des véhicules diesel')
                
                plt.title('Histogramme de CO₂ (g/km) en fonction du carburant')
                plt.xlabel('CO₂ (g/km)')
                plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
                plt.legend()
                
                # Réprésentation des distributions des véhicules essence et diesel en prenant en compte uniquement
                # leurs moyennes et leurs écarts-types (= aspect d'une loi normale avec ces moyennes et ces écarts-types):
                ## Représentation de la loi normale avec la moyenne et l'écart-type de la distribution des véhicules essence ES:
                
                plt.subplot(312)
                x_axis = np.arange(0,400,1)
                plt.plot(x_axis,
                         norm.pdf(x_axis, ES.mean(), ES.std()),
                         'g',
                         linewidth = 3,
                         alpha = 0.8,
                         label ='loi normale [ES)]')
                plt.xlim(0,400)
                plt.plot((ES.mean(), ES.mean()), (0, 0.015), 'g', lw=1.5, label = 'moyenne de la distribution ES')
                plt.plot((ES.median(), ES.median()), (0, 0.015), 'g--', lw=1.5, label = 'médiane de la distribution ES')
                
                ## Représentation de la loi normale avec la moyenne et l'écart-type de la distribution des véhicules diesel GO:
                plt.plot(x_axis,
                         norm.pdf(x_axis, GO.mean(), GO.std()),
                         'orange',
                         linewidth = 3,
                         alpha = 0.8,
                         label ='loi normale [GO]')
                plt.xlim(0,400)
                plt.plot((GO.mean(), GO.mean()), (0, 0.015), 'y', lw=1.5, label = 'moyenne de la distribution GO')
                plt.plot((GO.median(), GO.median()), (0, 0.015), 'y--', lw=1.5, label = 'médiane de la distribution GO')
                plt.title('Représentation des lois normales des distributions des véhicules essence et diesel suivant leurs moyennes et écarts-types')
                plt.xlabel('CO₂ (g/km)')
                plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
                plt.legend()
                
                # Boite à moustache de la distribution en fonction du carburant:
                plt.subplot(313)
                sns.boxplot(data = df, y = 'Carburant' , x = 'CO2', palette = ['green','gold'], notch=True)
                plt.xticks(rotation = 'vertical')
                plt.title('Boite à moustache de CO₂ (g/km) en fonction du type de carburant')
                plt.xlabel('CO₂ (g/km)')
                plt.xlim(0,400)
                plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
                
                st.pyplot(fig)
                
                st.write('___')
                
                st.markdown("##### Graphe 4D représentant les rejets de CO₂ par type de carburant en fonction de la masse et de la puissance des véhicules.  \n##### Naviguez dans le graphe 4D en choisissant la vue 👇")
                import streamlit as st
                from PIL import Image
                
                graph4D = st.radio("",
                                   ["Vidéo", "Vue 1", "Vue 2", "Vue 3", "Vue 4"],
                                   #key="visibility",
                                   horizontal = True)
                if graph4D == 'Vidéo':
                    st.video('Graphe_4D.mp4', format="video/mp4", start_time=0)
                if graph4D == 'Vue 1':
                    image = Image.open('4D1.png')
                    st.image(image, caption='Représentation des rejets de CO₂ des  véhicules en fonction de leurs masses, leurs poids et leurs carburants')
                if graph4D == 'Vue 2':
                    image = Image.open('4D2.png')
                    st.image(image, caption='Représentation des rejets de CO₂ des  véhicules en fonction de leurs masses, leurs poids et leurs carburants')
                if graph4D == 'Vue 3':
                    image = Image.open('4D3.png')
                    st.image(image, caption='Représentation des rejets de CO₂ des  véhicules en fonction de leurs masses, leurs poids et leurs carburants')
                if graph4D == 'Vue 4':
                    image = Image.open('4D4.png')
                    st.image(image, caption='Représentation des rejets de CO₂ des  véhicules en fonction de leurs masses, leurs poids et leurs carburants')

               
 
        
    with tab2:
        st.markdown("##### Quel dataset voulez-vous analyser? 👇")
        choix_dataset = st.radio("",
                             ["Dataset complet (véhicules essence et diesel)",
                              "Véhicules diesel uniquement",
                              "Véhicules essence uniquement"],
                             #key="visibility",
                             horizontal = True)
        
        st.write('___')
        st.markdown("##### Quel modèle voulez-vous analyser? 👇")
        choix_model = st.radio(" ",
                             ["Modèle général",
                              "Modéle affiné"],
                           #key="visibility",
                           horizontal = True)
      
    with tab2:
        st.markdown("**Méthodologie**:  \n1. sélection du dataset,  \n2. construction d'un premier modèle général à partir de l'ensemble des variables du dataset,  \n3. construction d'un second modèle affiné après sélection des variables les plus influentes,  \n3. pour chaque modèle: analyse des metrics et résidus et sélection des données les plus pertinentes, puis retour à l'étape 1")
        st.write('___')
        c1, c2, c3= st.columns((0.4, 0.4, 1))
        with c1:
            st.markdown("###### Dataset à analyser: 👇")
            choix_dataset = st.radio("",
                                     ["Dataset complet (véhicules essence et diesel)",
                                      "Véhicules diesel uniquement",
                                      "Véhicules essence uniquement"],
                                     key="visibility")
        
        with c2:
            st.markdown("###### Modèle de régression à analyser: 👇")
            choix_model = st.radio("",
                                   ["Modèle général",
                                    "Modèle affiné"],
                                   key="visibility")
        with c3:
             st.markdown("###### Analyse: 👇")
             choix_param = st.radio("",
                                    ["Metrics & Coefficients des variables",
                                     "Résidus"],
                                    key="visibility")

        st.write('___')
                              
        if choix_dataset == 'Dataset complet (véhicules essence et diesel)':
            dataset = data
            cible = target_reg
            model = 'lr.joblib'
           
        if choix_dataset == 'Véhicules diesel uniquement':
            dataset = data_go
            cible = target_go
            model = 'lr_go.joblib'
            
        if choix_dataset == 'Véhicules essence uniquement':
            dataset = data_es
            cible = target_es
            model = 'lr_es.joblib'
        
        if choix_model == "Modèle général":
            #Standardisation, split du dataset, régression:
                X_train, X_test, y_train, y_test = standardisation_lr(dataset, cible)
                lr, pred_train, pred_test = regression_lineaire(model, X_train, y_train, X_test, y_test)
        
        if choix_model == "Modèle affiné":
            #Standardisation, split du dataset, régression:
                X_train, X_test, y_train, y_test = standardisation_lr(dataset, cible)
                lr_sfm, pred_train, pred_test, sfm_train, sfm_test = selecteur(X_train, y_train, X_test, y_test)
        
        if choix_param == "Metrics & Coefficients des variables":
            c1, c2, c3, c4 = st.columns((1, 1.2, 0.2, 1.1))
            if choix_model == "Modèle général":
                with c1:
                    st.write("##### **Metrics:**")
                    st.write('')
                    metrics_lr(lr, X_train, y_train, X_test, y_test, pred_train, pred_test)
            
                with c2:
                    st.write("##### **Coefficients des variables:**")
                    coef_lr(lr, X_train)
            
            if choix_model == "Modèle affiné":
                with c1:
                    st.write("##### **Metrics:**")
                    st.write('')
                    metrics_sfm(lr_sfm, X_train, y_train, X_test, y_test, pred_train, pred_test, sfm_train, sfm_test)
                    
                with c2:
                    st.write("##### **Coefficients des variables retenues par le modèle:**")
                    coef_sfm(lr_sfm, sfm_train)
                
                with c4:
                    
                    if choix_dataset == 'Dataset complet (véhicules essence et diesel)':
                        st.markdown("##### Représentation graphique de la cible CO₂ par type de carburant en fonction de la masse et de la puissance des véhicules:")
                        import streamlit as st
                        from PIL import Image
                            
                        graph4D = st.radio("",
                                           ["Vidéo", "Vue 1", "Vue 2", "Vue 3", "Vue 4"],
                                           key="visibility",
                                           horizontal = True)
                        if graph4D == 'Vidéo':
                            st.video('Graphe_4D.mp4', format="video/mp4", start_time=0)
                        if graph4D == 'Vue 1':
                            image = Image.open('4D1.png')
                            st.image(image)
                        if graph4D == 'Vue 2':
                            image = Image.open('4D2.png')
                            st.image(image)
                        if graph4D == 'Vue 3':
                            image = Image.open('4D3.png')
                            st.image(image)
                        if graph4D == 'Vue 4':
                            image = Image.open('4D4.png')
                            st.image(image)
                                            
        if choix_param == "Résidus":
                    
            if choix_model == "Modèle général":
                with c1:
                    st.write("##### **Analyse graphique des résidus:**")
                    residus, residus_norm, residus_std = graph_res(y_train, y_test,
                                                                   pred_train,
                                                                   pred_test)
                    st.write('')
                    st.write('')
                    st.write('')
                    
                    st.write("##### **Analyse graphique spécifique des résidus élevés et fortement influents:**")               
                    df_res(X_train, y_train, pred_train, residus)
                    
            if choix_model == "Modèle affiné":
                with c1:
                    st.write("##### **Analyse graphique des résidus:**")
                    residus, residus_norm, residus_std = graph_res_sfm(y_train, y_test,
                                                                       pred_train,
                                                                       pred_test)
                    st.write('')
                    st.write('')
                    st.write('')
                    
                    st.write("##### **Analyse graphique spécifique des résidus élevés et fortement influents:**")               
                    df_res(sfm_train, y_train, pred_train, residus)
    
        
    with tab3:
        st.markdown("**:blue[1. Premier modèle]**")
        st.markdown(":blue[Résidus]")
        st.markdown("blablabla et blablabla")
        st.markdown(":blue[Metrics]")
        st.markdown("blablabla et blablabla")
        st.markdown(' ')
        
        st.markdown("**:blue[2. Modèle affiné]**")
        st.markdown(":blue[Résidus]")
        st.markdown("reblablabla et reblablabla")
        st.markdown(":blue[Metrics]")
        st.markdown("reblablabla et reblablabla")
        st.markdown(' ')
        


#_______________________________________________________________________________________________________
#
#                                   Page 4 : classification 
#_______________________________________________________________________________________________________

# CHARGEMENT DES JEUX DE DONNEES NETTOYES ET DES TARGETS CORRESPONDANTES: ------------------------------
df = pd.read_csv('Model_C02.csv', index_col = 0)
    
# On sépare les variables numériques et catégorielles
var_num = df.select_dtypes(exclude = 'object') # On récupère les variables numériques
var_cat = df.select_dtypes(include = 'object') # On récupère les variables catégorielles

# On récupère la variable cible
target_class = df['Cat_CO2'].squeeze()     # Variable cible pour la classification

var_num = var_num.drop(['CO2'], axis = 1)  # Les variables cibles sont éliminées des variables numériques
var_cat = var_cat.drop(['Cat_CO2'], axis = 1)

# Les variables catégorielles sont transformées en indicatrices
var_cat_ind = pd.get_dummies(var_cat)

# On récupère les variables explicatives
data = var_num.join(var_cat_ind)

# FONCTIONS: ----------------------------------------------------------------------------


def classification(model, X_train, y_train, X_test, y_test):
            
    # Entraînement et prédictions:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) # Prédictions sur les valeurs de test
    
    # Evaluation
    score = accuracy_score(y_test, y_pred)
        
    return [y_pred, score]

# Fonction pour afficher les 3 matrices de confusion des 3 modèles optimisés:
def matrice(matrice, titre):
    classes = ['A','B','C','D','E','F','G']
    plt.imshow(matrice, interpolation='nearest',cmap='Blues')
    plt.title(titre)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.grid(False)
    
    for i, j in itertools.product(range(matrice.shape[0]), range(matrice.shape[1])):
        plt.text(j, i, matrice[i, j],
                 horizontalalignment="center",
                 color="white" if (matrice[i, j] > ( matrice.max() / 2) or matrice[i, j] == 0) else "black")
    plt.ylabel('Catégories réelles')
    plt.xlabel('Catégories prédites')
    st.pyplot()
    
def report(rapport, titre):
    classes = ['A','B','C','D','E','F','G']
    plt.imshow(rapport, interpolation='nearest',cmap='Blues')
    plt.title(titre)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.grid(False)
    

    plt.ylabel('Catégories réelles')
    plt.xlabel('Catégories prédites')
    st.pyplot()

if page == pages[4]:
    st.write('#### Modélisation: Classification multi-classes')
        
    st.markdown("Explication de la démarche:  \n - un **premier modèle** est généré à partir de l'ensemble des hyperparamètres,  \n - un **second modèle optimisé** est généré après sélection des meilleurs hyperparamètres.")
    st.markdown('Nous procédons à une classification multiple. Nous avons donc choisi les classifieurs adaptés.')
    st.markdown('Nous en avons sélectionné 3 pour cette étude: SVM, KNN et Random Forest')
    tab1, tab2, tab3, tab4 = st.tabs(['Données', 'Classifications multiples', 'Comparaison des modèles', 'A vous de jouer!'])
    
    
    # Séparation en données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(data, target_class,
                                                    test_size = 0.25,
                                                    random_state = 2,
                                                    stratify = target_class)

    # Les variables numériques doivent être standardisées
    cols = ['puiss_max', 'masse_ordma_min']
    sc = StandardScaler()
    X_train[cols] = sc.fit_transform(X_train[cols])
    X_test[cols] = sc.transform(X_test[cols])
    
    with tab1:
        st.write('#### Revue des données à classifier')
        st.markdown('Le DataFrame utilisé pour la classification multiple comporte : \n')
        c1, c2 = st.columns((1.5, 1.5))
        
        with c1:
            st.markdown('Les variables catégorielles :\n')
            st.write(var_cat.head())
            st.markdown('Les variables numériques :\n')
            st.write(var_num.head())       
        
        with c2:
            st.markdown('La variable cible :\n')
            st.write(target_class.head())
            st.markdown('Répartition de la variable cible :')
            sns.countplot(target_class)
            st.pyplot()
            
    with tab2:
        st.markdown("##### Quel modèle voulez-vous analyser? 👇")
        choix_model = st.radio(" ",
                             ["SVM",
                              "KNN",
                              "Random Forest"],
                           key="visibility",
                           horizontal = True)
        
        c1, c2, c3 = st.columns((1.2, 1.5, 1.5))
        
        
        if choix_model == 'SVM':
            model = SVC(gamma = 'scale')
            model_optim = SVC(C = 200, 
                              kernel = 'rbf', 
                              gamma = 0.1) 
           
        if choix_model == 'KNN':
            model = KNeighborsClassifier()
            model_optim = KNeighborsClassifier(n_neighbors = 1, 
                                               metric = 'minkowski', 
                                               leaf_size = 1)
            
        if choix_model == 'Random Forest':            
            model = RandomForestClassifier()
            model_optim = RandomForestClassifier(n_estimators = 400,
                                                 criterion = 'gini',
                                                 max_features = 'sqrt')        
                   
        # Prédictions
        y_pred, score = classification(model, X_train, y_train, X_test, y_test)
        y_pred_optim, score_optim = classification(model_optim, X_train, y_train, X_test, y_test)
        
        #Les matrices de confusion obtenues
        matrix = confusion_matrix(y_test, y_pred)
        matrix_optim = confusion_matrix(y_test, y_pred_optim)
        
        fig = plt.figure(figsize = (10,10))
        st.set_option('deprecation.showPyplotGlobalUse', False)

        # Espacement des graphes:
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.5,
                            hspace=0.4)
        with c1:
            st.write("##### **Metrics:**")
            st.write('Le modèle standard a une accuracy de', round(score, 2))        
            st.write('Le modèle optimisé a une accuracy de', round(score_optim, 2))
            
        with c2:
            plt.subplot(121)
            matrice(matrix, 'Matrice de confusion du modèle standard')
            st.write('Le rapport de classification standard')
            st.text(classification_report(y_test, y_pred)) 
            
        
        with c3:
            plt.subplot(122)
            matrice(matrix_optim, 'Matrice de confusion du modèle optimisé')
            st.write('Le rapport de classification optimisée')
            st.text(classification_report(y_test, y_pred_optim)) 
                
        
        
    with tab3:
        
        """# Affichage et comparaison des 3 matrices de confusion des 3 modèles optimisés
        st.write("Comparaison des matrices de confusion des 3 modèles optimisés \n")

        plt.figure(figsize = (16,6))

        # Espacement des graphes:
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.5,
                            hspace=0.4)

        # Matrice de confusion du modèle SVM optimisé:
        plt.subplot(131)
        matrice(cnf_matrix_svc_opt, 'Matrice de confusion du modèle SVM optimisé')

        # Matrice de confusion du modèle KNN optimisé:
        plt.subplot(132)
        matrice(cnf_matrix_knn_opt, 'Matrice de confusion du modèle knn optimisé')

        # Matrice de confusion du modèle RF optimisé:
        plt.subplot(133)
        matrice(cnf_matrix_rf_opt, 'Matrice de confusion du modèle RF optimisé')
        st.markdown("**:blue[Vous avez carte blanche! Sélectionnez vous-même les hyperparamètres et tentez d'être meilleur que l'algorithme GridSearchCV!]**")
        st.markdown(":blue[Résidus]")
        st.markdown("blablabla et blablabla")
        st.markdown(":blue[Metrics]")
        st.markdown("blablabla et blablabla")
<<<<<<< Updated upstream
        st.markdown(' ')"""


#_______________________________________________________________________________________________________
#
#                                   Page 5 : Interprétation SHAP multi-classes 
#_______________________________________________________________________________________________________

#CHARGEMENT DES LIBRAIRIES: ----------------------------------------------------------------------------

import shap
from sklearn.tree import plot_tree

# CHARGEMENT DES MODELES: ------------------------------------------------------------------------
shap_values_rf = load('shap_values_rf.joblib')
shap_values_rf_opt = load('shap_values_rf_opt.joblib')
shap_values_svc_opt = load('shap_values_svc_opt.joblib')
shap_values_knn_opt = load('shap_values_knn_opt.joblib')

df_class = pd.read_csv('df.csv', index_col = 0)
df_class = df_class.drop('Cat_CO2', axis = 1)
df_class = df_class.drop('CO2', axis = 1)

target_class = pd.read_csv('target_class.csv', index_col = 0)
target_class = target_class.squeeze()

# Preprocessing dataset:
  
# On sépare les variables numériques et catégorielles
var_num = df_class.select_dtypes(exclude = 'object') # On récupère les variables numériques
var_cat = df_class.select_dtypes(include = 'object') # On récupère les variables catégorielles

# Les variables catégorielles sont transformées en indicatrices
var_cat_ind = pd.get_dummies(var_cat, drop_first = True)

# On récupère les variables explicatives
feats = var_num.join(var_cat_ind)


# Pour rappel, le dataset feats regroupe les variables numériques et catégorielles
# On répartit équitablement les classes entre les jeux d'entrainement et de test.
X_train, X_test, y_train, y_test = train_test_split(feats, target_class,
                                                    test_size = 0.25,
                                                    random_state = 2,
                                                    stratify = target_class)

# Les variables numériques doivent être standardisées
cols = ['puiss_max', 'masse_ordma_min']
sc = StandardScaler()
X_train[cols] = sc.fit_transform(X_train[cols])
X_test[cols] = sc.transform(X_test[cols])

# ANIMATION STREAMLIT------------------------------------------------------------------------------------------------------------------------------
if page == pages[5]:
    st.write('#### Interprétation SHAP multi-classes')
    st.markdown("###### Modèle de classification à analyser: 👇")
    choix_model_shap = st.radio("",
                             ["Random Forest",
                              "Random Forest optimisé",
                              "SVC optimisé",
                              "KNN optimisé"],
                             key="visibility",
                             horizontal=True)
    
    if choix_model_shap == "Random Forest optimisé":
        model = 'rf_opt'
        shap_values = shap_values_rf_opt
    if choix_model_shap == "SVC optimisé":
        model = 'svc_opt'
        shap_values = shap_values_svc_opt
    if choix_model_shap == "KNN optimisé":
        model = 'knn_opt'
        shap_values = shap_values_knn_opt
    if choix_model_shap == "Random Forest":
        model = 'knn'
        shap_values = shap_values_rf
    st.write('')
    st.write('')

    tab1, tab2, tab3 = st.tabs(['Interprétabilité globale', 'Interprétabilité locale', 'A vous de jouer!'])
    
    with tab1:
        c1, c2, c3, c4 = st.columns((0.7, 0.1, 0.75, 0.2))
        with c1:
            st.write("L'interprétabilité globale permet d'expliquer le fonctionnement du modèle de point de vue général.")
            st.write('')
            st.markdown("###### Summary plot à afficher: 👇")
            choix_model_shap = st.radio("",
                                     ["summary plot global",
                                      "summary plot par catégorie"],
                                     key="visibility",
                                     horizontal=True)
            
            if choix_model_shap == "summary plot par catégorie":
                st.write('')
                st.write('')
                st.markdown("###### Catégorie à analyser: 👇")
                choix_categorie = st.radio("",
                                     ["Catégorie A",
                                      "Catégorie B",
                                      "Catégorie C",
                                      "Catégorie D",
                                      "Catégorie E",
                                      "Catégorie F",
                                      "Catégorie G"],
                                     key="visibility",
                                     horizontal=True)
                
        with c3:  
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            if choix_model_shap == "summary plot global":
            # Summary_plot:
                fig = plt.figure()
                shap.summary_plot(shap_values,
                                  X_test,
                                  plot_type="bar",
                                  class_names = ['A', 'B', 'C', 'D', 'E', 'F','G'])
                st.pyplot(fig)
                
            if choix_model_shap == "summary plot par catégorie":                
                if choix_categorie == "Catégorie A":
                    fig = plt.figure()
                    shap.summary_plot(shap_values[0],
                                      X_test,
                                      feature_names=feats.columns)
                    st.pyplot(fig)
                
                if choix_categorie == "Catégorie B":
                    fig = plt.figure()
                    shap.summary_plot(shap_values[1],
                                      X_test,
                                      feature_names=feats.columns)
                    st.pyplot(fig)
                    
                if choix_categorie == "Catégorie C":
                    fig = plt.figure()
                    shap.summary_plot(shap_values[2],
                                      X_test,
                                      feature_names=feats.columns)
                    st.pyplot(fig)
                    
                if choix_categorie == "Catégorie D":
                    fig = plt.figure()
                    shap.summary_plot(shap_values[3],
                                      X_test,
                                      feature_names=feats.columns)
                    st.pyplot(fig)
                    
                if choix_categorie == "Catégorie E":
                    fig = plt.figure()
                    shap.summary_plot(shap_values[4],
                                      X_test,
                                      feature_names=feats.columns)
                    st.pyplot(fig)
                    
                if choix_categorie == "Catégorie F":
                    fig = plt.figure()
                    shap.summary_plot(shap_values[5],
                                      X_test,
                                      feature_names=feats.columns)
                    st.pyplot(fig)
                    
                if choix_categorie == "Catégorie G":
                    fig = plt.figure()
                    shap.summary_plot(shap_values[6],
                                      X_test,
                                      feature_names=feats.columns)
                    st.pyplot(fig)
                    
    with tab2:
        c1, c2  = st.columns((0.75, 1))
        with c1:
            st.write("L'interprétabilité locale permet d'expliquer le fonctionnement du modèle pour une instance.")
            st.write('')
            st.write('AFFICHER MATRICE CORRESPONDANT AU MODELE SELECTIONNE')
            

            
            
            
                
