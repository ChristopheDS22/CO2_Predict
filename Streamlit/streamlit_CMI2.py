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
         'Modélisation : Regression multiple', 'Modélisation : Classification multi-classes', 'Interprétabilité SHAP', 
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




#_______________________________________________________________________________________________________
#
#                                   Page 3 : régression 
#_______________________________________________________________________________________________________

#CHARGEMENT DES LIBRAIRIES: ----------------------------------------------------------------------------

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
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from scipy.stats import jarque_bera

## Les fonctions de sauvegarde et chargement de modèles
from joblib import dump, load

from sklearn.model_selection import train_test_split


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
    st.write("R2 modèle_train =", round(lr_sfm.score(sfm_train, y_train),2))
    st.write("R2 obtenu par CV =", round(cross_val_score(lr_sfm,sfm_train,y_train).mean(),2))
    st.write("R2 modèle_test =", round(lr_sfm.score(sfm_test, y_test),2))
    st.write("")
    st.write('RMSE train =', round(np.sqrt(mean_squared_error(y_train, pred_train)),2))
    st.write('RMSE test =', round(np.sqrt(mean_squared_error(y_test, pred_test)),2))
    st.write("")
    st.write("MAE_train:", round(mean_absolute_error(y_train, pred_train),2))
    st.write("MAE_test:", round(mean_absolute_error(y_test, pred_test),2))


def metrics_lr(lr, X_train, y_train, X_test, y_test, pred_train, pred_test):
    # Affichage des metrics:
    st.write("R² modèle_train =", round(lr.score(X_train, y_train),2))
    st.write("R² obtenu par CV =", round(cross_val_score(lr,X_train,y_train, cv = 5).mean(),2))
    st.write("R² modèle_test =", round(lr.score(X_test, y_test),2))
    st.write("")
    st.write('RMSE_train =', round(np.sqrt(mean_squared_error(y_train, pred_train)),2))
    st.write('RMSE_test =', round(np.sqrt(mean_squared_error(y_test, pred_test)),2))
    st.write("")
    st.write("MAE_train:", round(mean_absolute_error(y_train, pred_train),2))
    st.write("MAE_test:", round(mean_absolute_error(y_test, pred_test),2))
    
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
    sns.boxplot(residus)
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
                        top=0.9,wspace=0.2,
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
    sns.boxplot(residus)
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
    
    

# ANIMATION STREAMLIT------------------------------------------------------------------------------------------------------------------------------
if page == pages[3]:
    st.write('#### Modélisation: Régréssion multiple')
    st.markdown("Chaque modèle de régréssion a été construit selon la même structure:  \n - un **premier modèle général** est généré à partir de l'ensemble des variables du dataset,  \n - un **second modèle affiné** est calculé après sélection des variables les plus influentes.")
    tab1, tab2, tab3, tab4 = st.tabs(['Analyse de la variable cible CO₂', 'Régréssions multiples', 'Comparaison des modèles', 'A vous de jouer!'])
    
    with tab1:
        st.caption("Graphique")
        c1, c2 = st.columns((1,1))
        with c1:
            from scipy.stats import norm
            fig = plt.figure(figsize =(10,15))
            
            # Espacement des graphes:
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.2,
                                hspace=0.6) 
            plt.subplot(511)
            
            # Analyse de la distribution target (CO2 (g/km)):
            dist = pd.DataFrame(target_reg)
            
            # Histogramme de distribution:
            plt.hist(dist, bins=60, density=True, alpha=0.6, color='b')
            plt.title('Histogramme de CO2 (g/km)')
            plt.xlim(0,400)
            plt.xlabel('CO2 (g/km)')
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
            plt.subplot(512)
            sns.boxplot(x=dist.CO2, notch=True)
            plt.title('Boite à moustache de CO2 (g/km)')
            plt.xlabel('CO2 (g/km)')
            plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
            plt.xlim(0,400)
            
            # Histogrammes de distribution des véhicules essence et des véhicules diesel :
            plt.subplot(513)
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
            
            plt.title('Histogramme de CO2 (g/km) en fonction du carburant')
            plt.xlabel('CO2 (g/km)')
            plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
            plt.legend()
            
            # Réprésentation des distributions des véhicules essence et diesel en prenant en compte uniquement
            # leurs moyennes et leurs écarts-types (= aspect d'une loi normale avec ces moyennes et ces écarts-types):
            ## Représentation de la loi normale avec la moyenne et l'écart-type de la distribution des véhicules essence ES:
            
            plt.subplot(514)
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
            plt.xlabel('CO2 (g/km)')
            plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
            plt.legend()
            
            # Boite à moustache de la distribution en fonction du carburant:
            plt.subplot(515)
            sns.boxplot(data = df, y = 'Carburant' , x = 'CO2', palette = ['green','gold'], notch=True)
            plt.xticks(rotation = 'vertical')
            plt.title('Boite à moustache de CO2 (g/km) en fonction du type de carburant')
            plt.xlabel('CO2 (g/km)')
            plt.xlim(0,400)
            plt.grid(linestyle = ':', c = 'g', alpha = 0.3)
            st.pyplot(fig)
        
        # Représentation graphique en 4D de l'influence de ces 3 variables significatives sur la variable explicative target (= rejet CO2):
        #from mpl_toolkits.mplot3d import Axes3D
        #%matplotlib notebook
        
        #fig = plt.figure(figsize = (9,9))
        #ax = fig.add_subplot(111, projection='3d')
        
        #z = y_train
        #x = sfm_train['puiss_max']
        #y = sfm_train['masse_ordma_min']
        
        #ax.scatter(x, y, z,  c=sfm_train['Carburant'], cmap = ('viridis'))
        #ax.set_xlabel('Puissance véhicules (valeurs standardisées)')
        #ax.set_ylabel('Masse véhicules (valeurs standardisées)')
        #ax.set_zlabel('CO2 (g/km)')
        
        #plt.legend(['ES','GO'])
        #plt.title('Représentation des rejets de CO2 des  véhicules en fonction de leurs masses, leurs poids et leurs carburants')
        #st.pyplot(fig)
        
        st.caption("Interprétation")

        
    with tab2:
        choix_dataset = st.selectbox('Quel dataset voulez-vous analyser?',
                                     ('Dataset complet (véhicules essence et diesel)',
                                      'Véhicules diesel uniquement',
                                      'Véhicules essence uniquement'))
        
        choix_model = st.selectbox('Quel dataset voulez-vous analyser?',
                                   ("Modèle général",
                                    "Modéle affiné"))
                              
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
        

        c1, c2, c3 = st.columns((0.7, 1.1, 2.2))
        
        if choix_model == "Modèle général":
            with c1:
                st.write("##### **Metrics:**")
                st.write('')
            
            #Standardisation, split du dataset, régression:
                X_train, X_test, y_train, y_test = standardisation_lr(dataset, cible)
                lr, pred_train, pred_test = regression_lineaire(model, X_train, y_train, X_test, y_test)
                metrics_lr(lr, X_train, y_train, X_test, y_test, pred_train, pred_test)
            
            with c2:
                st.write("##### **Coefficients des variables:**")
                coef_lr(lr, X_train)
            
            with c3:
                st.write("##### **Analyse graphique des résidus:**")
                residus, residus_norm, residus_std = graph_res(y_train, y_test,
                                                               pred_train,
                                                               pred_test)
        if choix_model == "Modéle affiné":
            with c1:
                st.write("##### **Metrics:**")
                st.write('')
            
            #Standardisation, split du dataset, régression:
                X_train, X_test, y_train, y_test = standardisation_lr(dataset, cible)
                lr_sfm, pred_train, pred_test, sfm_train, sfm_test = selecteur(X_train, y_train, X_test, y_test)
                metrics_sfm(lr_sfm, X_train, y_train, X_test, y_test, pred_train, pred_test, sfm_train, sfm_test)
            
            with c2:
                st.write("##### **Coefficients des variables retenues par le modèle:**")
                coef_sfm(lr_sfm, sfm_train)    
            
            with c3:
                st.write("##### **Analyse graphique des résidus:**")
                residus, residus_norm, residus_std = graph_res_sfm(y_train, y_test,
                                                                   pred_train,
                                                                   pred_test)

        
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
        
        
    with tab4:
        st.markdown("**:blue[Vous avez carte blanche! Sélectionnez vous-même les paramètres et tentez d'être meilleur que l'algorithme SFM!]**")
        st.markdown(":blue[Résidus]")
        st.markdown("blablabla et blablabla")
        st.markdown(":blue[Metrics]")
        st.markdown("blablabla et blablabla")
        st.markdown(' ')
        

#------------------------------------  Page 4 : classification ---------------------------------------------

if page == pages[4]:
    st.write('#### Modélisation: Classification multi-classes')
    st.markdown("Explication de la démarche:  \n - un **premier modèle** est généré à partir de l'ensemble des hyperparamètres,  \n - un **second modèle optimisé** est généré après sélection des meilleurs hyperparamètres.")
    tab1, tab2, tab3 = st.tabs(['Classifications multiples', 'Comparaison des modèles', 'A vous de jouer!'])
    
    with tab1:
        st.caption("Graphique")
        st.caption("Interprétation")

        
    with tab2:
        st.markdown("**:blue[1. Premier modèle]**")
        st.markdown(":blue[Résidus]")
        st.markdown("blablabla et blablabla")
        st.markdown(":blue[Metrics]")
        st.markdown("blablabla et blablabla")
        st.markdown(' ')
        
        st.markdown("**:blue[2. Modèle optimisé]**")
        st.markdown(":blue[Résidus]")
        st.markdown("reblablabla et reblablabla")
        st.markdown(":blue[Metrics]")
        st.markdown("reblablabla et reblablabla")
        st.markdown(' ')
        
        
    with tab3:
        st.markdown("**:blue[Vous avez carte blanche! Sélectionnez vous-même les hyperparamètres et tentez d'être meilleur que l'algorithme GridSearchCV!]**")
        st.markdown(":blue[Résidus]")
        st.markdown("blablabla et blablabla")
        st.markdown(":blue[Metrics]")
        st.markdown("blablabla et blablabla")
        st.markdown(' ')
