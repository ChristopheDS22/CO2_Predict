# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 22:01:06 2022

@author: Gilles.NGAMENYE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Titre du streamlit
st.title('Projet CO2 Predict')

# Sommaire du Sreamlit
st.sidebar.title('Sommaire')
pages = ['Introduction','Exploration et analyse des données', 'Analyse de données - à supprimer', 'Dataviz - à supprimer', 
         'Modélisation : Classification multiple', 'Modélisation : Regréssion']

page = st.sidebar.radio('Aller vers', pages)

# Emissions de polluants, CO2 et caractéristiques des véhicules
# commercialisés en France en 2013
df_2013 = pd.read_csv('data_2013.csv' , sep = ';', encoding='unicode_escape')

st.write('### Visualisation du dataset')
st.dataframe(df_2013.head())

if page == pages[0]:
    st.write('## Introduction au projet')
    
    st.markdown('Identifier les véhicules qui émettent le plus de CO2 est important pour identifier les caractéristiques techniques qui jouent un rôle dans la pollution.')
    st.markdown(' Prédire à l’avance cette pollution permet de prévenir dans le cas de l’apparition de nouveaux types de véhicules (nouvelles séries de voitures par exemple)')
    st.markdown('Le projet est effectué à partir du dataset regroupant les émissions de CO2 et polluants des véhicules commercialisées en France en 2013')
    st.markdown('[Source du dataset]( https://www.ecologie.gouv.fr/normes-euros-demissions-polluants-vehicules-lourds-vehicules-propres)')
    
if page == pages[1]:
    st.write('## Exploration des données')
    
    st.markdown('La variable cible est CO2 (g/km). Elle peut être expliquée en tant que variable continue ou discrète')
    #st.markdown('Il y a ', df_2013.duplicated().sum(), 'doublons dans le dataset') # On checke les doublons
    
    # Suppression des doublons
    df_2013 = df_2013.drop_duplicates()
    df_2013.duplicated().sum()
    
    
    # Remplacement des valeurs manquantes des 'EL' par 0:
    df_2013['CO2 (g/km)'] = df_2013['CO2 (g/km)'].fillna(0)

    # Les valeurs manquantes de consommation semblent coller avec celles des véhicules électriques
    df_2013['Consommation urbaine (l/100km)'] = df_2013['Consommation urbaine (l/100km)'].fillna(0)
    df_2013['Consommation extra-urbaine (l/100km)'] = df_2013['Consommation extra-urbaine (l/100km)'].fillna(0)
    df_2013['Consommation mixte (l/100km)'] = df_2013['Consommation mixte (l/100km)'].fillna(0)

    # Les valeurs manquantes des variables NOX (g/km), HC+NOX (g/km) et Particules (g/km) sont remplacées par des moyennes
    df_2013['NOX (g/km)'] = df_2013['NOX (g/km)'].fillna(df_2013['NOX (g/km)'].mean())
    df_2013['HC+NOX (g/km)'] = df_2013['HC+NOX (g/km)'].fillna(df_2013['HC+NOX (g/km)'].mean())
    df_2013['Particules (g/km)'] = df_2013['Particules (g/km)'].fillna(df_2013['Particules (g/km)'].mean())
    df_2013['CO type I (g/km)'] = df_2013['CO type I (g/km)'].fillna(df_2013['CO type I (g/km)'].mean())

    # La variable HC (g/km) a près de 77% de valeurs manquantes. Elle ne sera pas retenue.
    df_2013 = df_2013.drop(['HC (g/km)'], axis = 1)

    # On vérifie le remplacement des valeurs manquantes des variables numériques
    df_2013.isna().sum()
    

    
    # Harmonisation des nomenclatures de carburants
    df_2013['Carburant'] = df_2013['Carburant'].replace(to_replace = ['GN/ES', 'GP/ES'], value = ['ES/GN', 'ES/GP'])

    # Nouveau count
    df_2013['Carburant'].value_counts()

    # Corrélation entre les variables quantitatives
    # Extraction des variables quantitatives
    df_quant = df_2013[['Puissance administrative', 'Puissance maximale (kW)', 'Consommation urbaine (l/100km)', 'Consommation extra-urbaine (l/100km)', 'Consommation mixte (l/100km)', 'CO2 (g/km)', 'CO type I (g/km)', 'Particules (g/km)', 'NOX (g/km)', 'masse vide euro min (kg)', 'masse vide euro max (kg)']]


    
    # Création d'une variable catégorielle 'Cat' selon la norme européenne de pollution des véhicules:

    # Création d'une liste:
    A = []

    # Création d'une boucle identifiant les catégories (A à G) selon l'émission de CO2:
    for i in df_2013['CO2 (g/km)']:
        if i<=100:
            A.append('A')
        if 100<i<=120:
            A.append('B')
        if 120<i<=140:
            A.append('C')
        if 140<i<=160:
            A.append('D')
        if 160<i<=200:
            A.append('E')
        if 200<i<250:
            A.append('F')
        if i>=250:
            A.append('G')

    # Création d'une colonne 'Cat' dans le DataFrame df_2013:
    df_2013['Cat'] = A 

#------------------------------
#partie streamlit des représentations
#------------------------------

    st.write('### Variable à expliquer')


    fig=px.box(df_2013['CO2 (g/km)'],x='CO2 (g/km)',title='Variable CO2 continue',notched=True)
    st.plotly_chart(fig)

    fig2= px.histogram(df_2013,title='Variable CO2 discrète',x='Cat',category_orders=dict(Cat=['A','B','C','D','E','F','G']))
    st.plotly_chart(fig2)
    
 #   from plotly.subplots import make_subplots
#essais avec subplots
#    fig3=make_subplots(rows=1,cols=2)
#    fig3.add_trace(
#        go.box(df_2013['CO2 (g/km)']),row=1,col=1
#        )
#    fig3.add_trace(
#        go.scatter(x=[20, 30, 40], y=[50, 60, 70]),
#    row=1, col=2
#    )
#    fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
#    fig.show()
#    fig = make_subplots(rows=1, cols=2)

    st.write('### Variables explicatives')
    st.write('Il y a deux types de variables explicatives : qualitatives et quantitatives')

   
    var_num_2013 = df_2013.select_dtypes(exclude = 'object') # On récupère les variables numériques
    var_cat_2013 = df_2013.select_dtypes(include = 'object') # On récupère les variables catégorielles

    tab_num=pd.DataFrame(var_num_2013.columns,columns=['Quantitatives'])
    tab_cat=pd.DataFrame(var_cat_2013.columns,columns=['Qualitatives'])
    
    st.dataframe(pd.concat([tab_num,tab_cat],axis=0))

    st.write('Variables quantitatives')



    st.write('Variables qualitatives')

    fig_mq= px.histogram(df_2013,title='Répartition par marque',y='Marque').update_yaxes(categoryorder='total ascending')
    st.plotly_chart(fig_mq)
    st.write('on observe une sur-représentation de Mercedes-Benz dans le dataframe')



    st.write('### Etat des lieux des données')
    st.dataframe(df_2013.info())
    st.markdown('Beaucoup de valeurs manquantes pour les variables HC (g/km) et HC+NOX (g/km)')
    
    if st.checkbox('Afficher les valeurs manquantes'):
        st.dataframe(df_2013.isna().sum())  

    st.write('#### Description statistique des données')
    st.dataframe(df_2013.describe())
    st.markdown('Les ordres de grandeurs sont très larges : il sera nécessaire de procéder à une standardisation des valeurs')

    
    st.markdown('#### Corrélation des variables selon Pearson')
    st.dataframe(df_quant.corr())
    st.markdown("Il apparaît qu'il y a une forte corrélation entre les différents types de consommation et les émissions de CO2")
    st.markdown("Il y a aussi une corrélation marquée entre les masses des véhicules (min et max) et les émissions de CO2")
    st.markdown("La variable HC+NOX (g/km) n'a pas été retenue car elle a trop de valeurs manquantes")






if page == pages[2]:
    st.write('## Analyse des données')


if page == pages[3]:
    st.write('## Viusalisation des données')
    
    fig = plt.figure(figsize = (10, 5))
    sns.catplot(x = 'Carburant', y = 'CO2 (g/km)', kind = 'box', height=5, aspect=2, data = df_2013)
    plt.title("Répartition de la distribution de l émission de CO2 en fonction du type d'énergie")
    st.pyplot(fig);
    
if page == pages[4]:
    
    st.write('## Machine Learning - Classification multiple')
    st.write('### Déterminer la classe du véhicule selon les normes européennes')
    st.markdown('Les véhicules doivent être classés dans les catégories de A à G (du moins polluant au plus polluant) selon leurs émissions de CO2')
    
    df= pd.read_csv('ML_C02.csv', index_col = 0)
    # Le document ML_02.csv est le DataFrame obtenu après nettoyage et analyse du DataFrame initial df_2013.csv
    # C'est à partir de ses données que nous effectuerons tout le travail de Machine Learning

    # Suppression des doublons
    df = df.drop_duplicates()

    # Mise à jour des variables
    # Suppression d'une variable en double (mq et Marque)
    df = df.drop('mq', axis = 1)

    # On renomme quelques variables
    variables = {'gamme2' : 'gamme',
                 'Puissance maximale (kW)' : 'puiss_max',
                 'Consommation urbaine (l/100km)' : 'conso_urb',
                 'HC+NOX (g/km)' : 'hcnox',
                 'masse vide euro min (kg)' : 'masse_ordma_min',
                 'norme EURO' : 'norme_EUR'}

    df = df.rename(variables, axis = 1)

    st.markdown('Le DataFrame comporte 5 variables numériques et 10 variables catégorielles')
    st.dataframe(df.info())
    
    # On sépare les variables numériques et catégorielles
    var_num = df.select_dtypes(exclude = 'object') # On récupère les variables numériques
    var_cat = df.select_dtypes(include = 'object') # On récupère les variables catégorielles

    # On récupère la variable cible
    target = df['cat_poll']
    var_cat = var_cat.drop('cat_poll', axis = 1)  # La variable cible est éliminée des variables catégorielles

    # Les variables catégorielles sont transformées en indicatrices
    var_cat_ind = pd.get_dummies(var_cat)

    # On récupère les variables explicatives
    feats = var_num.join(var_cat_ind)
    
    st.markdown('Distribution de la variable cible')
    st.dataframe(target.value_counts(normalize = True))
    st.markdown('On a une distribution légèrement déséquilibrée')
        

    # Commented out IPython magic to ensure Python compatibility.
    # Les différents types de modèles de Machine Learning
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.neighbors import KNeighborsClassifier

    # Les fonctions de paramétrage de la modélisation
    from sklearn.model_selection import train_test_split, KFold, cross_validate
    from sklearn.model_selection import GridSearchCV

    # Les fonctions de preprocessing
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder

    # Les algos de rééchantillonnage (Dataset déséquilibré)
    # from imblearn.over_sampling import RandomOverSampler, SMOTE
    # from imblearn.under_sampling import RandomUnderSampler,  ClusterCentroids

    # Les métriques
    # from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score

    # Les fonctions de sauvegarde et chargement de modèles
    from joblib import dump, load

    #la visualisation
    # %matplotlib inline
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    st.write('### **Cas 1 : Modélisation sans les variables qualitatives**')

    # On ne conserve que les variables quantitatives pour effectuer la modélisation
    feats_quant = feats[var_num.columns]

    feats_quant.head()

    # Séparation en données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(feats_quant, target, test_size = 0.25) # Question : Faut-il rajouter un paramètre random_state?

    # Les variables numériques doivent être standardisées
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    st.markdown('Nous procédons à une classification multiple. Nous avons donc choisi les classifieurs adaptés.')
    st.markdown('Nous en avons sélectionné 3 pour cette étude: SVC, KNN et Random Forest')
    selection_modele = st.selectbox("Choix du modèle", options = ["SVC", "KNN", "Random Forest"])
    
    def train_model(selection_modele):
        if selection_modele == "SVC":
            model = SVC(gamma = 'scale')
            
        if selection_modele == "KNN":
            model = KNeighborsClassifier()
            
        if selection_modele == "Random Forest":
            model = RandomForestClassifier()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)
        
        st.write('La matrice de confusion obtenue :')
        matrix = pd.crosstab(y_test, y_pred, rownames = ['Classes réelles'], colnames = ['Classes prédites'])
        st.dataframe(matrix)
        
        if selection_modele == 'Random Forest':
            feature_scores = pd.Series(model.feature_importances_, index=feats_quant.columns).sort_values(ascending=False)
            fig = plt.figure(figsize = (10, 5))
            sns.barplot(x=feature_scores, y=feature_scores.index)
            plt.xlabel('Feature Importance Score')
            plt.ylabel('Features')
            plt.title("Visualisation des features les plus importants")
            st.pyplot(fig);
            
        return score.round(2)
    
    st.write("L'accuracy du classifieur choisi est de", train_model(selection_modele))
    
    st.write("###### Voting Classifier")
    
    # Classifieur SVC
    clf_svc2 = SVC(gamma = 'scale')        # Instanciation du classifieur
    clf_svc2.fit(X_train, y_train)         # Entraînement du classifieur
    y_pred_svc2 = clf_svc2.predict(X_test) # Prédictions du classifieur


    # Classifieur KNN
    clf_knn2 = KNeighborsClassifier()       # Instanciation du classifieur
    clf_knn2.fit(X_train, y_train)          # Entraînement du classifieur
    y_pred_knn2 = clf_knn2.predict(X_test)  # Prédictions du classifieur
    
    # Classifieur Random Forest
    clf_rf2 = RandomForestClassifier()     # Instanciation du classifieur
    clf_rf2.fit(X_train, y_train)          # Entraînement du classifieur
    y_pred_rf2 = clf_rf2.predict(X_test)   # Prédictions du classifieur

    # Voting Classifier
    clf_vc2 = VotingClassifier([('rf', clf_rf2), ('svc', clf_svc2), ('knn', clf_knn2)], voting = 'hard')

    # Création du cross-validator
    cv3 = KFold(n_splits = 3) # Question : comment choisir les autres paramètres du CV? Comment définir le nombre optimal de splits?

    st.markdown('Validation croisée et évaluation des classifieurs')
    for clf, label in zip([clf_rf2, clf_svc2, clf_knn2, clf_vc2], ['Random Forest', 'SVC', 'KNN', 'Voting Classifier']):
        scores = cross_validate(clf, feats_quant, target, cv=cv3, scoring=['accuracy','f1_weighted'])
        st.write("[%s]: \n Accuracy: %0.2f (+/- %0.2f)" % (label, scores['test_accuracy'].mean(), scores['test_accuracy'].std()),
              "F1 score: %0.2f (+/- %0.2f)" % (scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std()))

    st.write('Suite au Voting Classifier, on constate que le modèle Random Forest est celui qui donne de bien meilleurs résultats')

    st.write("##### Optimisation des hyperparamètres du classifieur choisi")
    
    st.markdown('On cherche à optimiser les classifieurs en sélectionnant les meilleurs hyperparamètres')
    
    st.write("###### Classifieur SVC")
    st.markdown(" Nous avons fait fait varier les hyperparamètres selon le dictionnaire suivant:")  
    '''
    parametres_svc = {
            'C':[1,50,100,200],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma':[0.1, 0.5,1]
    }
    
    '''
    st.markdown("Après un GridSearchCV, les meilleurs paramètres pour le SVC sont  {'C': 100, 'gamma': 1, 'kernel': 'rbf'}")
    # On refait la classification avec les paramètres optimisés
    clf_svc_optim = SVC(C = 100, gamma = 1, kernel = 'rbf')
    clf_svc_optim.fit(X_train, y_train)
    y_pred_svc_optim = clf_svc_optim.predict(X_test)
    
    st.markdown('*Matrice de confusion du SVC optimisé*')
    matrix_optim = pd.crosstab(y_test, y_pred_svc_optim, rownames = ['Classes réelles'], colnames = ['Classes prédites SVC'])
    st.dataframe(matrix_optim)
    st.write("L'accuracy du classifieur SVC optimisé est de:",accuracy_score(y_test, y_pred_svc_optim).round(2))
    
    st.write("###### Classifieur KNN")
    st.markdown(" Nous avons fait fait varier les hyperparamètres suivants:")  
    '''
    parametres_knn = {
        'leaf_size':list(range(1,5)),
        'n_neighbors': list(range(1,10)),
        'p':[1,2],
        'metric': ['minkowski','manhattan','chebyshev']
    }
    
    '''
    st.markdown("Après un GridSearchCV, les meilleurs paramètres pour le KNN sont  {'leaf_size': 4, 'metric': 'minkowski', 'n_neighbors': 1, 'p': 1}")
    # On refait la classification avec les paramètres optimisés
    clf_knn_optim = KNeighborsClassifier(n_neighbors = 1, leaf_size = 4, metric = 'minkowski', p = 1)
    clf_knn_optim.fit(X_train, y_train)
    y_pred_knn_optim = clf_knn_optim.predict(X_test)
    
    st.markdown('*Matrice de confusion du KNN optimisé*')
    matrix_optim = pd.crosstab(y_test, y_pred_knn_optim, rownames = ['Classes réelles'], colnames = ['Classes prédites KNN'])
    st.dataframe(matrix_optim)
    st.write("L'accuracy du classifieur KNN optimisé est de:",accuracy_score(y_test, y_pred_knn_optim).round(2))
    
    st.write("###### Classifieur Random Forest")
    st.markdown(" Nous avons fait fait varier les hyperparamètres suivants:")  
    '''
    parametres_rf= {
        'n_estimators':[200,300,400,500,600,700],
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2']              
    }
    
    '''
    st.markdown("Après un GridSearchCV, les meilleurs paramètres pour le Random Forest sont {'criterion': 'entropy', 'max_features': 'log2', 'n_estimators': 600}")
    # On refait la classification avec les paramètres optimisés
    clf_rf_optim = RandomForestClassifier(n_estimators = 600, criterion = 'entropy', max_features = 'log2')
    clf_rf_optim.fit(X_train, y_train)
    y_pred_rf_optim = clf_rf_optim.predict(X_test)
    
    st.markdown('*Matrice de confusion du Random Forest optimisé*')
    matrix_optim = pd.crosstab(y_test, y_pred_rf_optim, rownames = ['Classes réelles'], colnames = ['Classes prédites Random Forest'])
    st.dataframe(matrix_optim)
    st.write("L'accuracy du classifieur Random Forest optimisé est de:",accuracy_score(y_test, y_pred_rf_optim).round(2))