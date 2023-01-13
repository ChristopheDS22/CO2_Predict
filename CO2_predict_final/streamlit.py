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
pages = ['Introduction','Exploration et analyse des données', 
         'Modélisation : Regréssion', 'Modélisation : Classification multiple', 'Interprétabilité', 
         'Conclusion']

page = st.sidebar.radio('Aller vers', pages)

# Emissions de polluants, CO2 et caractéristiques des véhicules
# commercialisés en France en 2013
df_2013 = pd.read_csv('data_2013.csv' , sep = ';', encoding='unicode_escape')



if page == pages[0]:
    st.write('## Introduction au projet')
    
    st.write('### Visualisation du dataset')
    st.dataframe(df_2013.head())
    
    st.markdown('Identifier les véhicules qui émettent le plus de CO2 est important pour identifier les caractéristiques techniques qui jouent un rôle dans la pollution.')
    st.markdown(' Prédire à l’avance cette pollution permet de prévenir dans le cas de l’apparition de nouveaux types de véhicules (nouvelles séries de voitures par exemple)')
    st.markdown('Le projet est effectué à partir du dataset regroupant les émissions de CO2 et polluants des véhicules commercialisées en France en 2013')
    st.markdown('[Source du dataset]( https://www.ecologie.gouv.fr/normes-euros-demissions-polluants-vehicules-lourds-vehicules-propres)')
    
if page == pages[1]:
    st.write('## Exploration des données')
    
    st.markdown('La variable cible est CO2 (g/km). Elle peut être expliquée en tant que variable continue ou discrète')
    #st.markdown('Il y a ', df_2013.duplicated().sum(), 'doublons dans le dataset') # On checke les doublons
    
    #on renomme les variables pour plus de facilité d'utilisation
    variables = {'Modèle dossier' : 'modele_dossier',
             'Modèle UTAC':'modele_UTAC',
             'Désignation commerciale':'design_comm',
             'Type Variante Version (TVV)':'TVV',
             'Puissance administrative':'puissance_adm',
             'Puissance maximale (kW)' : 'puiss_max',
             'Boîte de vitesse':'boite0',
             'Consommation urbaine (l/100km)':'conso_urb',
             'Consommation extra-urbaine (l/100km)':'conso_extra_urb',
             'Consommation mixte (l/100km)':'conso_mixte',
             'CO2 (g/km)':'CO2',
             'CO type I (g/km)':'CO2_type_1',
             'HC (g/km)':'HC',
             'NOX (g/km)':'NOX',
             'HC+NOX (g/km)':'hcnox',
             'Particules (g/km)':'particules',
             'masse vide euro min (kg)': 'masse_ordma_min',
             'masse vide euro max (kg)': 'masse_ordma_max',
             'Champ V9':'champ_V9',
             'Date de mise à jour':'date_maj'
            }


    df_2013 = df_2013.rename(variables, axis = 1)
    
    # Suppression des doublons
    df_2013 = df_2013.drop_duplicates()
    df_2013.duplicated().sum()
    
    if st.checkbox('Informations détaillées des variables'):
        st.dataframe(df_2013.info())
    
    st.write('### Variables quantitatives')
    st.markdown('''Il y a 13 variables quantitatives : \n
                - puissance adm : puissance administrative en kW
                - puiss_max : puissance maximale en kW
                - conso_urb : consommation urbaine de carburant (en l/100km)
                - conso_extra_urb : consommation extra urbaine de carburant (en l/100km)
                - conso_mixte : consommation mixte de carburant (en l/100km)
                - CO2 : l'émission de CO2 (en g/km), (ça sera la variable à expliquer)
                - CO2_type_1 : le résultat d’essai de CO type I
                - HC : les résultats d’essai HC
                - NOX : les résultats d’essai NOx
                - hcnox : les résultats d’essai HC+NOX
                - particules : le résultat d’essai de particules
                - masse_ordma_min : la masse en ordre de marche mini
                - masse_ordma_max : la masse en ordre de marche maxi''')                
                
    st.markdown('Récapitulatif statistique des variables quantitatives')
    st.write(df_2013.describe())
    
    
    st.markdown('''Beaucoup de valeurs manquantes pour les variables HC et hcnox. 
                Les ordres de grandeurs sont très larges : il sera nécessaire de procéder à une standardisation des valeurs.''')
    
    st.write('### Variables qualitatives')
    st.markdown('''Il y a 13 variables qualitatives : \n
                - Marque : marque du véhicule
                - modele_dossier : modèle
                - modèle_UTAC : modèle UTAC (nécessité de comprendre la différence avec modèle dossier)
                - designation : designation commerciale
                - CNIT : Code National d'identification du type
                - TVV : Type Variante Version ou type Mines
                - Carburant : type de carburant
                - Hybride : information permettant d'identifier les véhicules hybrides
                - boite0: type de boîte de vitesse et le nombre de rapports
                - champ_V9 : champ V9 du vertificat d'immatriculation qui contient la norme euro
                - Carrosserie : Carrosserie
                - date_maj : la date de la dernière mise à jour
                - gamme : gamme''')
                
    st.write('#### Corrélation des variables selon Pearson')
    st.write(df_2013.corr())
    st.markdown("Il apparaît qu'il y a une forte corrélation entre les différents types de consommation et les émissions de CO2")
    st.markdown("Il y a aussi une corrélation marquée entre les masses des véhicules (min et max) et les émissions de CO2")

    # Variable 'Marque'
    # Répartition des voitures par marques
    df_2013.Marque.value_counts()
    
    #Variables 'Modèle dossier' et 'Modèle UTAC'
    # Les variables modele_dossier et modele_UTAC semblent assez similaires : quelle est la différence?
    pd.crosstab(df_2013.modele_dossier,df_2013.modele_UTAC)

    test_mod=df_2013[-(df_2013.modele_dossier==df_2013.modele_UTAC)]
    pd.crosstab(test_mod.modele_dossier,test_mod.modele_UTAC)

    # Variables 'designation commerciale', 'CNIT', et 'TVV'
    # On créé des nouvelles variables qu'on enrichira des valeurs ci-desous
    df_2013.insert(5,'cat','T')
    df_2013.insert(6,'mq','T')
    df_2013.insert(7,'genre','T')
    
    #la variable CNIT génère beaucoup de doublons, à quoi correspond-t-elle?
    #les 3 premiers caractères représentent la catégorie
    df_2013.cat=df_2013.CNIT.str[:3]

    #les 3 suivants sont la marque
    df_2013.mq=df_2013.CNIT.str[3:6]

    #les 2 suivants le genre (VP dans la majeure partie des cas)
    df_2013.genre=df_2013.CNIT.str[6:8]


    #Variable 'Carburant'
    # Harmonisation des nomenclatures de carburants
    df_2013['Carburant'] = df_2013['Carburant'].replace(to_replace = ['GN/ES', 'GP/ES'], value = ['ES/GN', 'ES/GP'])
    
    #Variable Hybride
    # Répartition des véhicules hybrides
    df_2013.Hybride.value_counts()
    
    # Variable 'boite0'
    # Il faut éclater la variable boite0 en 2 : type de boîte d'un côté et nombre de rapports de l'autre
    # On créé des nouvelles variables qu'on enrichira des valeurs ci-desous

    df_2013[['boite', 'rapport']]=df_2013.boite0.str.split(expand=True)
    
    # Suppression des boites 'S', 'V' et 'N' (boîtes indéfinissables)
    df_2013 = df_2013[df_2013['boite']!='S']
    df_2013 = df_2013[df_2013['boite']!='V']
    df_2013 = df_2013[df_2013['boite']!='N']

    # La modalité D semble être automatique
    df_2013['boite'].replace(to_replace = ['D'], value = 'A', inplace = True)
    
    # Variable V9
    # Exploitation de la variable Champ V9 : permet de ressortir la norme EURO
    df_2013['champ_V9'].replace(to_replace = [np.nan], value = [df_2013['champ_V9'].mode()], inplace = True) # Remplacement des valeurs manquantes par le mode
    df_2013['norme_EURO'] = df_2013['champ_V9'].apply(lambda x: x[-5:])

    # Variable 'Gamme du véhicule'
    # On modifie les labels qui ne ne sont pas corrects et on les harmonise
    def change_gamme(x):
        if x in ['MOY-INFERIEURE','MOY-INFER','MOY-INF']:
            return('MOY-INFERIEURE')
        if x in ['MOY-SUPER']:
            return('MOY-SUPERIEURE')
        else : return(x)


    df_2013['gamme2']=df_2013.gamme.apply(change_gamme)

    
    st.write('### Valeurs manquantes')
    st.markdown('Beaucoup de valeurs manquantes pour les variables HC et hcnox')
    if st.checkbox('Afficher les valeurs manquantes'):
        st.dataframe(df_2013.isna().sum()) 

    st.markdown('Les valeurs manquantes des carburants "EL" sont remplacées par 0:')
    df_2013['CO2'] = df_2013['CO2'].fillna(0)
    
    # Les valeurs manquantes de consommation semblent coller avec celles des véhicules électriques
    df_2013['conso_urb'] = df_2013['conso_urb'].fillna(0)
    df_2013['conso_extra_urb'] = df_2013['conso_extra_urb'].fillna(0)
    df_2013['conso_mixte'] = df_2013['conso_mixte'].fillna(0)

    st.markdown('Les valeurs manquantes des variables NOX, hcnox et particules sont remplacées par des moyennes')
    df_2013['NOX'] = df_2013['NOX'].fillna(df_2013['NOX'].mean())
    df_2013['hcnox'] = df_2013['hcnox'].fillna(df_2013['hcnox'].mean())
    df_2013['particules'] = df_2013['particules'].fillna(df_2013['particules'].mean())
    df_2013['CO2_type_1'] = df_2013['CO2_type_1'].fillna(df_2013['CO2_type_1'].mean())

    st.markdown('La variable HC a près de 77% de valeurs manquantes. Elle ne sera pas retenue.')
    df_2013 = df_2013.drop(['HC'], axis = 1)


    
    # Création d'une variable Cat_CO2 pour la classification des véhicules
    label = pd.cut(df_2013.CO2,
                   bins = [-1,100,120,140,160,200,249,600],
                   labels = ['A','B','C','D','E', 'F','G'])
    
    df_2013['Cat_CO2'] = label 

    st.write('### Sélection des variables utiles')
    st.markdown('On ne conserve que les variables qui vont être utiles pour la modélisation')


    df = df_2013.drop(columns = ['modele_UTAC', 'modele_dossier', 'design_comm', 'CNIT', "TVV", 'puissance_adm', 'Hybride',
                              'boite0', 'conso_urb', 'conso_extra_urb', 'conso_mixte', 'CO2_type_1', 'NOX', 'hcnox', 'norme_EURO',
                              'particules','masse_ordma_max', 'champ_V9', 'date_maj','gamme', 'rapport','cat', 'mq', 'genre'])

    # et on supprime les doublons
    df = df.drop_duplicates()

    st.write('Dataset après préprocessing')
    st.write(df)
    
    df.to_csv('Model_C02.csv')
    # Le df ainsi obtenu pour la modélisation est enregistré dans un nouveau document ML_CO2.csv
    # Il sera utilisé pour la modélisation
    
    #------------------------------
    #partie streamlit des représentations
    #------------------------------

    st.write('### Variables à expliquer')


    fig=px.box(df_2013['CO2'],x='CO2',title='Variable CO2 continue',notched=True)
    st.plotly_chart(fig)

    fig2= px.histogram(df_2013,title='Variable CO2 discrète',x='Cat_CO2',category_orders=dict(Cat=['A','B','C','D','E','F','G']))
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

    
    
    st.write('## Visualisation des données')
    
    fig = plt.figure(figsize = (10, 5))
    sns.catplot(x = 'Carburant', y = 'CO2', kind = 'box', height=5, aspect=2, data = df_2013)
    plt.title("Répartition de la distribution de l émission de CO2 en fonction du type d'énergie")
    st.pyplot(fig);

if page == pages[2]:
    st.write('## Modélisation : Regréssion')
    st.write('### Déterminer les émissions de CO2 selon les caractéristiques techniques du véhicule')
    
if page == pages[3]:
    
    st.write('## Modélisation : Classification multiple')
    st.write('### Déterminer la classe du véhicule selon les normes européennes')
    st.markdown('Les véhicules doivent être classés dans les catégories de A à G (du moins polluant au plus polluant) selon leurs émissions de CO2')
    
    df = pd.read_csv('Model_C02.csv', index_col = 0)
    st.markdown('Le DataFrame utilisé pour la classification multiple comporte : \n')
        
    # On sépare les variables numériques et catégorielles
    var_num = df.select_dtypes(exclude = 'object') # On récupère les variables numériques
    var_cat = df.select_dtypes(include = 'object') # On récupère les variables catégorielles

    # On récupère les variables cibles
    target_reg = df['CO2']                         # Variable cible pour la regression
    target_class = df['Cat_CO2']                   # Variable cible pour la classification

    var_num = var_num.drop(['Cat_CO2', 'CO2'], axis = 1)  # Les variables cibles sont éliminées des variables numériques
    
    st.markdown('Les variables catégorielles :\n')
    st.write(var_cat.head())
    st.write(df.columns)

    st.markdown('Les variables numériques :\n')
    st.write(var_num.head())

    # Les variables catégorielles sont transformées en indicatrices
    var_cat_ind = pd.get_dummies(var_cat)

    # On récupère les variables explicatives
    feats = var_num.join(var_cat_ind)
    
    st.markdown('Distribution de la variable cible')
    st.dataframe(target_class.value_counts(normalize = True))
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
         
    # Les métriques
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate

    # Les fonctions de sauvegarde et chargement de modèles
    from joblib import dump, load

    #la visualisation
    # %matplotlib inline
    import matplotlib.pyplot as plt
    import seaborn as sns    

    # Séparation en données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(feats, target_class, test_size = 0.25)

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