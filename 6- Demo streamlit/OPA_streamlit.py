# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:18:19 2021

@author: Pierre
"""
import streamlit as st
from streamlit_utils import feature_engineering
import streamlit.components.v1 as components
import pandas as pd
from backtest import run_backtest

#Titre sidebar
st.sidebar.title("Pystreet - L'algorithme de trading automatique")
st.sidebar.header("Menu")
st.sidebar.text("")


#Création du menu sidebar
page = st.sidebar.radio(label = "",  options = ['Introduction', 'Modelisation', 'Backtesting', 'Conclusion'])

#Paramétrage de la sidebar
st.sidebar.info("__Auteurs__: \n \n \n \n \n Arthur BEGOU [Linkedin](https://bit.ly/3nIL8Xs) \n \n \n \n Thierry MACÉ [Linkedin](https://www.linkedin.com/in/thierry-mac%C3%A9-852b52127/) \n \n \n \n  Data Analyst formation continue, ​​diplomé en Mai 2021, [DataScientest](https://datascientest.com/formation-data-analyst) \n \n \n \n \n _Sources données : [Kaggle.com](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)_")


#Page Introduction
if page == 'Introduction':
    st.header("Introduction")

    st.markdown("""
                En finance, __la volatilité__ est l'ampleur des variations du cours d'un actif financier (actions, obligations, devises ou matières premières). Elle sert de paramètre de quantification du risque de rendement et de prix de l'actif. Lorsque la volatilité est élevée, la possibilité de gain est plus importante, mais le risque de perte l'est aussi.

L'objectif de ce projet est de developper un algorithme de trading automatique (achat/vente) en fonction de la valeur de la volatilité d'une action. Le machine learning (ML) sera appliqué pour prédire la volatilité future.
                
__Alors, prêt à nous confier votre épargne ?__""")

#if st.button('Hit me'):
    #page = 'Modelisation'

#Page Modélisation
           
if page == 'Modelisation':
    st.header("Modelisation")
    st.subheader("Modelisation")
    st.markdown("""
                L'objectif de notre modélisation par Machine learning est de réaliser un modèle permettant de prédire au mieux la volatilité de notre action

Dans cette partie, nous détaillerons toutes les transformations appliquées à nos données jusqu'à l'entrainement de notre modèle

Tout d'abord, nous commençons par créer notre variable cible Volatility_next_week à partir de la variable Return

Nous avons choisi d'estimer la volatilité car contrairement à la prédiction du prix de l'action, la prédiction de la volatilité est possible sur la base des données des jours précédents.                
""")

    st.subheader("Formule du calcul Retour sur Investissement")
    st.latex(r'R_t = \frac{P_t - P_{t-1}}{P_{t-1}}')
    st.subheader("Formule du calcul Volatilité")
    st.latex(r'V_t(7) = \text{Std}(R_{t-1}, ...., R_{t-7})')
    st.markdown('*Volatilité au temps $t$ calculée sur les 7 derniers jours.')
    st.subheader("Feature engineering")
    st.markdown("""
                Pour améliorer les performances de notre modèle, nous créons une série de nouvelle variables p
                """)
    st.code("""
# Ajout de l'historique des prix des 10 jours précédents (variable 'Open_t-i')
N = 10
for i in range(1, N+1):
    df['Open_t-'+ str(i)] = df['Open'].shift(i)

# Aout des moyennes mobiles des prix à 7, 14 et 28 jours (variable 'MA(i)')
MA = [5, 10, 20]
for i in MA:
    df['MA(' + str(i) + ')'] = df['Open'].rolling(i).mean()


# Ajout du retour sur investissement, indispensable au calcul de la volatilité (variable 'R_t(i)')
R = [1, 5, 10, 20]
for i in R:
    df['R_t(' + str(i) + ')'] = df['Open'].pct_change(i)

# Ajout de la volatilité sur 3 intervalles de temps (variable 'V_t(i)'
V = [5, 10, 20]
for i in V : 
    df['V_t(' + str(i) + ')'] = df['R_t(1)'].rolling(i).std()

# Aout de l'historique du Volume sur les 10 jours précédentes (variable 'Volume_t-i')
N = 10
for i in range(1, N+1):
    df['Volume_t-'+ str(i)] = df['Volume'].shift(i)

# Création de la Variable Cible
df['target'] = df['V_t(5)'].shift(-5)
df['target'] = df['target'].apply(lambda x : 1 if x >= 0.015 else 0)
        """)

    st.subheader("Recherche d’hyperparamètres et sélection du modèle optimal")
    st.markdown("""
                Les meilleurs paramètres trouvés gràce au gridsearch appliqué au modèle GradientBoostingClassifier sont {'max_depth': 1, 'min_samples_split': 2, 'n_estimators': 30}
                """)
    st.subheader("Définition de la stratégie de trading")
    st.markdown("""
                La stratégie repose sur la prediction de la volatilité effectué précédemment par Machine learning.

Si la volatilité prédite du jour est au dessus de notre seuil de volalitité (donc égale à 1), alors nous appliquons la strategie achat ou vente, tel que:

si le cours de l'action ('Open') d'aujourd'hui est supérieur au cours de la veille, nous vendons nos actions
dans le cas contraire, nous achetons
Notre idée sous-jacente est que dans une marché très volatile, une stratégie simple peut consister à continuellement parier contre le marché, c'est à dire vendre ses actions au moment où le cours monte et acheter quand le cours descend.
                """)
                
if page == 'Backtesting':
    st.header("Backtesting")
    st.subheader("Introduction au Backtesting")
    st.markdown("""
                Wikipédia donne la définition suivante du Backtesting : "Le backtesting ou test rétro-actif de validité consiste à tester la pertinence d'une modélisation ou d'une stratégie en s'appuyant sur un large ensemble de données historiques réelles." Appliqué aux marchés des capitaux, "le backtesting est un type spécifique de tests historiques qui détermine la performance d'une stratégie financière, si elle avait été effectivement utilisée pendant des périodes passées et dans les mêmes conditions du marché." Le principal avantage de ce type de tests "réside dans la compréhension de la vulnérabilité d'une stratégie grâce à son application à des conditions réelles effectivement rencontrées dans le passé."
                """)
    st.subheader("Résultats du Backtesting")
    cash_choice = st.number_input('Entrez un montant à investir')
    if st.button('Lancer le Backtesting'):
      
      df = pd.read_csv("Data/Stocks/ibm.us.txt", index_col = 'Date')
      df = df.drop('OpenInt', axis = 1)
      # création d'un index au format datetime pour le backtesting
      df.index = pd.to_datetime(df.index)
      df = feature_engineering(df)
      X, y = df.drop("target", axis = 1), df['target']
      X_train = X.iloc[:-len(X)//10]
      y_train = y.iloc[:-len(X)//10]
      
      X_test = X.iloc[-len(X)//10:]
      y_test = y.iloc[-len(X)//10:]
      
      fig, output = run_backtest(X_test, cash_choice)
      st.bokeh_chart(fig)
      st.write(output)
      st.write("__Gràce à notre algorithme, et votre investissement de__", cash_choice, "__vous auriez gagné__", str(cash_choice*0.46931), "__€__")
      st.balloons()

if page == 'Conclusion':
    st.title("Texte de la conclusion")
    
    
