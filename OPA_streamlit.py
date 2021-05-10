# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:18:19 2021

@author: Pierre
"""
import streamlit as st
from streamlit_utils import feature_engineering
import streamlit.components.v1 as components
import pandas as pd
from backtest import run_backtest, run_goldencross

#Titre sidebar
st.sidebar.image('./images/logo.png')
st.sidebar.title("L'algorithme de trading automatique")
st.sidebar.header("Menu")
st.sidebar.text("")


#Création du menu sidebar
page = st.sidebar.radio(label = "",  options = ['1- Introduction', '2- Modelisation', '3- Backtesting', '4- Conclusions'])

#Paramétrage de la sidebar
st.sidebar.info("__Auteurs__: \n \n \n \n \n Arthur BEGOU [Linkedin](https://bit.ly/3nIL8Xs) \n \n \n \n Thierry MACÉ [Linkedin](https://www.linkedin.com/in/thierry-mac%C3%A9-852b52127/) \n \n \n \n  Data Analyst formation continue, ​​diplomé en Mai 2021, [DataScientest](https://datascientest.com/formation-data-analyst) \n \n \n \n \n _Sources données : [Kaggle.com](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)_")


#Page Introduction
if page == '1- Introduction':
    st.image('./images/introduction.png')
    #st.header("1- Introduction")
    st.markdown("""
                En finance, __la volatilité__ est l'ampleur des variations du cours d'un actif financier (actions, obligations, devises ou matières premières). Elle sert de paramètre de quantification du risque de rendement et de prix de l'actif. Lorsque la volatilité est élevée, la possibilité de gain est plus importante, mais le risque de perte l'est aussi.

L'objectif de ce projet est de developper un algorithme de trading automatique (achat/vente) en fonction de la valeur de la volatilité d'une action. Le machine learning (ML) sera appliqué pour prédire la volatilité future.
                
__Alors, prêt à nous confier votre épargne ?__""")

#if st.button('Hit me'):
    #page = 'Modelisation'

#Page Modélisation
           
if page == '2- Modelisation':
    st.image('./images/modelisation.png')
    #st.header("2- Modelisation")
    st.subheader("Objectif")
    st.markdown("""
                L'objectif de notre modélisation par Machine learning est de réaliser un modèle permettant de prédire au mieux la volatilité de notre action
""")

    st.subheader("Choix de la volatilité")
    st.image('./images/volatilite.png')
    st.markdown("""
                La volatilité quantifie la dispersion d'une série temporelle.
                Nous l'avons choisi comme variable cible car contrairement à la prédiction du prix de l'action, la prédiction de la volatilité est possible sur la base des données des jours précédents.   
                La volatilité est créée à partir du retour sur investissement
""")
    st.markdown("__Formule du calcul Retour sur Investissement__")
    st.latex(r'R_t = \frac{P_t - P_{t-1}}{P_{t-1}}')
    st.markdown("__Formule du calcul Volatilité__")
    st.latex(r'V_t(7) = \text{Std}(R_{t-1}, ...., R_{t-7})')
    st.markdown('*Volatilité au temps $t$ calculée sur les 7 derniers jours.')
    st.subheader("Feature engineering")
    st.markdown("""
                Pour améliorer les performances de notre modèle, nous créons une série de nouvelle variables
                """)
    st.image('./images/features.png')
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
    st.image('./images/strategie.png')

    st.markdown("""
                La stratégie repose sur la prediction de la volatilité effectué précédemment par Machine learning.

Si la volatilité prédite du jour est au dessus de notre seuil de volalitité (donc égale à 1), alors nous appliquons la strategie achat ou vente, tel que:

* Si le cours de l'action ('Open') d'aujourd'hui est supérieur au cours de la veille, nous vendons nos actions

* Dans le cas contraire, nous achetons

Notre idée sous-jacente est que dans une marché très volatile, une stratégie simple peut consister à continuellement parier contre le marché, c'est à dire vendre ses actions au moment où le cours monte et acheter quand le cours descend.
                """)
                
if page == '3- Backtesting':
    #st.header("3- Backtesting")
    st.image('./images/backtesting.png')
    st.subheader("Qu'est ce que le Backtesting ?")
    st.markdown("""
                Le backtesting ou test rétro-actif de validité consiste à tester la pertinence d'une modélisation ou d'une stratégie en s'appuyant sur un large ensemble de données historiques réelles. 
                """)
    st.subheader("Résultats du Backtesting")
    stock_choice = st.selectbox('Choisissez une action', options=['IBM (modèle entrainé)','Google (modèle non entrainé)','Tesla (modèle non entrainé)'])
    cash_choice = st.number_input('Entrez un montant à investir')
    st.markdown(" ")

    
    if st.button('Lancer le Backtesting'):
      
      if cash_choice <= 0:
          st.markdown("Veuillez entrer un montant supérieur à 0")  
          
      if cash_choice >0:
          st.markdown(" ")
          st.markdown(" ")
          df = pd.read_csv("Data/Stocks/" + stock_choice + ".txt" , index_col = 'Date')
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
          
          _, output_golden_cross = run_goldencross(X_test, cash_choice)
          
          earnings = int(cash_choice*(1 + 0.01*output['Return [%]']) - cash_choice)
          
          st.write("Grâce à notre algorithme et votre investissement de",
                   cash_choice,"€, vous auriez gagné",
                   str(earnings),
                   "€")
          if earnings > 0:
              st.balloons()
          
          st.write("Grâce à la [Golden Cross](https://www.tradingwithrayner.com/golden-cross-trading-strategy/#:~:text=A%20Golden%20Cross%20occurs%20when,the%20200%2Dday%20moving%20average), une autre stratégie célèbre de trading , vous auriez gagné",
                      str(int(cash_choice*(1 + 0.01*output_golden_cross['Return [%]']) - cash_choice)),
                      "__€__")
    
          st.write("Si vous aviez acheté l'action et rien fait, vous auriez gagné",
                      str(int(cash_choice*(1 + 0.01*output_golden_cross['Buy & Hold Return [%]']) - cash_choice)),
                      "__€__")
          
          st.markdown(" ")

          
          st.bokeh_chart(fig)
          st.write(output)
      
if page == '4- Conclusion':
    #st.header("4- Conclusions")
    st.subheader("Récapitulatif")
    st.image('./images/recapitulatif.png')

    st.subheader("1- Améliorations des données ")
    st.markdown("__Exploiter une autre valeur cible que la volatilité__")

     
    st.subheader("2- Améliorations du machine learning ")
    st.markdown("__Entrainer les données sur l'ensemble du dataset__")
    st.markdown("__Utiliser plusieurs sources pour prédire la volatilité__")
   
    
    st.subheader("3- Améliorations de la stratégie")
    st.markdown("__Changement du seuil de la volatilité__")

    st.markdown("__Imaginer une stratégie totalement différente __")
   
    st.markdown("__Combiner plusieurs stratégies et notamment avec des stratégies d'analyse __")
   
    st.markdown("__Seuils de déclenchement__")
   




    
    
