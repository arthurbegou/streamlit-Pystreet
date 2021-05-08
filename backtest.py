# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:27:19 2021

@author: Pierre
"""

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG

from streamlit_utils import make_pipeline
import pandas as pd    

class Volatility_Strategy(Strategy):
    def init(self):
        self.pipeline = make_pipeline()
        return
    
    def next(self):
        # On récupère les features de aÇourd'hui
        jour_en_cours = self.data.df.iloc[-1:]
        
        # On fait la prédiction
        volatilite_estimee = self.pipeline.predict(jour_en_cours)
        
        if volatilite_estimee == 1:
            # Si le prix a monté
            if jour_en_cours['Open'][0] > jour_en_cours['Open_t-1'][0]:
                # On vend
                self.sell()
                return
            else:
                # Sinon on achète
                self.buy()
                return
        return
    
    
class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()
    
    
def run_backtest(X_test, cash_choice):
    
    bt = Backtest(X_test, Volatility_Strategy, commission=0.00,cash=cash_choice,
              exclusive_orders=True)
    output = bt.run()
    fig = bt.plot(open_browser = False)
    
    return fig, pd.Series(bt._results)
        
    



def run_goldencross(X_test, cash_choice):

    bt = Backtest(X_test, SmaCross,
                  cash=cash_choice, commission=0,
                  exclusive_orders=True)
    
    output = bt.run()
    fig = bt.plot(open_browser = False)
    
    return fig, pd.Series(bt._results)