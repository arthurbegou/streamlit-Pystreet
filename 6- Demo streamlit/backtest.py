# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:27:19 2021

@author: Pierre
"""

from backtesting import Backtest, Strategy
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
    
    
def run_backtest(X_test, cash_choice):
    
    bt = Backtest(X_test, Volatility_Strategy, commission=0.00,cash=cash_choice,
              exclusive_orders=True)
    output = bt.run()
    fig = bt.plot(open_browser = False)
    
    return fig, pd.Series(bt._results)
        
    