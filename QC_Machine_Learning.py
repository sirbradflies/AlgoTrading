import clr

clr.AddReference("System")
clr.AddReference("QuantConnect.Algorithm")
clr.AddReference("QuantConnect.Common")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


class NeuralNetworkAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2008, 1, 1)  # TODO: Make parametric
        self.SetEndDate(2010, 12, 31)
        self.stocks = 10
        self.lookback = 20
        self.leverage = 0.5
        self.model = MLPRegressor(hidden_layer_sizes=(128,),
                                  warm_start=True)
        self.resolution = Resolution.Daily
        self.AddUniverse(self.Universe.Index.QC500)
        self.AddEquity("SPY", Resolution.Minute)  # TODO: Remove?
        self.Schedule.On(self.DateRules.EveryDay(),
                         self.TimeRules.AfterMarketOpen("SPY", 1),
                         self.train_model)

    def train_model(self):
        X, Y = [], []
        for symbol in self.ActiveSecurities.Keys:
            X += [self.get_features(symbol, ago=1, size=self.lookback)]
            Y += self.get_features(symbol, ago=0, size=1)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        self.model.fit(np.array(X_train), np.array(Y_train))
        val_score = self.model.score(X_test, Y_test)
        self.Debug(f'Datapoints: {len(X_train)}\t'
                   f'Epochs: {self.model.n_iter_}\t'
                   f'Val. Score: {val_score:.4f}')
        # TODO: Save model after training

        pred_returns = {}
        for symbol in self.ActiveSecurities.Keys:
            x_pred = np.array([self.get_features(symbol, ago=0, size=self.lookback)])
            pred_returns[symbol] = self.model.predict(x_pred)

        self.trade(pred_returns)

    def get_features(self, symbol, ago=0, size=1):   # TODO: Add more features
        """ Extract features for model training and prediction """
        history = self.History(symbol, size + ago + 1, self.resolution)
        history['returns'] = history['close'] / history['close'].shift(1) - 1
        return list(history['returns'].dropna().values[:size])

    def trade(self, pred_returns):
        """  Rank returns and select the top for long and bottom for short """
        buy_stocks = self.get_top_stocks(pred_returns,
                                         long=True,
                                         stock_nr=self.stocks)
        sell_stocks = self.get_top_stocks(pred_returns,
                                          long=False,
                                          stock_nr=self.stocks)
        self.Log(f'Buy stocks: {buy_stocks}\nSell stocks: {sell_stocks}')

        for symbol in self.Securities.Keys:  # TODO: Print portfolio and order status
            if symbol in buy_stocks.index:  # TODO: Check that SetHoldings does nothing if position is already held
                self.SetHoldings(symbol, self.leverage / self.stocks)
            elif symbol in sell_stocks.index:
                self.SetHoldings(symbol, -self.leverage / self.stocks)
            else:
                self.Liquidate(symbol)

    def get_top_stocks(self, pred_returns, long=True, stock_nr=1, commission=0.01, contingency=4):
        """
        Calculate best stocks to long or short accounting
        for commissions and current portfolio positions
        """
        sign = -1 if long else +1
        ranking = {}
        for symbol, returns in pred_returns.items():
            position = self.Portfolio[symbol]
            if (long and position.IsLong) or (not long and position.IsShort):  # Symbol already in the position desired
                ranking[symbol] = returns  # No commissions
            elif (not long and position.IsLong) or (long and position.IsShort):  # Symbol in the opposite position
                ranking[symbol] = returns + sign * 2 * commission * contingency  # Twice the commissions
            else:  # Symbol not in the portfolio
                ranking[symbol] = returns + sign * commission * contingency
        return pd.DataFrame.from_dict(
            ranking, orient='index', columns=['return']
        ).sort_values('return', ascending=not long).head(stock_nr)

    def OnSecuritiesChanged(self, changes):
        self.Log(f"OnSecuritiesChanged({self.UtcTime}): {changes}")