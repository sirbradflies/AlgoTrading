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
from sklearn.model_selection import train_test_split as split


class NeuralNetworkAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2008, 1, 1)
        self.SetEndDate(2010, 12, 31)
        self.stocks = 10
        self.lookback = 255
        self.leverage = 0.5
        self.model = MLPRegressor(hidden_layer_sizes=(128, 128),
                                  warm_start=True,
                                  validation_fraction=0.2)
        self.AddUniverse(self.Universe.Index.QC500)
        self.AddEquity("SPY", Resolution.Minute)  # TODO: Remove?
        self.Schedule.On(self.DateRules.EveryDay(),
                         self.TimeRules.AfterMarketOpen("SPY", 1),
                         self.train_model)
        self.X_train, self.Y_train, self.X_test, self.Y_test = [], [], [], []

    def train_model(self):
        train_stocks, test_stocks = split(list(self.ActiveSecurities.Keys),
                                          test_size=0.2)
        self.X_train, self.Y_train = self.get_datapoints(self.X_train, self.Y_train,
                                                         train_stocks)
        self.X_test, self.Y_test = self.get_datapoints(self.X_test, self.Y_test,
                                                       test_stocks)

        self.model.fit(np.array(self.X_train), np.array(self.Y_train))
        self.Debug(f'Datapoints: {len(self.X_train)}\t'
                   f'Epochs: {self.model.n_iter_}\t'
                   f'Val. Score: {self.model.score(self.X_test, self.Y_test):.4f}')
        #self.ObjectStore.SaveBytes("model", bytearray(self.model))  # TODO: Save model after training
        #self.model = self.ObjectStore.ReadBytes("model")

        pred_returns = {}
        for symbol in self.ActiveSecurities.Keys:
            x_pred = np.array([self.get_features(symbol, ago=0, size=self.lookback)])
            pred_returns[symbol] = self.model.predict(x_pred)

        self.trade(pred_returns)

    def get_datapoints(self, X, Y, stocks, max_datapoints=5000):
        """ Update train/test datapoints capping array maximum length """
        for symbol in stocks:
            X += [self.get_features(symbol, ago=1, size=self.lookback)]
            Y += self.get_features(symbol, ago=0, size=1)
        X = X[-max_datapoints:] if len(X) > max_datapoints else X
        Y = Y[-max_datapoints:] if len(Y) > max_datapoints else Y
        return X, Y

    def get_features(self, symbol, ago=0, size=1):   # TODO: Add more features
        """ Extract features for model training and prediction """
        close = self.History(symbol, size + ago + 1, Resolution.Daily)['close']
        returns = list((close / close.shift(1) - 1).dropna().head(size))
        return ([0]*size+returns)[-size:]  # 0 padding for missing values

    def trade(self, pred_returns):
        """  Rank returns and select the top for long and bottom for short """
        buy_stocks = self.rank_stocks(pred_returns, long=True).head(self.stocks)
        short_stocks = self.rank_stocks(pred_returns, long=False).head(self.stocks)
        liquidate_stocks = set(self.Securities.Keys)-set(buy_stocks)-set(short_stocks)
        self.Log(f'Buy stocks: {buy_stocks}\nSell stocks: {short_stocks}')

        for symbol in liquidate_stocks:
            self.Liquidate(symbol)
        for symbol in buy_stocks.index:
            self.SetHoldings(symbol, self.leverage / self.stocks)
        for symbol in short_stocks.index:
            self.SetHoldings(symbol, -self.leverage / self.stocks)

    def rank_stocks(self, pred_returns, long=True, commission=0.01, contingency=4):
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
        ).sort_values('return', ascending=not long)