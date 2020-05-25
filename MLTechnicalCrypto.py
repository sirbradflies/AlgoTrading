"""
Crypto trading bot using maching learning
@version: 0.1
"""

import clr
clr.AddReference("System")
clr.AddReference("QuantConnect.Algorithm")
clr.AddReference("QuantConnect.Common")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *

import numpy as np
from sklearn.neural_network import MLPRegressor


class NeuralNetworkAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2018, 7, 1)
        self.SetCash(1000000)
        self.SetBrokerageModel(BrokerageName.AlphaStreams)
        self.lookback = 63
        self.datapoints = 1000
        self.score = 0
        self.SetBrokerageModel(BrokerageName.GDAX, AccountType.Cash)
        self.resolution = Resolution.Daily
        self.ticker = "BTCUSD"
        self.symbol = self.AddCrypto(self.ticker, self.resolution, Market.GDAX)
        self.model = MLPRegressor()
        self.Train(self.DateRules.EveryDay(),
                   self.TimeRules.At(6, 0),
                   self.train_model)
        self.Schedule.On(self.DateRules.EveryDay(),
                         self.TimeRules.At(10,0,0),
                         self.predict_return)

    def train_model(self):
        """ Train model with new data and uses the last 20% for scoring """
        X, Y = self.get_data(n_features=self.lookback,
                             n_targets=1,
                             datapoints=self.datapoints)
        train_len = int(len(X) * 0.8)
        self.model.fit(X[:train_len], Y[:train_len])
        self.score = self.model.score(X[train_len:], Y[train_len:])
        self.Plot("Model", "Score", max(float(self.score),-1))
        self.Debug(f'{self.Time}\tPoints:{len(X)}\tScore:{self.score:.4f}')

    def predict_return(self):
        """ If model score is better than random predicts prices and trade """
        if self.IsMarketOpen(self.ticker) and self.score > 0:
            # Symbol tradeable and model score better than 0
            features, _ = self.get_data(n_features=self.lookback, n_targets=0)
            if self.model.predict(features) > 0:
                self.SetHoldings(self.ticker, 1)
            else:
                self.Liquidate(self.ticker)

    def get_data(self, n_features, n_targets, datapoints=1):
        """
        Extract features and targets for model training and prediction
        for the last (n_features + n_targets) periods
        """
        close = self.History([self.ticker],
                             n_features + n_targets + datapoints,
                             self.resolution)["close"]
        returns = None
        for lag in range(1, datapoints+1):
            price_changes = (close / close.shift(lag) - 1).fillna(0).values
            if len(price_changes)<datapoints+n_features+n_targets:
                pad_len = datapoints+n_features+n_targets-len(price_changes)
                price_changes = np.pad(price_changes, (pad_len,), mode="constant")
            price_changes = price_changes[lag: lag+n_features+n_targets]
            returns = price_changes if returns is None \
                else np.vstack((returns, price_changes))
        if len(returns.shape)<2:
            returns = np.transpose(returns.reshape(-1, 1))
        return returns[:, :n_features], returns[:, n_features:]