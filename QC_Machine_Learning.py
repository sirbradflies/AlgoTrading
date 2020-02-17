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


class KerasNeuralNetworkAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2008, 1, 1)  # TODO: Make parametric
        self.SetEndDate(2010, 12, 31)
        self.long_stocks = 5
        self.short_stocks = 5
        self.SetCash(100000)
        self.lookback = 10
        self.portfolio_pct = 0.5
        self.commission_pct = 1.0
        self.model = MLPRegressor(
            hidden_layer_sizes=(64,), warm_start=False, early_stopping=True
        )
        self.resolution = Resolution.Daily
        self.AddUniverse(self.Universe.Index.QC500)
        self.AddEquity("SPY", Resolution.Minute)  # TODO: Remove?
        self.Schedule.On(self.DateRules.EveryDay(),
                         self.TimeRules.AfterMarketOpen("SPY", 1),
                         self.train_model)
        self.X, self.Y = [], []

    def train_model(self):
        for symbol in self.Securities.Keys: # TODO: Add more features
            if self.has_features(symbol, ago=1, size=self.lookback):
                self.X += [self.get_features(symbol, ago=1, size=self.lookback)]
                self.Y += self.get_features(symbol, ago=0, size=1)

        self.model.fit(np.array(self.X), np.array(self.Y))
        self.Debug(f'Datapoints: {self.X.shape[0]}')
        self.Debug(f'Training Epochs: {self.model.n_iter_}')
        self.Debug(f'Score: {self.model.validation_scores_[-1]:.5f}')
        self.Plot('Model Score', 'Score', self.model.validation_scores_[-1])
        # TODO: Save model after training

        pred_returns = {}
        for symbol in self.Securities.Keys:
            if self.has_features(symbol, ago=0, size=self.lookback):
                x_pred = np.array([self.get_features(symbol, ago=0, size=self.lookback)])
                pred_returns[symbol] = self.model.predict(x_pred)

        self.trade(pred_returns)

    def has_features(self, symbol, ago=0, size=1):
        ticks_needed = size + ago + 1
        history = self.History(symbol, ticks_needed, self.resolution)
        return ('close' in history) and (len(history) == ticks_needed)

    def get_features(self, symbol, ago=0, size=1):
        history = self.History(symbol, size + ago + 1, self.resolution)
        history['returns'] = history['close'] / history['close'].shift(1) - 1
        return list(history['returns'].dropna().values[:size])

    def trade(self, pred_returns):
        """  Rank returns and select the top for long and bottom for short """
        buy_ranking, sell_ranking = {}, {}
        for symbol, returns in pred_returns.items():  # TODO: Create get_top_returns (long/short mode) function
            if self.Portfolio[symbol].IsLong:
                buy_ranking[symbol] = returns
                sell_ranking[symbol] = returns + 2 * self.commission_pct
            elif self.Portfolio[symbol].IsShort:
                buy_ranking[symbol] = returns - 2 * self.commission_pct
                sell_ranking[symbol] = returns
            else:
                buy_ranking[symbol] = returns - self.commission_pct
                sell_ranking[symbol] = returns + self.commission_pct

        buy_stocks = pd.DataFrame.from_dict(
            buy_ranking, orient='index', columns=['return']
        ).sort_values('return', ascending=False).head(self.long_stocks)
        sell_stocks = pd.DataFrame.from_dict(
            sell_ranking, orient='index', columns=['return']
        ).sort_values('return', ascending=False).tail(self.short_stocks)

        self.Log(f'Buy stocks: {buy_stocks}')
        self.Log(f'Sell stocks: {sell_stocks}')

        for symbol in self.Securities.Keys:  # TODO: Print portfolio and order status
            if symbol in buy_stocks.index:  # TODO: Check that SetHoldings does nothing if position is already held
                self.SetHoldings(symbol, self.portfolio_pct / self.long_stocks)
            elif symbol in sell_stocks.index:
                self.SetHoldings(symbol, -self.portfolio_pct / self.short_stocks)
            else:
                self.Liquidate(symbol)

    def OnSecuritiesChanged(self, changes):
        self._changes = changes
        self.Log(f"OnSecuritiesChanged({self.UtcTime}):: {changes}")