"""
Basic Machine Learning bot for Quantconnect

@author: Francesco Baldisserri
@email: fbaldisserri@gmail.com
@version: 0.2
"""

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
        self.SetStartDate(2018, 7, 1)
        self.SetEndDate(2019, 6, 30)
        self.lookback = 252
        self.portfolio_stocks = 20
        self.long_short_ratio = 0.5  # 1.0 Long only <-> 0.0 Short only
        self.long_pos = int(self.portfolio_stocks * self.long_short_ratio)
        self.short_pos = self.portfolio_stocks - self.long_pos
        self.model = MLPRegressor(hidden_layer_sizes=(128,64), max_iter=1000,
                                  early_stopping=True, tol=0, warm_start)
        self.resolution = Resolution.Daily
        self.AddUniverse(self.Universe.Index.QC500)
        self.UniverseSettings.Resolution = self.resolution
        self.Schedule.On(self.DateRules.EveryDay(),
                         self.TimeRules.At(10, 0, 0),
                         self.train_model)
        self.X, self.Y, self.X_val, self.Y_val = None, None, None, None

    def train_model(self):
        active_stocks = [s for s in list(self.ActiveSecurities.Keys)
                             if self.IsMarketOpen(s)]
        if len(active_stocks) >= self.portfolio_stocks:
            train_stocks, val_stocks = train_test_split(active_stocks)
            self.X, self.Y = self.add_data(self.X, self.Y, train_stocks)
            self.model.fit(self.X, self.Y)
            self.X_val, self.Y_val = self.add_data(self.X_val, self.Y_val, val_stocks)
            self.score = self.model.score(self.X_val, self.Y_val)
            self.Debug(f'Time: {self.Time}\tPoints: {len(self.X)}\t'
                       f'Epochs: {self.model.n_iter_}\tScore: {self.score:.4f}')
            if self.score > 0:  # If model better than random then trade
                features, _ = self.get_data(symbols=active_stocks,
                                            features=self.lookback,
                                            targets=0)
                returns_predicted = pd.DataFrame(self.model.predict(features),
                                                 index=features.index)
                self.trade(returns_predicted)

    def add_data(self, X_old, Y_old, symbols, max_len=50000):
        """ Accumulate datapoints for model training and test """
        X_new, Y_new = self.get_data(symbols=symbols,
                                     features=self.lookback,
                                     targets=1)
        X = X_new if X_old is None else np.vstack((X_old, X_new))
        Y = Y_new if Y_old is None else np.vstack((Y_old, Y_new))
        return (X[-max_len:], Y[-max_len:]) if len(X) > max_len else (X, Y)

    def get_data(self, symbols, features, targets):
        """ Extract datapoints for model training and prediction """
        history = self.History(symbols, targets + features + 1, self.resolution)
        close = history['close'].unstack(-1)
        returns = (close / close.shift(1, axis=1) - 1).iloc[:, 1:].fillna(0)
        return returns.iloc[:, :features], returns.iloc[:, features:]  # Return X, Y

    def trade(self, returns):
        """ Rank returns and select the top for long and bottom for short """
        to_long = self.rank_stocks(returns, long=True).head(self.long_pos).index
        to_short = self.rank_stocks(returns, long=False).head(self.short_pos).index
        invested = [str(s.ID) for s in self.Securities.Keys if self.Portfolio[s].Invested]
        to_sell = set(invested) - set(to_long) - set(to_short)
        for symbol in to_sell:
            self.Liquidate(symbol)
        for symbol in to_long:
            self.SetHoldings(symbol, self.long_short_ratio / self.long_pos)
        for symbol in to_short:
            self.SetHoldings(symbol, -(1 - self.long_short_ratio) / self.short_pos)
        self.Log(f'Buy stocks: {to_long}\nSell stocks: {to_short}')
        self.Log(f'Portfolio changes: {len(to_sell)}/{len(invested)}')

    def rank_stocks(self, pred_returns, long=True, commissions_pct=0.01):
        """
        Calculate best stocks to long or short from predicted returns
        and accounting for commissions and current portfolio positions
        """
        friction = (-1 if long else +1) * commissions_pct
        ranking = {}
        for symbol, row in pred_returns.iterrows():
            exp_return = row[0] * max(0, self.score)  # Normalizing return by model score when positive
            position = self.Portfolio[symbol]
            if (long and position.IsLong) or (not long and position.IsShort):  # Symbol already in the position desired
                ranking[symbol] = exp_return  # No commissions
            elif (not long and position.IsLong) or (long and position.IsShort):  # Symbol in the opposite position
                ranking[symbol] = exp_return + 2 * friction  # Twice the commissions
            else:  # Symbol not in the portfolio
                ranking[symbol] = exp_return + friction  # One commission cost applied
        ranking = pd.DataFrame.from_dict(ranking, orient='index', columns=['return'])
        return ranking.sort_values('return', ascending=not long)
