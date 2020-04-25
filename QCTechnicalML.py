"""
Basic Machine Learning bot for Quantconnect
@version: 0.3
"""

import clr
clr.AddReference("System")
clr.AddReference("QuantConnect.Algorithm")
clr.AddReference("QuantConnect.Common")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


class NeuralNetworkAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2018, 7, 1)
        self.SetCash(1000000)
        self.SetBrokerageModel(BrokerageName.AlphaStreams)
        self.lookback = 61
        self.portfolio_stocks = 20
        self.long_short_ratio = 0.5  # 1.0 Long only <-> 0.0 Short only
        self.long_pos = int(self.portfolio_stocks * self.long_short_ratio)
        self.short_pos = self.portfolio_stocks - self.long_pos
        self.model = GradientBoostingRegressor()
        self.score = 0
        self.resolution = Resolution.Daily
        self.AddUniverse(self.Universe.Index.QC500)
        self.UniverseSettings.Resolution = self.resolution
        self.Train(self.DateRules.EveryDay(),
                   self.TimeRules.At(6, 0),
                   self.train_model)
        self.Schedule.On(self.DateRules.EveryDay(),
                         self.TimeRules.At(10, 0, 0),
                         self.trade_stocks)
        self.X, self.Y = pd.DataFrame(), pd.DataFrame()

    def train_model(self):
        self.add_data(list(self.ActiveSecurities.Keys))
        X_train, X_test, Y_train, Y_test = self.split_data(self.X, self.Y)
        self.model.fit(X_train, Y_train)
        if len(X_test) > 0:
            self.score = self.model.score(X_test, Y_test)
            self.Plot("Model", "Score", float(self.score))
            self.Debug(f'{self.Time}\tPoints: {len(X_train)}\t'
                       f'Score: {self.score:.4f}')

    def trade_stocks(self):
        stocks = [s for s in self.ActiveSecurities.Keys
                         if self.IsMarketOpen(s)]
        if len(stocks) >= self.portfolio_stocks and self.score > 0:  # If model better than random then trade
            features, _ = self.get_data(symbols=stocks,
                                        n_features=self.lookback,
                                        n_targets=0)
            returns_pred = pd.DataFrame(self.model.predict(features),
                                        index=features.index.get_level_values('symbol'))
            self.trade(returns_pred)

    def add_data(self, symbols, max_len=100000):  # TODO: Simplify add_data
        """ Accumulate datapoints for model training and test """
        X_new, Y_new = self.get_data(symbols=symbols,
                                     n_features=self.lookback,
                                     n_targets=1)
        self.X, self.Y = self.X.append(X_new), self.Y.append(Y_new)
        if len(self.X) > max_len:
            self.X, self.Y = self.X.tail(max_len), self.Y.tail(max_len)

    def get_data(self, symbols, n_features, n_targets):
        """ Extract datapoints for model training and prediction """
        history = self.History(symbols, n_features+n_targets+1, self.resolution)
        close = history['close'].unstack("time")
        data = (close / close.shift(1, axis=1) - 1).iloc[:, 1:].fillna(0)
        data["time"] = self.Time
        data.set_index(["time"], append=True, inplace=True)
        features = [f"feat_{i}" for i in range(n_features)]
        targets = [f"tgt_{i}" for i in range(n_targets)]
        data.set_axis(features+targets, axis=1, inplace=True)
        return data.iloc[:, :n_features], data.iloc[:, n_features:]

    def split_data(self, X, Y, test_split=0.2):
        """ Split the time series in train and test data"""
        time_ix = X.index.get_level_values("time")
        time = time_ix.unique().sort_values()
        train_len = len(time) - int(len(time) * test_split)
        train_ix, test_ix = time[:train_len], time[train_len:]
        X_train, Y_train = X[time_ix.isin(train_ix)], Y[time_ix.isin(train_ix)]
        X_test, Y_test = X[time_ix.isin(test_ix)], Y[time_ix.isin(test_ix)]
        return X_train, X_test, Y_train, Y_test

    def trade(self, returns):
        """ Rank returns and select the top for long and bottom for short """
        to_long = self.rank_stocks(returns, long=True).head(self.long_pos).index
        to_short = self.rank_stocks(returns, long=False).head(self.short_pos).index
        invested = [str(s.ID) for s in self.Securities.Keys
                    if self.Portfolio[s].Invested]
        to_sell = set(invested) - set(to_long) - set(to_short)
        for symbol in to_sell:
            self.Liquidate(symbol)
        for symbol in to_long:
            self.SetHoldings(symbol, self.long_short_ratio / self.long_pos)
        for symbol in to_short:
            self.SetHoldings(symbol, -(1 - self.long_short_ratio) / self.short_pos)
        self.Log(f'Buy stocks: {to_long}\nSell stocks: {to_short}')
        self.Log(f'Portfolio changes: {len(to_sell)}/{len(invested)}')

    def rank_stocks(self, pred_returns, long=True):
        """ Calculate best stocks to long or short from predicted returns """
        ranking = {symbol: row[0] for symbol, row in pred_returns.iterrows()}
        ranking = pd.DataFrame.from_dict(ranking, orient='index', columns=['return'])
        return ranking.sort_values('return', ascending=not long)