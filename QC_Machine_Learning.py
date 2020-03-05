# PROTOCOL: BT 1yr, BT 10yr, PT 1mth, LT 1mth/1k, LT1m/2k
#TODO: Test long/short only
#TODO: Test minute resolution
#TODO: Test crypto


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
        self.lookback = 21
        self.stocks = 20
        self.long_short_ratio = 1.0  # 1.0 Long only - 0.0 Short only
        self.long_stocks = int(self.stocks * self.long_short_ratio)
        self.short_stocks = self.stocks-self.long_stocks
        self.model = MLPRegressor(hidden_layer_sizes=(128,),
                                  warm_start=True)
        self.AddUniverse(self.Universe.Index.QC500)
        self.AddEquity("SPY", Resolution.Minute)
        self.Schedule.On(self.DateRules.EveryDay("SPY"),
                         self.TimeRules.AfterMarketOpen("SPY", 30),
                         self.train_model)
        self.X, self.Y = None, None

    def train_model(self):
        symbols = list(self.ActiveSecurities.Keys)
        if len(symbols) >= self.long_stocks+self.short_stocks:
            self.X = self.add_datapoints(self.X, symbols, ago=1, size=self.lookback)
            self.Y = self.add_datapoints(self.Y, symbols, ago=0, size=1)
            X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y)
            self.model.fit(X_train, Y_train)
            score = self.model.score(X_test, Y_test)
            self.Debug(f'Datapoints: {len(self.X)}\t'
                       f'Epochs: {self.model.n_iter_}\t'
                       f'Val. Score: {score:.4f}')
            #self.ObjectStore.SaveBytes("model", bytearray(self.model))  # TODO: Save model after training
            #self.model = self.ObjectStore.ReadBytes("model")

            Y_pred = self.get_features(symbols, ago=0, size=self.lookback)
            returns_pred = pd.DataFrame(self.model.predict(Y_pred), index=symbols)
            if score > 0:
                self.trade(returns_pred)

    def add_datapoints(self, data, symbols, ago, size, max_points=10000):
        """ Update train/test datapoints capping array length """
        new_data = self.get_features(symbols, ago=ago, size=size)
        data = new_data if data is None else np.vstack((data, new_data))
        return data[-max_points:] if len(data) > max_points else data

    def get_features(self, symbols, ago=0, size=1):   # TODO: Add more features
        """ Extract features for model training and prediction """
        history = self.History(symbols, size + ago + 1, Resolution.Daily)
        symbols_order = history.index.get_level_values(0).unique()
        close = history['close'].unstack(-1).loc[symbols_order]
        returns = (close / close.shift(1, axis=1) - 1).fillna(0)
        return returns.values[:, 1:1+size]  # placing symbols as index and time as columns

    def trade(self, pred_returns):
        """  Rank returns and select the top for long and bottom for short """
        long_stocks = self.rank_stocks(pred_returns, long=True).head(self.long_stocks).index
        short_stocks = self.rank_stocks(pred_returns, long=False).head(self.short_stocks).index
        invested_stocks = [s for s in self.Securities.Keys if self.Portfolio[s].Invested]
        liquidate_stocks = set(invested_stocks) - set(long_stocks) - set(short_stocks)
        self.Log(f'Buy stocks: {long_stocks}\nSell stocks: {short_stocks}')
        self.Log(f'Portfolio changes: {len(liquidate_stocks)}/{len(invested_stocks)}')

        for symbol in liquidate_stocks:
            self.Liquidate(symbol)
        for symbol in long_stocks:
            self.SetHoldings(symbol, self.long_short_ratio / self.long_stocks)
        for symbol in short_stocks:
            self.SetHoldings(symbol, -(1 - self.long_short_ratio) / self.short_stocks)

    def rank_stocks(self, pred_returns, long=True, commissions_pct=0.01, contingency=2):
        """
        Calculate best stocks to long or short from predicted returns
        and accounting for commissions and current portfolio positions
        """
        friction = (-1 if long else +1) * commissions_pct * contingency
        ranking = {}
        for symbol, row in pred_returns.iterrows():
            pred_return = row[0]
            position = self.Portfolio[symbol]
            if (long and position.IsLong) or (not long and position.IsShort):  # Symbol already in the position desired
                ranking[symbol] = pred_return  # No commissions
            elif (not long and position.IsLong) or (long and position.IsShort):  # Symbol in the opposite position
                ranking[symbol] = pred_return + 2 * friction  # Twice the commissions
            else:  # Symbol not in the portfolio
                ranking[symbol] = pred_return + friction  # One commission cost applied
        ranking = pd.DataFrame.from_dict(ranking, orient='index', columns=['return'])
        return ranking.sort_values('return', ascending=not long)
