"""
Machine Learning Value bot for Quantconnect

@author: Francesco Baldisserri
@email: fbaldisserri@gmail.com
@version: 0.5
"""

import clr

clr.AddReference("System")
clr.AddReference("QuantConnect.Algorithm")
clr.AddReference("QuantConnect.Common")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *

import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor


class NeuralNetworkAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2008, 1, 1)
        self.SetEndDate(2018, 1, 1)
        self.holdings = 20
        self.long_short_ratio = 0.5  # 1.0 Long only <-> 0.0 Short only
        self.long_pos = int(self.holdings * self.long_short_ratio)
        self.short_pos = self.holdings - self.long_pos
        self.feat_encoder, self.targ_encoder = MinMaxScaler(), MinMaxScaler()
        self.model = MLPRegressor(hidden_layer_sizes=(32,), early_stopping=True)
        self.history = pd.DataFrame()
        self.maxpoints = 10000
        self.last_update = self.last_execution = datetime(1, 1, 1)
        self.AddUniverse(self.top_fundamentals, self.store_fundamentals)
        self.Schedule.On(self.DateRules.EveryDay(),
                         self.TimeRules.At(10, 0, 0),
                         self.execute_strategy)

    def execute_strategy(self):
        """ Prepare the data, train the ML model and trade """
        active_stocks = [s for s in list(self.ActiveSecurities.Keys)
                         if self.IsMarketOpen(s)]
        if self.last_update > self.last_execution and \
                len(active_stocks) > self.holdings:  # New data, train the model
            self.last_execution = self.Time
            features, targets = self.get_data(training=True)
            if len(features) > 0:  # Enough features, train model
                score = self.train_model(features, targets)
                self.Plot("Algorithm", "Score", float(max(score, -1.0)))
                self.Debug(f'Time: {self.Time}\tPoints: {len(features)}\t'
                           f'Epochs: {self.model.n_iter_}\tScore: {score:.4f}')
                if score > 0:  # Valid model, predict returns and trade
                    features, _ = self.get_data(training=False,
                                                symbols=active_stocks)
                    pred_returns = self.predict_returns(features)
                    self.trade(pred_returns)

    def get_data(self, training=True, symbols=None):
        """ Return features and target both for training and prediction """
        data = self.history.dropna()
        if symbols is not None:
            data = data[data.index.get_level_values('symbol').isin(symbols)]
        target = data[['return']].unstack().shift(-1).stack(dropna=False)
        target = target.dropna() if training else target.loc[target.isnull().any(1)]
        mask = target.index.intersection(data.index)
        if len(mask) > self.maxpoints:
            mask = mask[-self.maxpoints:]
        return data.loc[mask, :], target.loc[mask, :]

    def train_model(self, features, target):
        """ Train model with passed data and return validation score """
        X = self.feat_encoder.fit_transform(features)
        Y = self.targ_encoder.fit_transform(target)
        return self.model.fit(X, Y).best_validation_score_

    def predict_returns(self, features):
        """ Return expected returns by symbol """
        X = self.feat_encoder.transform(features)
        Y = self.model.predict(X).reshape(-1, 1)
        return pd.DataFrame(self.targ_encoder.inverse_transform(Y),
                            index=features.index.get_level_values('symbol'),
                            columns=['return'])

    def trade(self, returns):
        """ Rank returns and select the top for long and bottom for short """
        long_ranking = self.rank_stocks(returns, long=True)
        to_long = long_ranking.head(self.long_pos).index
        short_ranking = self.rank_stocks(returns, long=False)
        to_short = short_ranking.head(self.short_pos).index
        invested = [s for s in self.Securities.Keys if self.Portfolio[s].Invested]
        to_sell = set(invested) - set(to_long) - set(to_short)

        for symbol in to_sell:
            self.Liquidate(symbol)
        for symbol in to_long:
            self.SetHoldings(symbol, self.long_short_ratio / self.long_pos)
        for symbol in to_short:
            self.SetHoldings(symbol, -(1 - self.long_short_ratio) / self.short_pos)

        self.Log(f'Longs: {to_long}\nShorts: {to_short}')
        self.Log(f'Changes: {len(to_sell)}/{len(invested)}')

    def rank_stocks(self, pred_returns, long=True):
        """ Calculate best stocks to long or short from predicted returns """
        ranking = {symbol: ret for symbol, ret in pred_returns.iteritems()}
        ranking = pd.DataFrame.from_dict(ranking, orient='index', columns=['return'])
        return ranking.sort_values('return', ascending=not long)

    def top_fundamentals(self, coarse):
        """ Return top 100 stocks by volume with fundamentals """
        if self.last_update.month == self.Time.month:
            return Universe.Unchanged
        else:
            self.last_update = self.Time
            ranked_stocks = sorted([x for x in coarse if x.HasFundamentalData],
                                   key=lambda x: x.DollarVolume, reverse=True)
            return [x.Symbol for x in ranked_stocks[:100]]

    def store_fundamentals(self, fine):
        """ Save fundamental features in a history dataframe """
        rows = []
        for x in fine:
            rows += [{'time': self.Time,
                      'symbol': x.Symbol,
                      'pe': x.ValuationRatios.PERatio,
                      'roe': x.OperationRatios.ROE.OneYear,
                      'return': x.ValuationRatios.PriceChange1M}]
        data = pd.DataFrame(rows).drop_duplicates(['time', 'symbol'])
        self.history = self.history.append(data.set_index(['time', 'symbol']))
        return [x.Symbol for x in fine]