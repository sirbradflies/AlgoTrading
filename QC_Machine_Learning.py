"""
Machine Learning Value bot for Quantconnect

@author: Francesco Baldisserri
@email: fbaldisserri@gmail.com
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
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split


class NeuralNetworkAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2008, 7, 1)
        self.SetEndDate(2019, 6, 30)
        self.portfolio_stocks = 20
        self.long_short_ratio = 0.5  # 1.0 Long only <-> 0.0 Short only
        self.long_pos = int(self.portfolio_stocks * self.long_short_ratio)
        self.short_pos = self.portfolio_stocks - self.long_pos
        self.feat_encoder, self.targ_encoder = None, None
        self.model = MLPRegressor(hidden_layer_sizes=(64, 32), tol=0,
                                  early_stopping=True, warm_start=True)
        self.history = pd.DataFrame()
        self.history_maxlen = 10000  # TODO: Convert to periods
        self.last_update = self.last_execution = datetime(1, 1, 1)
        self.AddUniverse(self.top_fundamentals, self.store_fundamentals)
        self.Schedule.On(self.DateRules.EveryDay(),
                         self.TimeRules.At(10, 0, 0),
                         self.execute_strategy)

    def execute_strategy(self):
        """ Prepare the data, train the ML model and trade """
        active_stocks = [s for s in list(self.ActiveSecurities.Keys)
                         if self.IsMarketOpen(s)]
        if self.last_update > self.last_execution \
                and len(active_stocks) > self.portfolio_stocks:  # New data, prepare features
            self.last_execution = self.last_update
            features, targets = self.get_data(training=True)
            if len(features) > 0:  # Enough features, train model
                score = self.train_model(features, targets)
                self.Debug(f'Time: {self.Time}\tPoints: {len(features)}\t'
                           f'Epochs: {self.model.n_iter_}\tScore: {score:.4f}')
                if score > 0:  # Valid model, make predictions and trade
                    features, _ = self.get_data(training=False, symbols=active_stocks)
                    pred_returns = self.predict_returns(features)
                    self.trade(pred_returns, score)

    def get_data(self, training=True, symbols=None):  # TODO: Test with lookback periods
        """ Return features and target both for training and prediction """
        data = self.history.dropna()
        if symbols is not None:
            data = data[data.index.get_level_values('symbol').isin(symbols)]
        target = data['return'].unstack('symbol').shift(-1).stack('symbol', dropna=False)
        target = target.dropna() if training else target.loc[target.isnull()]  
        features = data
        mask = target.index.intersection(features.index)  # TODO: Refactor
        return features.loc[mask, :], target.loc[mask]

    def train_model(self, features, target):
        """ Train model with passed data and return validation score """
        if self.feat_encoder is None or self.targ_encoder is None:
            self.feat_encoder = MinMaxScaler().fit(features)
            self.targ_encoder = MinMaxScaler.fit(target.values.reshape(-1, 1))
        X = self.feat_encoder.transform(features)
        Y = self.targ_encoder.transform(target.values.reshape(-1, 1))
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y)
        self.model.fit(X_train, Y_train)
        return self.model.score(X_val, Y_val)

    def predict_returns(self, features):
        """ Return expected returns by symbol """
        X = self.targ_encoder.transform(features)
        Y = self.model.predict(X).reshape(-1, 1)
        return pd.DataFrame(self.targ_encoder.inverse_transform(Y),
                            index=features.index.get_level_values('symbol'))

    def trade(self, returns, score):  # TODO: Clean
        """ Rank returns and select the top for long and bottom for short """
        to_long = self.rank_stocks(returns, long=True, confidence=score)\
            .head(self.long_pos).index
        to_short = self.rank_stocks(returns, long=False, confidence=score)\
            .head(self.short_pos).index
        invested = [s for s in self.Securities.Keys if self.Portfolio[s].Invested]
        to_sell = set(invested) - set(to_long) - set(to_short)
        for symbol in to_sell:
            self.Liquidate(symbol)
        for symbol in to_long:
            self.SetHoldings(symbol, self.long_short_ratio / self.long_pos)
        for symbol in to_short:
            self.SetHoldings(symbol, -(1 - self.long_short_ratio) / self.short_pos)
        self.Log(f'Buy stocks: {to_long}\nSell stocks: {to_short}')
        self.Log(f'Portfolio changes: {len(to_sell)}/{len(invested)}')

    def rank_stocks(self, pred_returns, long=True, commission=0.01, confidence=1):
        """
        Calculate best stocks to long or short from predicted returns
        and accounting for commissions and current portfolio positions
        """
        friction = (-1 if long else +1) * commission
        ranking = {}
        for symbol, row in pred_returns.iterrows():
            exp_return = row[0] * max(confidence, 0)  # Normalizing returns according to model score
            position = self.Portfolio[symbol]
            if (long and position.IsLong) or (not long and position.IsShort):  # Symbol already in the position desired
                ranking[symbol] = exp_return  # No commissions
            elif (not long and position.IsLong) or (long and position.IsShort):  # Symbol in the opposite position
                ranking[symbol] = exp_return + 2 * friction  # Twice the commissions
            else:  # Symbol not in the portfolio
                ranking[symbol] = exp_return + friction  # One commission cost applied
        ranking = pd.DataFrame.from_dict(ranking, orient='index', columns=['return'])
        return ranking.sort_values('return', ascending=not long)

    def top_fundamentals(self, coarse):
        """ Return top 1000 stocks by volume with fundamentals """
        if self.last_update.month == self.Time.month:
            return Universe.Unchanged
        else:
            self.last_update = self.Time
            ranked_stocks = sorted([x for x in coarse if x.HasFundamentalData],
                                   key=lambda x: x.DollarVolume, reverse=True)
            return [x.Symbol for x in ranked_stocks[:1000]]

    def store_fundamentals(self, fine):
        """ Save fundamental features in a dataframe """
        rows = []
        for x in fine:
            rows += [{'time': self.Time,
                      'symbol': x.Symbol,
                      'pe': x.ValuationRatios.PERatio,
                      'pb': x.ValuationRatios.PBRatio,
                      'pcf': x.ValuationRatios.PCFRatio,
                      'ni': x.OperationRatios.NetMargin.OneYear,
                      'roa': x.OperationRatios.ROA.OneYear,
                      'ae': x.OperationRatios.FinancialLeverage.OneYear,  # TODO: Correct assets/equity?
                      'return': x.ValuationRatios.PriceChange1M}]
        new_data = pd.DataFrame(rows).drop_duplicates(['time', 'symbol'])
        new_data = new_data.set_index(['time', 'symbol'])
        self.history = self.history.append(new_data)
        self.history = self.history.tail(self.history_maxlen) \
            if len(self.history) > self.history_maxlen else self.history
        return [x.Symbol for x in fine]