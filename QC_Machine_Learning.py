"""
Basic Machine Learning bot for Quantconnect

@author: Francesco Baldisserri
@email: fbaldisserri@gmail.com
@version: 0.1
"""

import clr
clr.AddReference("System")
clr.AddReference("QuantConnect.Algorithm")
clr.AddReference("QuantConnect.Common")

from System import *
from QuantConnect import *
from QuantConnect.Algorithm import *

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
        self.model = MLPRegressor()
        self.rebalance = False
        self.month = -1
        self.AddUniverse(self.top_fundamentals, self.store_fundamentals)
        self.data_table = pd.DataFrame()

    def OnData(self, data):  # TODO: Encode variables
        """ Prepare the data, train the ML model and trade """
        if self.rebalance:
            features, targets = self.get_data(training=True)
            active_stocks = [s for s in list(self.ActiveSecurities.Keys)
                             if self.IsMarketOpen(s)]
            if len(features) > 0 and len(active_stocks) > self.portfolio_stocks:
                score = self.train_model(features, targets)
                self.Debug(f'Time: {self.Time}\tPoints: {len(features)}\t'
                           f'Epochs: {self.model.n_iter_}\tScore: {score:.4f}')
                if score > 0:
                    features, _ = self.get_data(training=False, symbols=active_stocks)
                    pred_returns = self.predict_returns(features)
                    self.trade(pred_returns, score)
            self.rebalance = False

    def get_data(self, lookback=1, training=True, symbols=None):
        """ Return features and target both for training and prediction """
        data = self.data_table.unstack('symbol')
        data['target'] = data['target'].shift(-lookback, axis='index')
        data = data.stack('symbol')
        if symbols is not None:
            data = data[data.index.get_level_values('symbol').isin(symbols)]
        data = data.dropna() if training else data[data['target'].isnull()]
        targets = data.pop('target')
        features = data
        return features, targets

    def train_model(self, features, targets):
        """ Train model with passed data and return validation score """
        X_train, X_val, Y_train, Y_val = train_test_split(features, targets)
        self.model.fit(X_train, Y_train)
        return self.model.score(X_val, Y_val)

    def predict_returns(self, features):
        """ Return expected returns by symbol """
        Y = self.model.predict(features)
        return pd.DataFrame(Y, index=features.index.get_level_values('symbol'))

    def trade(self, returns, score):
        """ Rank returns and select the top for long and bottom for short """
        to_long = self.rank_stocks(returns, long=True, score=score)\
            .head(self.long_pos).index
        to_short = self.rank_stocks(returns, long=False, score=score)\
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

    def rank_stocks(self, pred_returns, long=True, commissions_pct=0.01, score=1):
        """
        Calculate best stocks to long or short from predicted returns
        and accounting for commissions and current portfolio positions
        """
        friction = (-1 if long else +1) * commissions_pct
        ranking = {}
        for symbol, row in pred_returns.iterrows():
            exp_return = row[0] * max(score, 0)  # Normalizing returns according to model score
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
        if self.month == self.Time.month:
            return Universe.Unchanged
        else:
            self.rebalance = True
            self.month = self.Time.month
            ranked_stocks = sorted([x for x in coarse if x.HasFundamentalData],
                                   key=lambda x: x.DollarVolume, reverse=True)
            return [x.Symbol for x in ranked_stocks[:1000]]

    def store_fundamentals(self, fine):
        """ Save fundamental features in a dataframe """
        rows = []
        for x in fine:
            rows += [{'time': self.Time,
                      'symbol': x.Symbol,
                      'mom': x.ValuationRatios.PriceChange1M,
                      'pe': x.ValuationRatios.PERatio,
                      'pb': x.ValuationRatios.PBRatio,
                      'pcf': x.ValuationRatios.PCFRatio,
                      'ni': x.OperationRatios.NetMargin.OneYear,
                      'roa': x.OperationRatios.ROA.OneYear,
                      'ae': x.OperationRatios.FinancialLeverage.OneYear,
                      'target': x.ValuationRatios.PriceChange1M}]  # TODO: Correct assets/equity?
        new_data = pd.DataFrame(rows).drop_duplicates(['time', 'symbol'])
        new_data = new_data.set_index(['time', 'symbol'])
        self.data_table = self.data_table.append(new_data)
        return [x.Symbol for x in fine]