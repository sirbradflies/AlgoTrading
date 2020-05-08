"""
Basic Machine Learning Technical bot for Quantconnect
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
        self.lookback = 252
        self.long_pos, self.short_pos = 10, 10
        self.pos_size = 1.0 / (self.long_pos + self.short_pos)
        self.model = GradientBoostingRegressor()
        self.score = 0
        self.AddUniverse(self.Universe.Index.QC500)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.Train(self.DateRules.EveryDay(),
                   self.TimeRules.At(6, 0),
                   self.train_model)
        self.Schedule.On(self.DateRules.EveryDay(),
                         self.TimeRules.At(10, 0, 0),
                         self.predict_prices)
        self.X, self.Y = pd.DataFrame(), pd.DataFrame()

    def train_model(self):
        """ Train model with new data and uses the last 20% for scoring """
        self.add_data(list(self.ActiveSecurities.Keys))
        train_len = int(len(self.X) * 0.8)
        self.model.fit(self.X.iloc[:train_len], self.Y.iloc[:train_len])
        self.score = self.model.score(self.X.iloc[train_len:],
                                      self.Y.iloc[train_len:])
        self.Plot("Model", "Score", float(self.score))
        self.Debug(f'{self.Time}\tPoints:{train_len}\tScore:{self.score:.4f}')

    def predict_prices(self):
        """ If model score is better than random predicts prices and trade """
        stocks = [s for s in self.ActiveSecurities.Keys if self.IsMarketOpen(s)]
        if len(stocks) >= (self.long_pos + self.short_pos) and self.score > 0:
            # Enough stocks are tradeable and model score better than 0
            features, _ = self.get_data(symbols=stocks,
                                        n_features=self.lookback,
                                        n_targets=0)
            returns_pred = pd.DataFrame(self.model.predict(features),
                                        index=features.index.get_level_values('symbol'))
            self.trade_stocks(returns_pred)

    def add_data(self, symbols, max_len=10000):
        """ Accumulate datapoints for model training and test """
        X_new, Y_new = self.get_data(symbols=symbols,
                                     n_features=self.lookback,
                                     n_targets=1)
        self.X, self.Y = self.X.append(X_new), self.Y.append(Y_new)
        if len(self.X) > max_len: # Limit the dataset length to a maximum
            self.X, self.Y = self.X.tail(max_len), self.Y.tail(max_len)

    def get_data(self, symbols, n_features, n_targets):
        """
        Extract features and targets for model training and prediction
        for the last (n_features + n_targets) periods
        """
        history = self.History(symbols, n_features + n_targets + 1,
                               self.UniverseSettings.Resolution)
        close = history['close'].unstack("time")
        price_changes = (close / close.shift(1, axis=1) - 1).iloc[:, 1:].fillna(0)
        price_changes["time"] = self.Time
        price_changes.set_index(["time"], append=True, inplace=True)

        feature_names = [f"feat_{i}" for i in range(n_features)]
        target_names = [f"tgt_{i}" for i in range(n_targets)]
        price_changes.set_axis(feature_names + target_names, axis=1, inplace=True)
        # Return features and targets
        return price_changes.iloc[:, :n_features], price_changes.iloc[:, n_features:]

    def trade_stocks(self, returns):
        """ Rank returns and select the top for longing and bottom for shorting """
        to_long = self.rank_stocks(returns, long=True).head(self.long_pos).index
        to_short = self.rank_stocks(returns, long=False).head(self.short_pos).index
        invested = [str(s.ID) for s in self.Securities.Keys
                    if self.Portfolio[s].Invested]
        to_sell = set(invested) - set(to_long) - set(to_short)
        for symbol in to_sell:
            self.Liquidate(symbol)
        for symbol in to_long:
            self.SetHoldings(symbol, self.long_pos * self.pos_size)
        for symbol in to_short:
            self.SetHoldings(symbol, -self.short_pos * self.pos_size)
        self.Log(f'Buy stocks: {to_long}\nSell stocks: {to_short}')
        self.Log(f'Portfolio changes: {len(to_sell)}/{len(invested)}')

    def rank_stocks(self, pred_returns, long=True):
        """ Calculate best stocks to long or short from predicted returns """
        ranking = {symbol: row[0] for symbol, row in pred_returns.iterrows()}
        ranking = pd.DataFrame.from_dict(ranking, orient='index', columns=['return'])
        return ranking.sort_values('return', ascending=not long)