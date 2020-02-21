import random
import requests
import pandas as pd
import backtrader as bt
from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from yahoofinancials import YahooFinancials as yf


class MachineLearning(bt.Strategy):
    params = dict(long_stocks=5, short_stocks=5, lookback=60,
                  min_datapoints=100, max_datapoints=10000)

    def __init__(self, model):
        self.order_target = 1.0/(self.p.long_stocks+self.p.short_stocks)
        self.model = model
        self.X, self.Y = [], []
        self.stock_return = {d: StockReturn(d, plot=False) for d in self.datas}

    def next(self):
        self.update_training_data()
        if len(self.X) >= self.p.min_datapoints:
            self.train_model()
            stock_returns = self.get_stocks_pred_return()
            # TODO: Include commissions friction in pred_return
            long_stocks = stock_returns.head(self.p.long_stocks).index.values
            short_stocks = stock_returns.tail(self.p.short_stocks).index.values
            for d in self.datas:
                if d._name in long_stocks:
                    self.order_target_percent(data=d, target=self.order_target)
                elif d._name in short_stocks:
                    self.order_target_percent(data=d, target=-self.order_target)
                else:
                    self.order_target_percent(data=d, target=0.0)

    def update_training_data(self):
        """ Add training data from all datafeeds """
        for d in self.datas:
            if self.has_features(d, ago=-1):
                self.X += self.get_features(d, ago=-1)
                self.Y += [self.stock_return[d][0]]

    def train_model(self):
        """ Train ML model with available data """
        training_points = min(len(self.X), self.p.max_datapoints)
        self.model.fit(self.X[-training_points:], self.Y[-training_points:])
        self.log((f'Model score {self.model.score(self.X, self.Y):.3f}'
                  f'\t({len(self.X)} datapoints)'))

    def get_stocks_pred_return(self):
        """ Return a dataframe with the predicted return for all stocks """
        stock_returns = {}
        for d in self.datas:
                stock_returns[d._name] = self.model.predict([self.get_features(d)]) \
                    if self.has_features(d) else 0
        return pd.DataFrame.from_dict(
            stock_returns, orient='index', columns=['pred_return']
        ).sort_values('pred_return', ascending=False)

    def has_features(self, datafeed, ago=0):
        """ Check if a datafeed has enough history to return features data """
        return len(self.stock_return[datafeed]) >= self.p.lookback - ago

    def get_features(self, datafeed, ago=0):
        """ Get the features from a datafeed at a certain point in time """
        return self.stock_return[datafeed].get(ago=ago, size=self.p.lookback)

    def notify_order(self, order):
        self.log_order(order)

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log('Operation Profit, Gross %.2f, Net %.2f' %
                     (trade.pnl, trade.pnlcomm))

    def log_position(self, datafeed):
        """ Return position value of a datafeed """
        position = self.positions[datafeed]
        self.log(f'Position: {datafeed._name}\t'
                 f'Value: {position.price * position.size:,.0f}')

    def log_order(self, order):
        """ Log order price, cost and commissions """
        msg = f'Order status is {order.Status[order.status]} - '
        msg += 'Sell' if order.issell() else 'Buy'
        self.log(f'{msg} {order.params.data._name}, '
                 f'Price: {order.executed.price:.2f}, '
                 f'Cost: {order.executed.value:.2f}, '
                 f'Comm: {order.executed.comm:.2f}')

    def log(self, txt, datetime=None):
        """ Logging function for this strategy """
        datetime = datetime or self.data.datetime.date(0)
        print(f'{datetime.isoformat()} - {txt}')


class StockReturn(bt.Indicator):
    lines = ('stock_return',)

    def __init__(self):
        self.lines.stock_return = self.data_close / self.data_close(-1) - 1


class CustomBroker(bt.Observer):
    """ Tailored broker to observe long short positions and trades """
    lines = ('long_short', 'value')
    plotinfo = dict(plot=True, subplot=True)

    def next(self):
        self.lines.value[0] = self._owner.broker.getvalue()
        self.lines.long_short[0] = self.get_positions_values()

    def get_positions_values(self):
        """ Sum total value of all open positions """
        positions_value = 0
        for feed in self._owner.broker.positions:
            position = self._owner.broker.positions[feed]
            positions_value += position.price * position.size
        return positions_value


def get_sp500_symbols(date=dt.today()):
    """ Get S&P500 actualized list of symbols at a given_date """
    date_format = '%Y/%m/%d'
    json_url = 'https://gist.github.com/kafkasl/078f2c65c4299d367b57c9835b34c333/raw/dec8a9fa8e6ccf06f665e75f58f4850c37f3e290/sp500_constituents.json'
    sp_components = requests.get(json_url).json()
    sp_dates = sorted([dt.strptime(d,date_format) for d in sp_components.keys()])
    closest_sp_date = max([d for d in sp_dates if d < date]).strftime(date_format)
    return sp_components[closest_sp_date]


def get_yahoo_prices(symbol, start, end, frequency='daily'):
    """ Get historic prices in a dataframe from Yahoo Finance """
    prices = yf(symbol).get_historical_price_data(
        start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), time_interval=frequency
    )
    df = pd.DataFrame(prices[symbol]['prices'])
    return df.set_index(df['date'].apply(dt.fromtimestamp))


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.00)
    start = dt(2010, 1, 1)
    end = dt(2020, 1, 1)
    sp500_symbols = get_sp500_symbols(date=start)
    stock_universe = random.sample(sp500_symbols, 10)
    encoder = MinMaxScaler()
    for symbol in stock_universe:  # TODO: Add fundamentals: https://www.backtrader.com/blog/2019-07-19-rebalancing-conservative/rebalancing-conservative/
        print(f'Preparing feed for {symbol}')
        df_prices = get_yahoo_prices(symbol, start, end, frequency='daily')
        datafeed = bt.feeds.PandasData(dataname=df_prices, name=symbol, plot=False)
        cerebro.adddata(datafeed)
    cerebro.addstrategy(MachineLearning, model=MLPRegressor(warm_start=True))
    cerebro.addobserver(CustomBroker)
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    for analyzer in [bt.analyzers.SharpeRatio, bt.analyzers.Returns, bt.analyzers.DrawDown]:
        cerebro.addanalyzer(analyzer)
    results = cerebro.run()
    print(f'Ending Portfolio Value: {cerebro.broker.getvalue():,.0f}')
    print(f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.2f}")
    print(f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
    print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")
    cerebro.plot(fmt_x_ticks='%Y-%b-%d', volume=False)  # TODO: Add timeline ticks
