import random
import pandas as pd
import backtrader as bt
from datetime import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from yahoofinancials import YahooFinancials as yf


class MachineLearning(bt.Strategy):
    params = dict(long_stocks=3, short_stocks=3)

    def __init__(self, model, encoder, features):
        self.order_target = 1.0/(self.p.long_stocks+self.p.short_stocks)
        self.pred_return = {}
        for d in self.datas:
            self.pred_return[d._name] = PredictedReturn(
                d, model=model, encoder=encoder, plot=False, features=features
            )

    def next(self):
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

    def get_stocks_pred_return(self):
        """ Return a dataframe with the predicted return for all stocks """
        stock_returns = {d._name: self.pred_return[d._name][0] for d in self.datas}
        return pd.DataFrame.from_dict(
            stock_returns, orient='index', columns=['pred_return']
        ).sort_values('pred_return', ascending=False)

    def log_order(self, order):
        """ Log order price, cost and commissions """
        msg = f'Order status is {order.Status[order.status]} - '
        msg += 'Sell' if order.issell() else 'Buy'
        self.log(f'{msg} {order.params.data._name}, '
                 f'Price: {order.executed.price:.2f}, '
                 f'Cost: {order.executed.value:.2f}, '
                 f'Comm: {order.executed.comm:.2f}')

    def log(self, txt, dt=None):
        """ Logging function for this strategy """
        dt = dt or self.data.datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')


class PredictedReturn(bt.Indicator):
    lines = ('pred_return',)
    params = dict(min_datapoints=10)

    def __init__(self, model, encoder, features, lookback=60):
        self.model = model
        self.encoder = encoder
        self.features = features
        self.lookback = lookback
        self.addminperiod(self.lookback + self.p.min_datapoints)
        self.lines.actual_return = self.data_close / self.data_close(-1) - 1

    def next(self):  # TODO: Make it declarative
        self.train_model()
        self.lines.pred_return[0] = self.model.predict([self.get_features(ago=0)])/self.data_close[-1]-1

    def train_model(self):  # TODO: Move train model to strategy
        features, target = [], []
        datapoints = len(self.data) - self.lookback
        for c in range(0, datapoints):  # TODO: Limit datapoints for efficiency
            features += [self.get_features(ago=-c-1)]
            target += self.data.adjclose.get(ago=-c)
        #X = self.encoder.transform(features)
        X = features
        self.model.fit(X, target)
        print(f'Model score for {self.data._name}: '
              f'{self.model.score(X, target):.4f}\t({datapoints} pts.)')

    def get_features(self, ago=0):
        feature_data = []
        for feature in self.data.features:
            feature_line = getattr(self.data,feature)
            feature_data = feature_line.get(ago=ago, size=self.lookback)
        return feature_data


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


def get_sp500_symbols():
    #TODO: Survivorship bias free listing
    # https://www.quora.com/How-can-I-get-the-complete-list-survivorship-bias-free-of-symbols-in-S-P-500
    """ Get S&P500 actualized list of symbols from wikipedia """
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    return list(table[0]['Symbol'])


def get_yahoo_prices(symbol, start, end, frequency='daily'):
    """ Get historic prices in a dataframe from Yahoo Finance """
    prices = yf(symbol.replace('.','-')).get_historical_price_data(
        start_date=start, end_date=end, time_interval=frequency
    )
    if not ('prices' in prices[symbol]
            and 'date' in prices[symbol]['prices'][0]):
        print(f'Keys missing for {symbol}!!!')
    df = pd.DataFrame(prices[symbol]['prices'])
    return df.set_index(df['date'].apply(dt.fromtimestamp))


class CustomFeed(bt.feeds.PandasData):
    features = ['adjclose', 'open', 'close']
    lines = tuple(features)
    params = {feature: -1 for feature in features}


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    stock_universe = random.sample(get_sp500_symbols(), 10)  # TODO: Select stocks public at the start date?
    encoder = MinMaxScaler()
    for symbol in stock_universe:  # TODO: Add fundamentals: https://www.backtrader.com/blog/2019-07-19-rebalancing-conservative/rebalancing-conservative/
        print(f'Preparing feed for {symbol}')
        df_prices = get_yahoo_prices(symbol,
                                     start='2010-01-01',
                                     end='2020-01-01',
                                     frequency='weekly')
        #encoder.partial_fit(df_prices[CustomFeed.features])
        datafeed = CustomFeed(dataname=df_prices, name=symbol, plot=False)
        cerebro.adddata(datafeed)
    cerebro.addstrategy(MachineLearning,
                        model=MLPRegressor(warm_start=True),
                        encoder=encoder,
                        features=CustomFeed.features)
    cerebro.addobserver(CustomBroker)
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    for analyzer in [bt.analyzers.SharpeRatio, bt.analyzers.Returns, bt.analyzers.DrawDown]:  # TODO: Add Sortino
        cerebro.addanalyzer(analyzer)
    results = cerebro.run()
    print(f'Ending Portfolio Value: {cerebro.broker.getvalue():.2f}')
    print(f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.2f}")
    print(f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
    print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")
    cerebro.plot(fmt_x_ticks='%Y-%b-%d', volume=False)  # TODO: Add timeline ticks
