import os
import requests
import pandas as pd
import backtrader as bt
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from iexfinance.stocks import get_historical_data
# TODO: Explore why there are peaks of short stocks


class MachineLearning(bt.Strategy):
    params = dict(long_stocks=2,
                  short_stocks=2)

    def __init__(self):  # TODO: Add position (long / short) indicator
        self.model = MLPRegressor(warm_start=True)
        self.order_target = 1.0/(self.p.long_stocks+self.p.short_stocks)
        self.pred_return = {d._name: PredictedReturn(d, model=self.model, plot=False)
                            for d in self.datas}
        # TODO: Add nr. of trades per tick

    def next(self):
        stock_returns = self.get_stocks_pred_return()
        # TODO: Change when New_Pred_Return > Old_Pred_Return + Commissions
        long_stocks = stock_returns.head(self.p.long_stocks).index.values
        short_stocks = stock_returns.tail(self.p.short_stocks).index.values
        # TODO: Check balancing strategy on: https://www.backtrader.com/blog/2019-07-19-rebalancing-conservative/rebalancing-conservative/
        for d in self.datas:
            self.log_position(d)
            if d._name in long_stocks:
                self.order_target_percent(data=d, target=self.order_target)
            elif d._name in short_stocks:
                self.order_target_percent(data=d, target=-self.order_target)
            else:
                self.order_target_percent(data=d, target=0.0)

    def notify_order(self, order):
        # Check if an order has been completed
        if order.status in [order.Completed]:
            self.log_order(order)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log('Operation Profit, Gross %.2f, Net %.2f' %
                     (trade.pnl, trade.pnlcomm))

    def log_position(self, datafeed):
        """ Return position value of a datafeed """
        self.log(f'Position: {datafeed._name}\t'
                 f'Value: {self.get_position_value(datafeed)}')

    def get_position_value(self, datafeed):
        """ Return the total value of a position """
        return self.positions[datafeed].price*self.positions[datafeed].size

    def get_stocks_pred_return(self):
        """ Return a dataframe with the predicted return for all stocks """
        stock_returns = {d._name: self.pred_return[d._name][0] for d in self.datas}
        return pd.DataFrame.from_dict(
            stock_returns, orient='index', columns=['pred_return']
        ).sort_values('pred_return', ascending=False)

    def log_order(self, order):
        """ Log order price, cost and commissions """
        msg = 'Sell' if order.issell() else 'Buy'
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

    def __init__(self, model, lookback=12):
        self.model = model
        self.lookback = lookback
        self.addminperiod(self.lookback + 1)
        self.lines.actual_return = self.data_close / self.data_close(-1) - 1

    def next(self):  # TODO: Make it declarative
        self.train_model()
        self.lines.pred_return[0] = self.model.predict([self.get_features(ago=0)])/self.data_close[-1]-1

    def train_model(self):
        X, y = [], []
        datapoints = len(self.data) - self.lookback
        for c in range(0, datapoints):  # TODO: Limit datapoints for efficiency
            X += [self.get_features(ago=-c-1)]
            #y += self.lines.actual_return.get(ago=-c)
            y += self.data_close.get(ago=-c)
        self.model.fit(X, y)  # TODO: Fix negative score
        print(f'Model score for {self.data._name}: {self.model.score(X, y)}')

    def get_features(self, ago=0):
        close = list(self.data_close.get(ago=ago, size=self.lookback))
        #volume = list(self.data_volume.get(ago=ago, size=self.lookback))  # TODO: Add back volume
        return close


def set_iex_mode(mode='sandbox'):
    """ Select IEX mode: 'sandbox' or 'live' """
    if mode == 'sandbox':
        os.environ['IEX_API_VERSION'] = 'iexcloud-sandbox'
        os.environ['IEX_TOKEN'] = 'Tpk_4cd09bd0d413453e855bb465198b29ca'
    elif mode == 'live':
        os.environ['IEX_API_VERSION'] = 'iexcloud-sandbox'
        os.environ['IEX_TOKEN'] = 'pk_760e9e753a004c6f9f45088c0d7cc42f'


def get_sp500_symbols():
    """ Get S&P500 actualized list of symbols from wikipedia """
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    return table[0]['Symbol']


def get_historic_prices(stock, start, end):
    """ Get historical prices from IEX data provider """
    return get_historical_data(stock, start, end,
                               output_format='pandas')


def download_yahoo_stocks(stock, start='1262304000', end='1577836800', crumb='GwEq4sk1BX9'):
    r = requests.get(
        f"https://query1.finance.yahoo.com/v7/finance/download/{stock}?period1={start}&period2={end}&interval=1d&events=history&crumb={crumb}")
    file = open(f"data\{stock}.csv", 'w')
    file.write(r.text)
    file.close()


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.01)  # TODO: Set commissions
    #stock_universe = get_sp500_symbols().sample(10)
    for symbol in get_sp500_symbols():
        download_yahoo_stocks(symbol)
    #set_iex_mode('sandbox')
    stock_universe = [f for f in os.listdir('data/') if '.csv' in f]
    for stock in stock_universe:
        #iex_prices = get_historic_prices(stock=stock,
        #                                 start=datetime(2018, 1, 1),
        #                                 end=datetime(2020, 1, 1))
        #datafeed = bt.feeds.PandasData(dataname=iex_prices,
        #                               name=stock,
        #                               openinterest=None)
        datafeed = bt.feeds.YahooFinanceCSVData(dataname=f'data/{stock}',
                                                fromdate=datetime(2010, 1, 1),
                                                todate=datetime(2020, 1, 1),
                                                plot=False)
        cerebro.resampledata(datafeed, timeframe=bt.TimeFrame.Days, compression=1)
    cerebro.addstrategy(MachineLearning)
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    for analyzer in [bt.analyzers.SharpeRatio, bt.analyzers.Returns, bt.analyzers.DrawDown]:
        cerebro.addanalyzer(analyzer)
    results = cerebro.run()
    print(f'Ending Portfolio Value: {cerebro.broker.getvalue():.2f}')
    print(f"Sharpe: {results[0].analyzers.sharperatio.get_analysis()['sharperatio']:.2f}")
    print(f"Norm. Annual Return: {results[0].analyzers.returns.get_analysis()['rnorm100']:.2f}%")
    print(f"Max Drawdown: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")
    cerebro.plot(fmt_x_ticks='%Y-%b-%d', volume=False)  # TODO: Add timeline ticks
