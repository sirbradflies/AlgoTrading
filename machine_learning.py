import pandas as pd
import backtrader as bt
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor


class MachineLearning(bt.Strategy):
    params = dict(long_stocks=1,
                  short_stocks=0,
                  contingency=0.01)

    def __init__(self):
        self.model = GradientBoostingRegressor()  # TODO: Test neural network
        self.order_target = 1.0/(self.p.long_stocks+self.p.short_stocks)
        self.pred_return = {d._name: PredictedReturn(d, model=self.model) for d in self.datas}

    def next(self):
        stock_returns = self.get_stocks_pred_return()
        long_stocks = stock_returns.head(self.p.long_stocks).index.values
        short_stocks = stock_returns.tail(self.p.short_stocks).index.values
        for d in self.datas:
            self.log_position(d)
            if d._name in long_stocks:
                self.order_target_percent(data=d, target=self.order_target)
            # TODO: Short on Y worst stocks
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
            self.log('Operation Profit, Gross %.2f, Ner %.2f' %
                     (trade.pnl, trade.pnlcomm))

    def log_position(self, datafeed):
        """ Return position value of a datafeed """
        self.log(f'Position: {datafeed._name}\tValue: {self.get_position_value(datafeed)}')

    def get_stocks_pred_return(self):
        """ Return a dataframe with the predicted return for all stocks """
        stock_returns = {d._name: self.pred_return[d._name][0] for d in self.datas}
        return pd.DataFrame.from_dict(
            stock_returns, orient='index', columns=['pred_return']
        ).sort_values('pred_return', ascending=False)


    def get_position_value(self, datafeed):  # TODO: Move to indicator
        """ Return position value of a datafeed """
        position = self.getposition(datafeed)
        return position.size * position.price

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

    def __init__(self, model, lookback=10):
        self.model = model
        # TODO: Add warm start
        self.lookback = lookback
        self.addminperiod(self.lookback + 1)
        self.lines.actual_return = self.data_close / self.data_close(-1) - 1

    def next(self):  # TODO: Make it declarative
        self.train_model()
        self.lines.pred_return[0] = self.model.predict([self.get_features(ago=0)])

    def train_model(self):
        X, y = [], []
        datapoints = len(self.data) - self.lookback
        for c in range(0, datapoints):  # TODO: Limit datapoints for efficiency and include warm_start
            X += [self.get_features(ago=-c-1)]
            y += self.lines.actual_return.get(ago=-c)
        self.model.fit(X, y)
        print(f'Model score for {self.data._name}: {self.model.score(X, y)}')

    def get_features(self, ago=0):
        close = list(self.data_close.get(ago=ago, size=self.lookback))
        volume = list(self.data_volume.get(ago=ago, size=self.lookback))
        return close+volume


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    #stock_universe = ['AMZN','AAPL','M','MSFT']
    stock_universe = ['AAPL', 'M', 'MSFT']
    for stock in stock_universe:
        datafeed = bt.feeds.YahooFinanceCSVData(dataname=f'data/{stock}.csv',
                                                fromdate=datetime(2018, 1, 1),
                                                todate=datetime(2020, 1, 1))
        cerebro.resampledata(datafeed, timeframe=bt.TimeFrame.Months, compression=1)
    # TODO: Add fundamentals
    cerebro.addstrategy(MachineLearning)
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='sharpe')
    thestrats = cerebro.run()
    print(f'Ending Portfolio Value: {cerebro.broker.getvalue():.2f}')
    print(f'Sharpe Ratio: {thestrats[0].analyzers.sharpe.get_analysis()}')
    cerebro.plot(fmt_x_ticks='%Y-%b-%d', volume=False)  # TODO: Add timeline ticks
