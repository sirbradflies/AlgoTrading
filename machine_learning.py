import backtrader as bt
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor


class MachineLearning(bt.Strategy):
    params = dict(order_target=1.0, contingency=0.01)

    def __init__(self):
        self.pred_price = PricePredictor(subplot=False)
        self.tick_return = TickReturn()

    def next(self):
        # TODO: Sort stocks by expected gain
        # TODO: Long on X best stocks
        if self.pred_price > self.data_close*(1+self.p.contingency):
            self.order_target_percent(target=self.p.order_target)
            self.buy()  # Long the stock
        # TODO: Short on Y worst stocks
        elif self.pred_price < self.data_close*(1-self.p.contingency):
            self.order_target_percent(target=-self.p.order_target)
            self.sell()  # Short the stock

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return  # TODO: Refactor

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, '
                         f'Price: {order.executed.price:.2f}, '
                         f'Cost: {order.executed.value:.2f}, '
                         f'Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, '
                         f'Price: {order.executed.price:.2f}, '
                         f'Cost: {order.executed.value:.2f}, '
                         f'Comm: {order.executed.comm:.2f}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                     (trade.pnl, trade.pnlcomm))

    def log(self, txt, dt=None):
        """ Logging function for this strategy """
        dt = dt or self.data.datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')


class TickReturn(bt.Indicator):
    lines = ('tick_return',)

    def __init__(self):
        self.lines.tick_return = self.data_close/self.data_close(-1)-1

class PricePredictor(bt.Indicator):
    lines = ('pred_price',)

    def __init__(self, lookback=10):
        self.model = GradientBoostingRegressor()  # TODO: Test neural network
        # TODO: Add warm start
        self.lookback = lookback
        self.addminperiod(self.lookback + 1)

    def next(self):  # TODO: Make it declarative
        self.train_model()
        self.lines.pred_price[0] = self.predict()

    def train_model(self):
        X, y = [], []
        datapoints = len(self.data) - self.lookback
        for c in range(0, datapoints):  # TODO: Limit datapoints for efficiency and include warm_start
            X += [self.get_features(ago=-c-1)]
            y += self.data_close.get(ago=-c)
        self.model.fit(X, y)
        print(f'Model score {self.model.score(X, y)}')

    def predict(self):
        return self.model.predict([self.get_features(ago=0)])

    def get_features(self, ago=0):
        close = list(self.data_close.get(ago=ago, size=self.lookback))
        volume = list(self.data_volume.get(ago=ago, size=self.lookback))
        return close+volume

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    #stock_universe = ['AMZN','AAPL','M','MSFT']
    stock_universe = ['M']
    for stock in stock_universe:
        datafeed = bt.feeds.YahooFinanceCSVData(dataname=f'data/{stock}.csv',
                                                fromdate=datetime(2010, 1, 1),
                                                todate=datetime(2020, 1, 1))
        cerebro.resampledata(datafeed, timeframe=bt.TimeFrame.Months, compression=1)  # TODO: Test monthly steps
    # TODO: Add fundamentals
    cerebro.addstrategy(MachineLearning)
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.PositionsValue, _name='positions')
    thestrats = cerebro.run()
    print(f'Sharpe Ratio: {thestrats[0].analyzers.sharpe.get_analysis()}')
    print(f'Ending Portfolio Value: {cerebro.broker.getvalue():.2f}')
    cerebro.plot()  # TODO: Add timeline ticks
