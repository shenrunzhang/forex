from tvDatafeed import TvDatafeed, Interval

tv = TvDatafeed()

data = tv.get_hist(symbol="EURUSD", exchange = "OANDA",interval=Interval.in_1_minute, n_bars=10000)

data.to_csv('eurusd1min5000.csv')