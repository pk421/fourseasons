__author__ = 'Mike'

def do_quandl_demo():
    # This comes from here: https://data-flair.training/blogs/python-for-stock-market/

    import quandl
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split

    import pdb; pdb.set_trace()
    amazon = quandl.get("WIKI/AMZN")
    amazon = amazon[['Adj. Close']]
    print amazon.tail()

    forecast_length = 3
    amazon['Predicted'] = amazon[['Adj. Close']].shift(-forecast_length)
    print amazon.tail()

    x = np.array(amazon.drop(['Predicted'], 1))
    x = x[:-forecast_length]
    print x

    y = np.array(amazon['Predicted'])
    y = y[:-forecast_length]
    print y

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4)

    svr_rbf=SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train, y_train)

    svr_rbf_confidence = svr_rbf.score(x_test, y_test)
    print "SVR Confidence: ", round(svr_rbf_confidence*100,2)

    lr = LinearRegression()
    lr.fit(x_train, y_train)

    lr_confidence = lr.score(x_test, y_test)
    print "LR Confidence: ", round(lr_confidence*100, 2)

def do_larger_regression_demo():
    # comes from here: https://blog.quantinsti.com/support-vector-machines-introduction/

    from pandas_datareader import data as web
    import numpy as np
    import pandas as pd
    from sklearn import mixture as mix
    import seaborn as sns
    import matplotlib.pyplot as plt
    import talib as ta
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    import yfinance

    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()


    df = web.get_data_yahoo('SPY', start='1993-01-01', end='2020-05-22')
    df = df[['Open', 'High', 'Low', 'Close']]

    gary_short = 8
    gary_long = 13
    gary_smoothing = 5
    n=gary_short
    training_fraction = 0.8
    split = int(training_fraction * len(df))

    df['open_shift'] = df['Open'].shift(1)
    df['high_shift'] = df['High'].shift(1)
    df['low_shift'] = df['Low'].shift(1)
    df['close_shift'] = df['Close'].shift(1)
    df['RSI'] = ta.RSI(np.array(df['close_shift']), timeperiod = n)
    df['SMA'] = df['close_shift'].rolling(window = n).mean()
    df['Corr'] = df['SMA'].rolling(window=n).corr(df['close_shift'])
    df['SAR'] = ta.SAR(np.array(df['high_shift']), np.array(df['low_shift']), 0.2, 0.2)
    df['ADX'] = ta.ADX(np.array(df['high_shift']), np.array(df['low_shift']), np.array(df['close_shift']), timeperiod=n)
    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))

    df = df.dropna()
    df=df.drop(['High', 'Low', 'Close'], axis=1)
    print df.head()

    ss = StandardScaler()
    unsupervised = mix.GaussianMixture(n_components=4,
                                       covariance_type='spherical',
                                       n_init=100,
                                       random_state=42)

    unsupervised.fit(np.reshape(ss.fit_transform(df[:split]), (-1, df.shape[1])))
    regime = unsupervised.predict(np.reshape(ss.transform(df[split:]), (-1, df.shape[1])))

    Regimes = pd.DataFrame(regime, columns=['Regime'], index=df[split:].index).join(df[split:], how='inner'). \
                                    assign(market_cu_return=df[split:].Return.cumsum()).reset_index(drop=False). \
                                    rename(columns={'index':'Date'})

    import pdb; pdb.set_trace()

    order = [0, 1, 2, 3]
    for i in order:
        print 'Mean for regime: ', unsupervised.means_[i][0]
        print 'Co-Variance for regime: ', unsupervised.covariances_[i]

    fig = sns.FacetGrid(data=Regimes, hue='Regime', hue_order=order, aspect=2, size=4)
    fig.map(plt.scatter, 'Date', 'market_cu_return', s=4).add_legend()
    plt.show()

    ss1 = StandardScaler()
    columns = Regimes.columns.drop(['Regime', 'Date'])
    Regimes[columns] = ss1.fit_transform(Regimes[columns])
    Regimes['Signal'] = 0
    Regimes.loc[Regimes['Return'] > 0, 'Signal'] = 1
    Regimes.loc[Regimes['Return'] < 0, 'Signal'] = -1

    cls = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto',
              kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)

    split2 = int(training_fraction * len(Regimes))

    X = Regimes.drop(['Signal', 'Return', 'market_cu_return', 'Date'], axis=1)
    y = Regimes['Signal']

    cls.fit(X[:split2], y[:split2])

    p_data = len(X)-split2

    df['Pred_Signal'] = 0
    df.iloc[-p_data:, df.columns.get_loc('Pred_Signal')] = cls.predict(X[split2:])

    # print df['Pred_Signal'] * df['Return'].shift(-1)

    df['str_ret'] = df['Pred_Signal'] * df['Return'].shift(-1)

    df['strategy_cu_return'] = 0.
    df['market_cu_return'] = 0.
    df.iloc[-p_data:, df.columns.get_loc('strategy_cu_return')] = np.nancumsum(df['str_ret'][-p_data:])
    df.iloc[-p_data:, df.columns.get_loc('market_cu_return')] = np.nancumsum(df['Return'][-p_data:])
    Sharpe = (df['strategy_cu_return'][-1] -df['market_cu_return'][-1]) / np.nanstd(df['strategy_cu_return'][-p_data:])

    plt.plot(df['strategy_cu_return'][-p_data:], color='g', label='Strategy Return')
    plt.plot(df['market_cu_return'][-p_data:], color='r', label='Market Return')
    plt.figtext(0.14, 0.9, s='Sharpe ratio: %.2f' %Sharpe)
    plt.legend(loc='best')
    plt.show()

    print 'here'

def multiple_linear_regression():
    # https://datatofish.com/multiple-linear-regression-python/

    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import linear_model
    import statsmodels.api as sm
    import Tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    Stock_Market = {'Year': [2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],
                    'Month': [12, 11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1],
                    'Interest_Rate': [2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
                    'Unemployment_Rate': [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
                    'Stock_Index_Price': [1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]
                    }

    df = pd.DataFrame(Stock_Market,columns=['Year','Month','Interest_Rate','Unemployment_Rate','Stock_Index_Price'])

    plt.scatter(df['Interest_Rate'], df['Stock_Index_Price'], color='red')
    plt.title('Stock Index Price Vs Interest Rate', fontsize=14)
    plt.xlabel('Interest Rate', fontsize=14)
    plt.ylabel('Stock Index Price', fontsize=14)
    plt.grid(True)
    plt.show()

    plt.scatter(df['Unemployment_Rate'], df['Stock_Index_Price'], color='green')
    plt.title('Stock Index Price Vs Unemployment Rate', fontsize=14)
    plt.xlabel('Unemployment Rate', fontsize=14)
    plt.ylabel('Stock Index Price', fontsize=14)
    plt.grid(True)
    plt.show()

    X = df[['Interest_Rate','Unemployment_Rate']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
    Y = df['Stock_Index_Price']

    # with sklearn
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)

    # prediction with sklearn
    New_Interest_Rate = 2.75
    New_Unemployment_Rate = 5.3
    print ('Predicted Stock Index Price: \n', regr.predict([[New_Interest_Rate ,New_Unemployment_Rate]]))

    # with statsmodels
    X = sm.add_constant(X) # adding a constant

    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X)

    print_model = model.summary()
    print(print_model)

    #### ALL THE REST OF THIS IS JUST THE GUI
    # # tkinter GUI
    # root= tk.Tk()
    #
    # canvas1 = tk.Canvas(root, width = 500, height = 300)
    # canvas1.pack()
    #
    # # with sklearn
    # Intercept_result = ('Intercept: ', regr.intercept_)
    # label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')
    # canvas1.create_window(260, 220, window=label_Intercept)
    #
    # # with sklearn
    # Coefficients_result  = ('Coefficients: ', regr.coef_)
    # label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')
    # canvas1.create_window(260, 240, window=label_Coefficients)
    #
    # # New_Interest_Rate label and input box
    # label1 = tk.Label(root, text='Type Interest Rate: ')
    # canvas1.create_window(100, 100, window=label1)
    #
    # entry1 = tk.Entry (root) # create 1st entry box
    # canvas1.create_window(270, 100, window=entry1)
    #
    # # New_Unemployment_Rate label and input box
    # label2 = tk.Label(root, text=' Type Unemployment Rate: ')
    # canvas1.create_window(120, 120, window=label2)
    #
    # entry2 = tk.Entry (root) # create 2nd entry box
    # canvas1.create_window(270, 120, window=entry2)
    #
    # def values():
    #     global New_Interest_Rate #our 1st input variable
    #     New_Interest_Rate = float(entry1.get())
    #
    #     global New_Unemployment_Rate #our 2nd input variable
    #     New_Unemployment_Rate = float(entry2.get())
    #
    #     Prediction_result  = ('Predicted Stock Index Price: ', regr.predict([[New_Interest_Rate ,New_Unemployment_Rate]]))
    #     label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    #     canvas1.create_window(260, 280, window=label_Prediction)
    #
    #     button1 = tk.Button (root, text='Predict Stock Index Price',command=values, bg='orange') # button to call the 'values' command above
    #     canvas1.create_window(270, 150, window=button1)
    #
    # #plot 1st scatter
    # figure3 = plt.Figure(figsize=(5,4), dpi=100)
    # ax3 = figure3.add_subplot(111)
    # ax3.scatter(df['Interest_Rate'].astype(float),df['Stock_Index_Price'].astype(float), color = 'r')
    # scatter3 = FigureCanvasTkAgg(figure3, root)
    # scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
    # ax3.legend(['Stock_Index_Price'])
    # ax3.set_xlabel('Interest Rate')
    # ax3.set_title('Interest Rate Vs. Stock Index Price')
    #
    # #plot 2nd scatter
    # figure4 = plt.Figure(figsize=(5,4), dpi=100)
    # ax4 = figure4.add_subplot(111)
    # ax4.scatter(df['Unemployment_Rate'].astype(float),df['Stock_Index_Price'].astype(float), color = 'g')
    # scatter4 = FigureCanvasTkAgg(figure4, root)
    # scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
    # ax4.legend(['Stock_Index_Price'])
    # ax4.set_xlabel('Unemployment_Rate')
    # ax4.set_title('Unemployment_Rate Vs. Stock Index Price')
    #
    # root.mainloop()

    import pdb; pdb.set_trace()
    print 'here'

def mike_linear_regression():
    import pandas as pd
    from pandas_datareader import data as web
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    import talib as ta

    df = web.get_data_yahoo('SPY', start='1993-01-01', end='2020-05-22')
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close']]

    gary_short = 8
    gary_long = 13
    gary_smoothing = 5
    n=10
    training_fraction = 0.8
    split = int(training_fraction * len(df))

    df['open_shift'] = df['Open'].shift(1)
    df['high_shift'] = df['High'].shift(1)
    df['low_shift'] = df['Low'].shift(1)
    # TODO: Note that this is using only adjusted data
    df['close_shift'] = df['Adj Close'].shift(1)

    # All indicators will use the already-shifted values above
    df['RSI'] = ta.RSI(np.array(df['close_shift']), timeperiod = n)
    df['SMA'] = df['close_shift'].rolling(window = n).mean()
    df['SMA_normalized'] = df['close_shift'].rolling(window = n).mean() / df['close_shift']
    df['Corr'] = df['SMA'].rolling(window=n).corr(df['close_shift'])
    df['SAR'] = ta.SAR(np.array(df['high_shift']), np.array(df['low_shift']), 0.2, 0.2)
    df['ADX'] = ta.ADX(np.array(df['high_shift']), np.array(df['low_shift']), np.array(df['close_shift']), timeperiod=n)

    # All indicators will use the already-shifted values above
    df['RSI_short'] = ta.RSI(np.array(df['close_shift']), timeperiod = gary_short)
    df['RSI_long'] = ta.RSI(np.array(df['close_shift']), timeperiod = gary_long)
    df['RSI_50'] = ta.RSI(np.array(df['close_shift']), timeperiod = 50)
    df['RSI_200'] = ta.RSI(np.array(df['close_shift']), timeperiod = 200)
    df['SMA_short'] = df['close_shift'].rolling(window = gary_short).mean()
    df['SMA_long'] = df['close_shift'].rolling(window = gary_long).mean()
    df['SMA_50'] = df['close_shift'].rolling(window = 50).mean()
    df['SMA_200'] = df['close_shift'].rolling(window = 200).mean()
    # df['SMA_normalized'] = df['close_shift'].rolling(window = gary_short).mean() / df['close_shift']
    df['Corr_short'] = df['SMA'].rolling(window=gary_short).corr(df['close_shift'])
    df['Corr_long'] = df['SMA'].rolling(window=gary_long).corr(df['close_shift'])
    df['Corr_50'] = df['SMA'].rolling(window=50).corr(df['close_shift'])
    df['Corr_200'] = df['SMA'].rolling(window=200).corr(df['close_shift'])
    df['SAR'] = ta.SAR(np.array(df['high_shift']), np.array(df['low_shift']), 0.2, 0.2)
    df['ADX'] = ta.ADX(np.array(df['high_shift']), np.array(df['low_shift']), np.array(df['close_shift']), timeperiod=gary_short)
    df['UpperBBand'], df['MiddleBBand'], df['LowerBBand'] = ta.BBANDS(np.array(df['close_shift']), timeperiod = 21)

    # TODO: This seems it should be using the close_shift, but the example doesn't show that
    df['Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    ### df['Return'] = df['Adj Close'] / df['Adj Close'].shift(1)

    df['Volatility_short'] = df['Return'].rolling(window = gary_short).std()
    df['Volatility_21'] = df['Return'].rolling(window = 21).std()
    df['Volatility_50'] = df['Return'].rolling(window = 50).std()
    df['Volatility_200'] = df['Return'].rolling(window = 200).std()

    df = df.dropna()

    # Data looks very clean to this point.

    standard_scaler = StandardScaler()

    # drop excludes certain columns. Don't need to exclude Date here because it's not a "column" it's the index
    new_columns = df.columns.drop(['Open', 'High', 'Low', 'SMA', 'SAR', 'ADX'])
    # new_columns = df.columns.drop(['Open', 'High', 'Low'])
    df[new_columns] = standard_scaler.fit_transform(df[new_columns])
    df['Signal'] = 0
    df.loc[df['Return'] > 0, 'Signal'] = 1
    df.loc[df['Return'] < 0, 'Signal'] = -1
    ### df.loc[df['Return'] > 1, 'Signal'] = 1
    ### df.loc[df['Return'] < 1, 'Signal'] = -1

    ### X = df[['SMA_normalized']]
    X = df[[
            'close_shift',
            # 'RSI_short', 'RSI_long', 'RSI_50', 'RSI_200',
            # 'RSI_200',
            # 'SMA_short', 'SMA_long', 'SMA_50', 'SMA_200',
            'SMA_200',
            # 'Corr_short', 'Corr_long', 'Corr_50', 'Corr_200',
            # 'SAR', 'ADX',
            # 'UpperBBand', 'MiddleBBand', 'LowerBBand',
            # 'Volatility_short', 'Volatility_21', 'Volatility_50', 'Volatility_200'
            # 'Volatility_200',
            ]]
    X = standard_scaler.fit_transform(X)
    Y = df['Close']
    ### Y = df['Signal']

    lin_reg = LinearRegression()
    lin_reg.fit(X[:split], Y[:split])

    print 'Intercept: ', lin_reg.intercept_
    print 'Coeffs: ', lin_reg.coef_

    p_data = len(X) - split

    df['Pred_Signal'] = 0
    df.iloc[-p_data:,df.columns.get_loc('Pred_Signal')] = lin_reg.predict(X[split:])
    df['str_ret'] = df['Pred_Signal'] * df['Return'].shift(-1)

    # FIXME: The strategy returns end up being greater than the actual returns in SPY around this time. Maybe scaling
    # is off???

    df['strategy_cu_return'] = 0.
    df['market_cu_return'] = 0.
    df.iloc[-p_data:, df.columns.get_loc('strategy_cu_return')] = np.nancumsum(df['str_ret'][-p_data:])
    df.iloc[-p_data:, df.columns.get_loc('market_cu_return')] = np.nancumsum(df['Return'][-p_data:])

    ### df.iloc[-p_data:, df.columns.get_loc('strategy_cu_return')] = np.nancumprod(df['str_ret'][-p_data:])
    ### df.iloc[-p_data:, df.columns.get_loc('market_cu_return')] = np.nancumprod(df['Return'][-p_data:])

    Sharpe = (df['strategy_cu_return'][-1] - df['market_cu_return'][-1]) / np.nanstd(df['strategy_cu_return'][-p_data:])

    plt.plot(df['strategy_cu_return'][-p_data:], color='g', label='Strategy Cum. Return')
    plt.plot(df['market_cu_return'][-p_data:], color='r', label='Market Price')
    plt.plot(df['Pred_Signal'][-p_data:], color='b', label='Pred Signal')
    plt.figtext(0.14, 0.9, s='Sharpe Ratio: %.2f' %Sharpe)
    plt.legend(loc='best')
    plt.show()


    import pdb; pdb.set_trace()
    print 'finish'


    # TODO: using the log returns allows us to more easily add up the returns at the end, but the results are strange
    # If we try to use the CAGR returns, these are going to go thru fit_transform() which will change them, and we
    # also would need a new way to multiple them by the Pred_Signal for the str_ret




if __name__ == '__main__':
    # do_quandl_demo()
    # do_larger_regression_demo()
    # multiple_linear_regression()
    mike_linear_regression()
