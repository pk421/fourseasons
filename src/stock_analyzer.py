from src.data_retriever import read_redis
import numpy as np
import toolsx as tools

def run_stock_analyzer():

    location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/list_sp_500.csv'
    in_file = open(location, 'r')
    stock_list = in_file.read().split('\n')
    in_file.close()

    stock_list = ['AAPL', 'GOOG', 'GLD', 'SLV', 'NEM', 'ABX', 'XOM', 'CVX']
    stock_list = ['GLD']

    for stock in stock_list:
        #read_redis returns a list of stocks, each is a list of time slices, each time slice is a dictionary of prices
        #since here we are only calling read_redis with 1 stock at a time, we will always pull the first (only) element
        #of the array.
        stock_data = read_redis(stock=stock, db_number=14, to_disk=False)[0]

        len_data = len(stock_data)
        simple_price_data = np.zeros(len_data)

        for x in xrange(0, len_data):
            simple_price_data[x] = stock_data[x]['AdjClose']

        v1 = 21
        volatility = tools.volatility_bs_annualized(simple_price_data, v1=v1)

        max_t = 0
        max_under_vol_return = 0
        max_over_vol_return = 0

        test_values = []

        for x in xrange(0, 100000):
            test_values.append(x * 0.0000025)

        for i, t in enumerate(test_values):
            under_vol_returns = 1
            under_vol_days = 0
            over_vol_returns = 1
            over_vol_days = 0

            for k, v in enumerate(volatility):
                
                if k >= v1 and k <= (len_data - 1):
                    today_return = np.log(np.abs(simple_price_data[k] / simple_price_data[k-1]))
                    if v <= t:
                        under_vol_returns = under_vol_returns * (1 + today_return)
                        under_vol_days += 1
                    elif v > t:
                        over_vol_returns = under_vol_returns * (1 + today_return)
                        over_vol_days += 1

                    #print today_return


            if under_vol_returns > max_under_vol_return:
                max_over_vol_return = over_vol_returns
                max_under_vol_return = under_vol_returns
                max_t = t

            if i % 100 == 0:
                print i, t, "\t", max_t, "\t", max_under_vol_return, "\t", under_vol_returns

        #print max_t,max_over_vol_return, max_under_vol_return

        for k, v in enumerate(volatility):
            #print stock_data[k]['Date'], "\t", k, "\t", round(v, 4)
            pass
        
        
#        for x in xrange(0,len(stock_data)):
#            today = stock_data[x]
#            print today['Symbol'], "\t", today['Date'], "\t", today['AdjClose']

#        for today in stock_data[0]:
#            print today['Symbol'], "\t", today['Date'], "\t", today['AdjClose']


    