from src.data_retriever import read_redis
import numpy as np

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

        print "Mean: ", np.mean(simple_price_data)
        print "Median: ", np.median(simple_price_data)
        
#        for x in xrange(0,len(stock_data)):
#            today = stock_data[x]
#            print today['Symbol'], "\t", today['Date'], "\t", today['AdjClose']

#        for today in stock_data[0]:
#            print today['Symbol'], "\t", today['Date'], "\t", today['AdjClose']


    