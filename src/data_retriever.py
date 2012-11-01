from data.redis import manage_redis
import os
import urllib2
import threading
import logging
import logging.handlers
import time

def get_yahoo_data(s, **kwargs):

    update_check = kwargs['update_check']
    logger = kwargs['log']
    logger.debug(s + 'Entering thread')
    file_path = "/home/wilmott/Desktop/fourseasons/fourseasons/tmp/" + s + ".csv"

    if (os.path.exists(file_path) == False) or os.path.getsize(file_path) < 3500 or update_check == False:
        post_request = "http://table.finance.yahoo.com/table.csv?s=" + s + "&a=1&b=1&c=1950&d=10e=29&f=2012&ignore=.csv"
        logger.debug(s + "\tbefore request")
        test_file = '\texception in urlopen'
        try:
            test_file = urllib2.urlopen(post_request, timeout=1).read()
        except:
            logger.debug(s + '\thttp request failed')
            logger.debug(s + test_file)
            return
        logger.debug(s + "\tafter request, will write to file")
        fout = open(file_path, 'w')
        fout.write(test_file)
        fout.close()
        return
    else:
        logger.debug(s + '\tskipping, file already present')
        return


#os.system("curl --silent 'http://download.finance.yahoo.com/d/quotes.csv?s=SLV&f=l' > tmp/SLV.csv")
#http://table.finance.yahoo.com/table.csv?a=["fmonth","fmonth"]&b=["fday","fday"]&c=["fyear","fyear"]&d=["tmonth","tmonth"]&e=["tday","tday"]&f=["tyear","tyear"]&s=["ticker", "ticker"]&y=0&g=["per","per"]&ignore=.csv

def multithread_yahoo_download(list_to_download='300B_1M.csv', thread_count = 10, update_check = True):
    logger = logging.getLogger(__name__)
    handler = logging.handlers.RotatingFileHandler('/home/wilmott/Desktop/fourseasons/fourseasons/log/' + \
                                                   __name__ + '.log', maxBytes=1024000, backupCount=5)
    formatter = logging.Formatter('%(asctime)s %(threadName)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('Starting Main Thread')

    stock_list = open('/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/' + list_to_download, 'r')
    symbols = stock_list.read().rstrip().split('\n')
    for index, s in enumerate(symbols):
        symbols[index] = s.strip('\r')
    stock_list.close()
    
    #used to test a single symbol
    #symbols = ['A']

    print "Number of symbols to fetch: ", len(symbols)

    #main_thread = threading.currentThread()
    for s in symbols:
        print s
        if len(threading.enumerate()) < (thread_count + 1):
            logger.debug(str(symbols.index(s)) + ' if \t' + s + '\t' + str(len(threading.enumerate())))
            d = threading.Thread(name='get_yahoo_data', target=get_yahoo_data, \
                                 args=[s], kwargs={'log':logger, 'update_check':update_check})
            d.setDaemon(True)
            d.start()


        else:
            logger.debug(str(symbols.index(s)) + ' else \t' + s + '\t' + str(len(threading.enumerate())))
            while (len(threading.enumerate()) >= (thread_count + 1)):
                #time.sleep(0.1)
                d.join()
            d = threading.Thread(name='get_yahoo_data', target=get_yahoo_data, \
                                 args=[s], kwargs={'log':logger, 'update_check':update_check})
            d.setDaemon(True)
            d.start()

    d.join()
    logger.debug("Ending Main Thread\n\n\n")

def extract_symbols_with_historical_data(search_in = '/home/wilmott/Desktop/fourseasons/fourseasons/tmp/'):

    search_in = '/home/wilmott/Desktop/fourseasons/fourseasons/tmp/'
    symbols = []

    for csv_item in filter(lambda x: '.csv' in x, os.listdir(search_in)):
        n = csv_item.rsplit('.csv')
        symbols.append(str(n[0] + '\n'))

    print type(symbols)
    print len(symbols)
    symbols.sort()
    
    fout = open(search_in + 'symbols_with_historical_data.csv', 'w')
    for s in symbols:
        fout.write(s)
    fout.close()

def objectify_data():

#    file_path = '/home/wilmott/Desktop/fourseasons/fourseasons/data/daily_price_data/'
    file_path = '/home/wilmott/Desktop/fourseasons/fourseasons/data/daily_price_data/test/'
    symbols = []

    for csv_item in filter(lambda x: '.csv' in x, os.listdir(file_path)):
        n = csv_item.rsplit('.csv')
        symbols.append(n[0])
    symbols.sort()

    symbols = ['AAPL']

    #stock_db = {}
    
    for symbol in symbols:
        print symbol
        all_data = open(file_path + symbol + '.csv', 'r').read().rstrip()

        days = all_data.split('\n')

        #this removes the header title information
        if len(days) > 0:
            del days[0]

        stock_price_set = []
        for day in days:
            day_items = day.split(',')
            day_dict = {}

            day_dict['Date'] = day_items[0]
            day_dict['Open'] = day_items[1]
            day_dict['High'] = day_items[2]
            day_dict['Low'] = day_items[3]
            day_dict['Close'] = day_items[4]
            day_dict['Volume'] = day_items[5]
            day_dict['AdjClose'] = day_items[6]

            stock_price_set.append(day_dict)

        manage_redis.fill_redis(stock_price_set)
        #stock_db[symbol] = stock_price_set

    #print "Now Sleeping For 15 Seconds"
    #time.sleep(15)


class PriceSet(object):

    def __init__(self):
        self.trading_days = []

    def fill_trading_days(self, input_data):
        pass




    
    
