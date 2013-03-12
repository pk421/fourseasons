from data.redis import manage_redis
import os
import requests
# import threading
import logging
import logging.handlers
import time
# import Queue
import datetime
import multiprocessing
from multi_proc import q_runner, worker1


def get_yahoo_data(input, output):
    # update_check = kwargs['update_check']
    # logger = kwargs['log']
    update_check = False

    start_year = '1950'
    start_month = '1'
    start_day = '1'
    current_year = str(datetime.datetime.today().year)
    current_month = str(datetime.datetime.today().month)
    current_day = str(datetime.datetime.today().day)

    for c in iter(input.get, 'STOP'):
        s = c
        print s
        # s = q.get()
        # print s, "\t", q.qsize()
        file_path = "/home/wilmott/Desktop/fourseasons/fourseasons/tmp/" + s + ".csv"
        #logger.debug(s + "\tbefore if" + str(os.path.exists(file_path)) + str(update_check))
        if (os.path.exists(file_path) == False) or update_check == False:
            query_url = 'http://table.finance.yahoo.com/table.csv?s=' + s + '&a=' + start_month + \
                                                                          '&b=' + start_day + \
                                                                          '&c=' + start_year + \
                                                                          '&d=' + current_month + \
                                                                          '&e=' + current_day + \
                                                                          '&f=' + current_year + \
                                                                          '&ignore=.csv'
            #logger.debug(s + "\tbefore request")
            test_file = '\texception in urlopen'
            try:
                test_file = requests.get(query_url, timeout=2)
                if test_file.status_code != 200:
                    test_file = requests.get(query_url, timeout=2)
            except:
                try:
                    test_file = requests.get(query_url, timeout=2)
                except:
                    try:
                        test_file = requests.get(query_url, timeout=10)
                    except:
                        # logger.info(s + ',http request failed 3 attempts')
                        ####################
                        # q.task_done()
                        continue
            #logger.debug(s + "\tafter request, will write to file")
            if test_file.status_code == 200:
                fout = open(file_path, 'w')
                fout.write(test_file.text)
                fout.close()
                # q.task_done()
                continue
        else:
            #logger.debug(s + '\tskipping, file already present')
            # q.task_done()
            continue
        #catch anything that might have gotten thru other statements...this is to debug
        # q.task_done()
    return

#os.system("curl --silent 'http://download.finance.yahoo.com/d/quotes.csv?s=SLV&f=l' > tmp/SLV.csv")
#http://table.finance.yahoo.com/table.csv?a=["fmonth","fmonth"]&b=["fday","fday"]&c=["fyear","fyear"]&d=["tmonth","tmonth"]&e=["tday","tday"]&f=["tyear","tyear"]&s=["ticker", "ticker"]&y=0&g=["per","per"]&ignore=.csv
#http://table.finance.yahoo.com/table.csv?a=1&b=1&c=1900&d=2&e=2&f=2020&s=AMMD&ignore=.csv

def multithread_yahoo_download(list_to_download='large_universe.csv', thread_count=1, update_check=False,
                               new_only=False):
    #kill off previous processes:
    #os.system('kill -9 $(lsof src.data_retriever.log*)')
    logger = logging.getLogger(__name__)
    handler = logging.handlers.RotatingFileHandler('/home/wilmott/Desktop/fourseasons/fourseasons/log/' + \
                                                   __name__ + '.log', maxBytes=1024000, backupCount=5)
    formatter = logging.Formatter('%(asctime)s: %(threadName)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('Starting Main Thread')

    stock_list = open('/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/' + list_to_download, 'r')
    symbols = stock_list.read().rstrip().split('\n')
    for index, s in enumerate(symbols):
        symbols[index] = s.strip('\r')
    stock_list.close()

    if new_only == True:
        symbols_with_data = extract_symbols_with_historical_data()
        symbols_without_data = []
        for s in symbols:
            if (s in symbols_with_data) == False:
                symbols_without_data.append(s)
        symbols = symbols_without_data
    
    #used to test a single symbol
    #symbols = ['A', 'AA', 'AAPL', 'F', 'X', 'GOOG', 'bogus_symbol']
    #symbols = ['MEE']
   
    # q = multiprocessing.Queue()
    # for s in symbols:
    #     q.put(s)

    print "Number of symbols to fetch: ", len(symbols)

    for d in range(thread_count):
        #logger.debug(str(symbols.index(s)) + ' if \t' + s + '\t' + str(len(threading.enumerate())))
        # d = threading.Thread(name=('get_yahoo_data_' + str(d)), target=get_yahoo_data, \
        #                      args=[queue], kwargs={'log':logger, 'update_check':update_check})
        # d.setDaemon(True)
        # d.start()
        

        # p = multiprocessing.Process(name=('get_yahoo_data_' + str(d)), target=get_yahoo_data, \
        #                      args=[q], kwargs={'log':logger, 'update_check':update_check})
        # p.daemon = True
        # p.start()
        pass

    r1 = q_runner(thread_count, symbols, get_yahoo_data)

    # queue.join()


    # pool = multiprocessing.Pool(thread_count)
    # for s in symbols:
    #     pool.apply_async(get_yahoo_data(args=[queue], kwargs={'log':logger, 'update_check':update_check})

    logger.debug("Ending Main Thread\n\n\n")
    handler.close()
    return

def extract_symbols_with_historical_data(search_in='/home/wilmott/Desktop/fourseasons/fourseasons/tmp/'):
    """
    This will search the search_in dir and return a list of symbols that already contain csv data.
    """

    search_in = '/home/wilmott/Desktop/fourseasons/fourseasons/tmp/'
    symbols = []

    for csv_item in filter(lambda x: '.csv' in x, os.listdir(search_in)):
        n = csv_item.rsplit('.csv')
        symbols.append(str(n[0]))

    print "# of symbols with existing data: ", len(symbols)
    symbols.sort()
    
    fout = open('/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/symbols_with_historical_data.csv', 'w')
    for s in symbols:
        fout.write(s + '\n')
    fout.close()

    return symbols


def load_redis(stock_list='do_all', db_number=15, file_location='data/test/'):
    start_time = datetime.datetime.now()

    base_path = '/home/wilmott/Desktop/fourseasons/fourseasons/'
    file_path = base_path + file_location + 'data/'
    list_path = base_path + 'data/stock_lists/'
    file_path = '/home/wilmott/Desktop/fourseasons/fourseasons/tmp/'
    symbols = []

    #if no symbol is specified, do it for all
    if stock_list == 'do_all':
        for csv_item in filter(lambda x: '.csv' in x, os.listdir(file_path)):
            if csv_item == 'data_validation_results.csv.info':
                continue
            n = csv_item.rsplit('.csv')
            symbols.append(n[0])
    else:
        symbols = open(list_path + stock_list, 'r').read().split()
        
    symbols.sort()

    #symbols = ['A', 'AAPL']
    
    validation_file = ''
    failed_symbols = []
    no_data_symbols = []
    
    for k, symbol in enumerate(symbols):
        try:
            all_data = open(file_path + symbol + '.csv', 'r').read().rstrip()
        except:
            print symbol, "\t not found in csv database"
            no_data_symbols.append(symbol)
            continue
            
        days = all_data.split('\n')

        #this removes the header title information
        if len(days) > 0:
            del days[0]

        stock_price_set = []
        for day in days:
            day_items = day.split(',')
            day_dict = {}
            try:
                day_dict['Symbol'] = str(symbol)
                day_dict['Date'] = str(day_items[0])
                day_dict['Open'] = float(day_items[1])
                day_dict['High'] = float(day_items[2])
                day_dict['Low'] = float(day_items[3])
                day_dict['Close'] = float(day_items[4])
                day_dict['Volume'] = float(day_items[5])
                day_dict['AdjClose'] = float(day_items[6])
            except:
                print symbol, '\n', day_items
                exit(1)

            stock_price_set.append(day_dict)

        validation_results = validate_data(stock_price_set)

        if validation_results is not True:
            validation_file = validation_file + validation_results
            failed_symbols.append(symbol)
            #print failed_symbols
            print symbol, "\t failed at least one validation test"
            continue
        manage_redis.fill_redis(stock_price_set, db_number=db_number)

        current_time = datetime.datetime.now() - start_time
        print k, ' of ', len(symbols), '\t', current_time, '\t', symbol, '\tinto redis'
        #stock_db[symbol] = stock_price_set

    fout = open(base_path + file_location + 'failed_validation_results.csv.info', 'w')
    fout.write(validation_file)
    fout.close()

    end_time = datetime.datetime.now()
    time_required = end_time - start_time
    print "\nFailed Validation Symbols: ", failed_symbols
    print "\nNumber of validation failures: ", len(failed_symbols)
    print "\nNo Data Symbols: ", no_data_symbols
    print "\nNumber of no data failures: ", len(no_data_symbols)
    print "\nTotal Loaded: ", (len(symbols) - len(failed_symbols) - len(no_data_symbols))
    print "Time Required: ", time_required

    return

def validate_data(stock_price_set):
    """Takes in a stock_price_set list of dictionaries representing the data and performs simple validation on it,
    noting any issues found in a csv format.
    """
    
    results = ''

    for day in stock_price_set:

        if day['High'] < day['Open']:
            results = results + day['Symbol'] + ',' + day['Date'] + ',' + 'High < Open\n'
        if day['High'] < day['Close']:
            results = results + day['Symbol'] + ',' + day['Date'] + ',' + 'High < Close\n'
        if day['High'] < day['Low']:
            results = results + day['Symbol'] + ',' + day['Date'] + ',' + 'High < Low\n'

        if day['Low'] > day['Open']:
            results = results + day['Symbol'] + ',' + day['Date'] + ',' + 'Low > Open\n'
        if day['Low'] > day['Close']:
            results = results + day['Symbol'] + ',' + day['Date'] + ',' + 'Low > Close\n'
        if day['Low'] > day['Low']:
            results = results + day['Symbol'] + ',' + day['Date'] + ',' + 'Low > Low\n'

        if day['Open'] == 0 or day['High'] == 0 or day['Low'] == 0 or day['Close'] == 0:
            results = results + day['Symbol'] + ',' + day['Date'] + 'Found a zero value in prices.\n'


    if results is not '':
        return results
    else:
        return True

def read_redis(stock='all_stocks', db_number=15, to_disk=True):
    """
    list_of_stocks, the object returned by read_redis, will be structured as follows:
    A list representing the data of many stocks. Each stock item in the list is itself a list of "day_dicts", where
    each day of data is represented as a dictionary.
    """

    if to_disk:
        #Assumes you are dumping real time data for now, since this data can't be re-downloaded from Yahoo
        manage_redis.read_realtime_data(db_number=db_number)
        return

    else:
        #This function assumes db=0, which is the db used for all of the Daily historical data, must call w/list arg!!
        list_of_stocks = manage_redis.read_redis([stock])
        return list_of_stocks
    




class PriceSet(object):

    def __init__(self):
        self.trading_days = []

    def fill_trading_days(self, input_data):
        pass


    
    
