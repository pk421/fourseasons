from data.redis import manage_redis
import os
import requests
import threading
import logging
import logging.handlers
import time
import Queue
import datetime

from util.profile import profile

"""
Good data sites:
ETF/ETN list in csv format, frequently updated: http://masterdata.com/HelpFiles/ETF_List.htm
"""
def get_yahoo_data(queue, **kwargs):
    update_check = kwargs['update_check']
    logger = kwargs['log']
    store_location = kwargs['store_location']

    start_year = '1970'
    start_month = '1'
    start_day = '1'
    current_year = str(datetime.datetime.today().year)
    current_month = str(datetime.datetime.today().month)
    current_day = str(datetime.datetime.today().day)
    start_date = start_year + '-' + start_month + '-' + start_day
    current_date = current_year + '-' + current_month + '-' + current_day

    request_string = 'https://api.tiingo.com/tiingo/daily/' + 'idu' + '/prices?startDate=' + start_date +  '&endDate=' + current_date
    headers = {'Content-Type': 'application/json', 'Authorization' : 'Token ec2bff038efec4926fd3673b8f17aeeb6525227f'}
    test_file = requests.get(request_string, headers=headers, timeout=2)

    while True:
        s = queue.get().lower()

        # If we only want to update with the latest data, we change the dates that we use, then later, we merge in the
        # existing redis data
        # if kwargs['merge_latest_data']:
        #     existing_data, start_date = get_merged_start_date(s)


        print s, "\t", queue.qsize()
        ### file_path = "/home/wilmott/Desktop/fourseasons/fourseasons/tmp/" + s + ".csv"
        file_path = "/home/wilmott/Desktop/fourseasons/fourseasons/" + store_location + s + ".csv"
        #logger.debug(s + "\tbefore if" + str(os.path.exists(file_path)) + str(update_check))
        start_unix_epoch = str(datetime.datetime(int(start_year),int(start_month),int(start_day),0,0).strftime('%s'))
        end_unix_epoch = str(datetime.datetime(int(current_year), int(current_month), int(current_day),0,0).strftime('%s'))

        # This is hacky. There is now a cookie involved, but it seems to last a long time. Go the "download csv" link
        # on yahoo's historical data page. Look at the link and find the tail of it "crumb=". That needs to be kept up
        # to date here with fresh queries that are done manually. This handles cases when the cookie expires. Note: The
        # yahoo finance query MUST BE DONE ON LINUX, NOT WINDOWS
        if (os.path.exists(file_path) == False) or update_check == False:
            # print "Doing query: ", s
#            query_url = 'http://table.finance.yahoo.com/table.csv?s=' + s + '&a=' + start_month + \
#                                                                           '&b=' + start_day + \
#                                                                           '&c=' + start_year + \
#                                                                           '&d=' + current_month + \
#                                                                           '&e=' + current_day + \
#                                                                           '&f=' + current_year + \
#                                                                           '&ignore=.csv'

            # query_url = 'http://ichart.finance.yahoo.com/table.csv?s=' + s + '&a=' + start_month + \
            #                                                               '&b=' + start_day + \
            #                                                               '&c=' + start_year + \
            #                                                               '&d=' + current_month + \
            #                                                               '&e=' + current_day + \
            #                                                               '&f=' + current_year + \
            #                                                               '&ignore=.csv'

            query_url = 'https://query1.finance.yahoo.com/v7/finance/download/' + s + '?period1=' + \
                                                                                start_unix_epoch + \
                                                                                '&period2=' + \
                                                                                end_unix_epoch + \
                                                                                '&interval=1d&events=history&crumb=oRQtY6jjCqs'


            # 5GB available: pilat.michael / patos / ______23 / ec2bff038efec4926fd3673b8f17aeeb6525227f
            # 2GB available: mcpilat / climbarok / ______23 / 682e2feeed3d495a6a68820c4bbb40ed5754ff5c

            headers = {'Content-Type': 'application/json', 'Authorization' : 'Token 682e2feeed3d495a6a68820c4bbb40ed5754ff5c'}
            # This request string just gives basic info
            # request_string = 'https://api.tiingo.com/tiingo/daily/' + s + '?token=682e2feeed3d495a6a68820c4bbb40ed5754ff5c'
            request_string = 'https://api.tiingo.com/tiingo/daily/' + s + '/prices?startDate=' + start_date +  '&endDate=' + current_date

            # print "Querying: ", request_string
            # logger.debug(s + "\tbefore request")
            try:
                # test_file = requests.get(query_url, timeout=2)
                test_file = requests.get(request_string, headers=headers, timeout=2)

                if test_file.status_code != 200:
                    test_file = requests.get(request_string, timeout=2)
            except:
                try:
                    test_file = requests.get(request_string, timeout=2)
                except:
                    try:
                        test_file = requests.get(request_string, timeout=10)
                    except:
                        logger.info(s + ',http request failed 3 attempts')
                        queue.task_done()
                        continue
            #logger.debug(s + "\tafter request, will write to file")
            if test_file.status_code == 200:
                try:
                    test_csv = get_csv_from_json(test_file.json(), s)
                except:
                    print "TEST: ", s, test_file
                    raise
                fout = open(file_path, 'w')
                fout.write(test_csv)
                fout.close()
                queue.task_done()
                # print "Wrote File: ", file_path
                continue
            else:
                print "ERROR: Bad HTTP Status Code: ", s, test_file.status_code
        else:
            logger.debug(s + '\tskipping, file already present')
            queue.task_done()
            continue
        #catch anything that might have gotten thru other statements...this is to debug
        queue.task_done()
    return

def get_csv_from_json(json_response, stock):
    header_line = 'Date,Open,High,Low,Close,Volume,AdjClose'

    body = ''
    for line_item in json_response:
        cleaned_date = line_item['date']
        cleaned_year = int(cleaned_date.split('-')[0])
        cleaned_month = int(cleaned_date.split('-')[1])
        cleaned_day = int(cleaned_date.split('-')[2].split('T')[0])
        cleaned_datetime = datetime.date(cleaned_year, cleaned_month, cleaned_day)
        cleaned_date_string = cleaned_datetime.strftime('%Y%m%d')

        try:
            # some of these are None for whatever the reason. We would have cleaned these up if we plugged in a zero,
            # but that's not desired. Instead, just break and prevent more recent dates from appearing in the string,
            # then this can get filtered out later
            open = round(line_item['open'], 2)
            high = round(line_item['open'], 2)
            low = round(line_item['open'], 2)
            close = round(line_item['open'], 2)
            volume = line_item['volume']
            adj_close = round(line_item['adjClose'], 2)
        except:
            print "LINE ITEM HERE: ", stock, line_item
            break

        line_string = '\n' + str(cleaned_date_string) + ',' + str(open) + ',' + str(high) + ',' + str(low) + ',' + str(close) + ',' + str(volume) + ',' + str(adj_close)
        body += line_string

    final_string = header_line + body
    return final_string

def get_merged_start_date(s):
    data = manage_redis.parse_fast_data(s, db_to_use=1)
    latest_date = data[-1]['Date'][0:4] + '-' + data[-1]['Date'][4:6] + '-' + data[-1]['Date'][6:8]
    return data, latest_date

#http://table.finance.yahoo.com/table.csv?a=["fmonth","fmonth"]&b=["fday","fday"]&c=["fyear","fyear"]&d=["tmonth","tmonth"]&e=["tday","tday"]&f=["tyear","tyear"]&s=["ticker", "ticker"]&y=0&g=["per","per"]&ignore=.csv
#http://table.finance.yahoo.com/table.csv?a=1&b=1&c=1900&d=2&e=2&f=2020&s=AMMD&ignore=.csv

def multithread_yahoo_download(list_to_download='large_universe.csv', thread_count=20, update_check=False,
                               new_only=False, store_location = 'tmp/', use_list = None):
    queue = Queue.Queue()
    #kill off previous processes:
    #os.system('kill -9 $(lsof src.data_retriever.log*)')
    logger = logging.getLogger(__name__)
    handler = logging.handlers.RotatingFileHandler('/home/wilmott/Desktop/fourseasons/fourseasons/log/' + \
                                                   __name__ + '.log', maxBytes=1024000, backupCount=5)
    formatter = logging.Formatter('%(asctime)s: %(threadName)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # logger.info('Starting Main Thread')

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
    
    if use_list:
        symbols = use_list
    #used to test a single symbol
    #symbols = ['A', 'AA', 'AAPL', 'F', 'X', 'GOOG', 'bogus_symbol']
    #symbols = ['MEE']
    for s in symbols:
        queue.put(s)
    # print "Number of symbols to fetch: ", len(symbols)

    for d in range(thread_count):
        #logger.debug(str(symbols.index(s)) + ' if \t' + s + '\t' + str(len(threading.enumerate())))
        d = threading.Thread(name=('get_yahoo_data_' + str(d)), target=get_yahoo_data, \
                             args=[queue], kwargs={'log':logger, 'update_check':update_check, \
                             'store_location':store_location})
        d.setDaemon(True)
        d.start()

    queue.join()

    # logger.debug("Ending Main Thread\n\n\n")
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


def load_redis(stock_list='do_all', db_number=99, file_location='tmp/', dict_size=2, use_list=None):
    start_time = datetime.datetime.now()

    base_path = '/home/wilmott/Desktop/fourseasons/fourseasons/'
    # file_path = base_path + file_location + 'data/'
    file_path = base_path + file_location
    list_path = base_path + 'data/stock_lists/'
    
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

    if use_list:
        symbols = use_list

    symbols.sort()

    # symbols = ['ABIO']
    
    validation_file = ''
    failed_symbols = []
    no_data_symbols = []
    data_in_csv_empty = []
    
    for k, symbol in enumerate(symbols):
        try:
            all_data = open(file_path + symbol + '.csv', 'r').read().rstrip()
        except:
            print symbol, "\t not found in csv database"
            no_data_symbols.append(symbol)
            continue
            
        days = all_data.split('\n')

        #this removes the header title information
        if len(days) > 0 and days[0] != '':
            del days[0]
        else:
            data_in_csv_empty.append(symbol)
            continue

        stock_price_set = []
        for day in days:
            day_items = day.split(',')
            day_dict = {}

            # By cutting out the dash, the dates can be compared as ints, much faster than using datetime lib
            simplified_date = str(day_items[0]).replace('-', '')

            try:
                day_dict['Symbol'] = str(symbol)
                day_dict['Date'] = str(simplified_date)
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
        
        ### manage_redis.fill_redis(stock_price_set, db_number=db_number, dict_size=dict_size)
        manage_redis.fill_fast_redis(stock_price_set, db_number=db_number, dict_size=dict_size)

        current_time = datetime.datetime.now() - start_time
        print k, ' of ', len(symbols), '\t', current_time, '\t', symbol, '\tinto redis'
        #stock_db[symbol] = stock_price_set

    # fout = open(base_path + file_location + 'failed_validation_results.csv.info', 'w')
    # fout.write(validation_file)
    # fout.close()

    end_time = datetime.datetime.now()
    time_required = end_time - start_time
    print "\nData In CSV Is Empty: ", data_in_csv_empty
    print "Failed Validation Symbols: ", failed_symbols
    print "Number of validation failures: ", len(failed_symbols)
    print "No Data Symbols: ", no_data_symbols
    # print "Number of no data failures: ", len(no_data_symbols)
    # print "Total Loaded: ", (len(symbols) - len(failed_symbols) - len(no_data_symbols))
    # print "Time Required: ", time_required

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

def read_redis(stock='all_stocks', db_number=99, dict_size=10, to_disk=False, start_date='-inf', end_date='+inf'):
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
        if not isinstance(stock, list):
            stock = [stock]
        list_of_stocks = manage_redis.read_redis(stock, db_number=db_number, dict_size=dict_size, start_date='-inf', \
                                                 end_date='+inf')
        return list_of_stocks


def fix_etf_list():
    """
    This simple script cleans up the format of the etf list that I have. It simplifies the list of things down to
    only symbols so that they can be used by this engine.
    """

    location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/etfs_etns_raw.csv'
    stock_list = open(location, 'r')
    symbols = stock_list.read().rstrip().split('\n')

    out_location = '/home/wilmott/Desktop/fourseasons/fourseasons/data/stock_lists/etfs_etns.csv'
    out_file = open(out_location, 'w')

    outstr = ''
    new_symbols = []
    for x in xrange(len(symbols)):
        # new_symbols.append(symbols[x].split(',')[1])
        # print symbols[x].split(',')[1]
        # out_file.write(symbols[x].split(',')[1])
        # out_file.write('\n')
        if symbols[x].split(',')[1] == 'Symbol':
            continue
        outstr += symbols[x].split(',')[1]
        outstr += '\n'
    out_file.write(outstr)
    out_file.close()
    return
    



    
class PriceSet(object):

    def __init__(self):
        self.trading_days = []

    def fill_trading_days(self, input_data):
        pass   
