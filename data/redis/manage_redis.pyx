import redis
from util.memoize import memoize, MemoizeMutable

def flushdb(db_number=15):
    # redis_db = redis.StrictRedis(host='localhost', port=6379, db=db_number)
    # redis_db.flushdb()

   
    # redis_db = redis.StrictRedis(host='localhost', port=6379, db=1) 
    # items = redis_db.keys()
    # items_to_del = []
    # print len(items)
    
    # for i in items:
    #     if 'rt_01' in i:
    #         continue
    #     else:
    #         items_to_del.append(i)

    # print len(items_to_del)

    # for i in items_to_del:
    #     redis_db.delete(i)

    return

def read_redis(stocks, db_number=15, dict_size=10, start_date='-inf', end_date='+inf'):
    redis_db = redis.StrictRedis(host='localhost', port=6379, db=db_number)

    list_of_stocks = []

    for s in stocks:
        all_days = []
        redis_symbol = "historical-D:" + s
        # if you want to change start_date/end_date, it must be in format: YYYYMMDD, pushed in as a string
        raw = redis_db.zrangebyscore(redis_symbol,start_date,end_date)
        for line in raw:
            day_dict = {}
#            print line\
            if dict_size == 10:
                for k, v in enumerate(line.split(',')):
                    day_dict['Symbol'] = s
                    if k == 0:
                        day_dict['Date'] = v
                        continue
                    if k == 1:
                        day_dict['Open'] = float(v)
                        continue
                    if k == 2:
                        day_dict['High'] = float(v)
                        continue
                    if k == 3:
                        day_dict['Low'] = float(v)
                        continue
                    if k == 4:
                        day_dict['Close'] = float(v)
                        continue
                    if k == 5:
                        day_dict['Volume'] = float(v)
                        continue
                    if k == 6:
                        day_dict['AdjOpen'] = float(v)
                        continue
                    if k == 7:
                        day_dict['AdjHigh'] = float(v)
                        continue
                    if k == 8:
                        day_dict['AdjLow'] = float(v)
                        continue
                    if k == 9:
                        day_dict['AdjClose'] = float(v)
                        continue

            if dict_size == 2:
                for k, v in enumerate(line.split(',')):
                    day_dict['Symbol'] = s
                    if k == 0:
                        day_dict['Date'] = v
                        continue
                    if k == 1:
                        day_dict['AdjClose'] = float(v)
                        continue

            all_days.append(day_dict)
        list_of_stocks.append(all_days)

    return list_of_stocks

def read_realtime_data(db_number=15):

    redis_db = redis.StrictRedis(host='localhost', port=6379, db=db_number)
    redis_keys = redis_db.keys(pattern='*')

    for redis_symbol in redis_keys:
#        redis_symbol = 'realtime_1min:Gold_Spot'
        start_date = '-inf'
        end_date = '+inf'

        out_file = open('/home/wilmott/Desktop/fourseasons/fourseasons/data/from_redis/' + redis_symbol.split(':')[0] +\
                        '-' + redis_symbol.split(':')[1] + '.csv', 'w')

        raw = redis_db.zrangebyscore(redis_symbol, start_date, end_date)

        for line in raw:
            for k,v in enumerate(line.split(',')):
                #print k, v
                if k == 6 and '/' in v:
                    #This block deals with data that was retrieved after hours and contains a time of "day/month"
                    #print line
                    raw.remove(line)
                    break
            #print line
            out_file.write(str(line))
            out_file.write('\n')

        out_file.close()
        print "\n", redis_symbol
        print "Total Entries: ", len(raw)

    return

def fill_redis(stock_price_set, store_under='historical-D:', delete_old_data=True, db_number=15, dict_size=10):
    """
    In order to use zrangebyscore, this is a little backwards. Basically we are storing the data as the "key" and then
    storing the date as the "value." When the data is retrieved, the values (store dates) can be sorted, which enables
    us to retrieve the dates in a nicely sorted order.
    """
    redis_db = redis.StrictRedis(host='localhost', port=6379, db=db_number)
    symbol = stock_price_set[0]['Symbol']
    if delete_old_data:
        redis_db.delete(store_under + symbol)

    pairs = {}
    for day in stock_price_set:
        #redis_db.zadd(day['Symbol'], pack_date(day['Date']), pack_data(day))
        pairs[pack_data(day, dict_size)] = pack_date(day['Date'])

    
    redis_db.zadd(store_under + symbol, **pairs)

def pack_date(in_date):
    """
        Packs a date and converts it to an int format to put in redis. in_date is simply in format yyyy-mm-dd
    """
    out_date = ''
    for i in in_date.split('-'):
        out_date += i
    return int(out_date)

def pack_data(in_data, dict_size):
    """
        Packs the other data into redis in the form of a csv of O, H, L, AC, Vol
        See here for adjustment algo for OHL: http://trading.cheno.net/downloading-yahoo-finance-historical-data-with-python/
    """

    adj_factor = in_data['AdjClose'] / in_data['Close']
    in_data['AdjOpen'] = str(round((in_data['Open'] * adj_factor), 2))
    in_data['AdjHigh'] = str(round ((in_data['High'] * adj_factor), 2))
    in_data['AdjLow'] = str(round((in_data['Low'] * adj_factor), 2))

    if dict_size == 10:
        serializable = [in_data['Date'], in_data['Open'], in_data['High'], in_data['Low'], in_data['Close'],
                        in_data['Volume'], in_data['AdjOpen'], in_data['AdjHigh'], in_data['AdjLow'], in_data['AdjClose']]
    elif dict_size == 2:
         serializable = [in_data['Date'], in_data['AdjClose']]       
    out_data = ','.join(str(p) for p in serializable)
    return out_data


def fill_realtime_redis(price_set, store_under='historical-D:', delete_old_data=False, db_number=15):
    redis_db = redis.StrictRedis(host='localhost', port=6379, db=db_number)
    symbol = price_set['Symbol']
    if delete_old_data:
        redis_db.delete(store_under + symbol)

    pairs = {}
    pairs[pack_realtime_data(price_set)] = price_set['Timestamp']

    redis_db.zadd(store_under + symbol, **pairs)


def pack_realtime_data(in_data):
    """
    Packs real time data of format: Contract Date, Last, Previous, H, L, Change, % Change, Time
    """

    serializable = [in_data['Last'], in_data['Previous'], in_data['High'], in_data['Low'], in_data['Change'],
                    in_data['ChangePct'], in_data['Time'], in_data['Date']]
    
    out_data = ','.join(str(p) for p in serializable)
    return out_data

#@MemoizeMutable
def get_data(stock=None):
    if stock:
        return read_redis([stock], db_number=15, dict_size=10)[0]

def fill_fast_redis(stock_price_set, db_number=15, dict_size=10):
    """
    This is slow to load because it will load the items in into a "ZRangeByScore" type database, then read from there
    and push them into a "fast" style database, then delete the key, value from the ZRangeByScore db. It does extra
    work, but it basically allows it to simply access the zrangebyscore db using the same api and would make it easy
    to switch between the two methods of storage. Eventually, with more confidence, I think ZRangeByScore could be
    eliminated as long as we could SORT the keys before putting them into a set/get style.
    """
    redis_reader = redis.StrictRedis(host='localhost', port=6379, db=15)
    redis_writer = redis.StrictRedis(host='localhost', port=6379, db=db_number)

    stock_symbol = stock_price_set[0]['Symbol']
    # print "Symbol: ", stock_symbol

    # print stock_price_set
    fill_redis(stock_price_set, store_under='historical-D:', delete_old_data=True, db_number=15, dict_size=dict_size)
    # read_redis(stock_symbol, db_number=15, dict_size=dict_size, start_date='-inf', end_date='+inf')

    raw_read = get_data(stock_symbol)
    st_version = str(raw_read)
    # print st_version
    input_key = 'historical:fast:' + stock_symbol
    redis_writer.set(input_key, st_version)
    # print stock_symbol, " Loaded"

    redis_reader.flushdb()
    return
    

    # for x in xrange(0, len(stock_list)):
    #     stock_list[x] = stock_list[x].strip('\n')
    # # print stock_list

    # for stock in stock_list:
    #     raw_read = read_redis(stock, db_number=0, dict_size=10)[0]
    #     st_version = str(raw_read)
    #     # print st_version
    #     input_key = 'historical:fast:' + stock
    #     redis_writer.set(input_key, st_version)
    #     print stock, " Loaded"

@MemoizeMutable
# cdef list parse_data(str stock):
def parse_fast_data(stock):
    """
    This should be used when a massive raw string of stock data was dumped into redis as a value. This will parse out
    the csv and return a list of dictionaries. Note carefully: it was designed to parse out the data of the type stored
    in db 0 in redis, which is not raw csv, but actually contains the OHLC stings, etc.
    """

    # cdef str get_query, stock_data, day, e
    # cdef list output_data, all_days, keys, values, k
    # cdef struct todays_dict

    redis_db = redis.StrictRedis(host='localhost', port=6379, db=0)
    get_query = 'historical:fast:' + stock
    stock_data = redis_db.get(get_query)

    output_data = []

    try:
        all_days = stock_data.split('}, {')
        for day in all_days:
            day = day.strip('{}[]')

            keys = []
            values = []
            for e in day.split(', '):
                k = e.split(': ')
                keys.append(k[0].strip('\''))
                try:
                    values.append(float(k[1]))
                except:
                    # This handles situations when the key is Symbol or Date and cannot be a float
                    # print "\n\n\n****", stock, day, e, k
                    values.append(k[1].strip('\''))

            todays_dict = dict(zip(keys, values))
            output_data.append(todays_dict)
    except:
        return None

    return output_data