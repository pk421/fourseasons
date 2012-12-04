import redis

def flushdb(db_number=15):
    redis_db = redis.StrictRedis(host='localhost', port=6379, db=db_number)
    redis_db.flushdb()
    return

def read_redis(stocks, start_date='-inf', end_date='+inf'):
    redis_db = redis.StrictRedis(host='localhost', port=6379, db=0)

    list_of_stocks = []

    for s in stocks:
        all_days = []
        redis_symbol = "historical-D:" + s
        raw = redis_db.zrangebyscore(redis_symbol,start_date,end_date)
        for line in raw:
            day_dict = {}
#            print line
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
                    print line
                    raw.remove(line)
                    break
            print line
            out_file.write(str(line))
            out_file.write('\n')

        out_file.close()
        print "\n", redis_symbol
        print "Total Entries: ", len(raw)

    return

def fill_redis(stock_price_set, store_under='historical-D:', delete_old_data=True, db_number=15):
    redis_db = redis.StrictRedis(host='localhost', port=6379, db=db_number)
    symbol = stock_price_set[0]['Symbol']
    if delete_old_data:
        redis_db.delete(store_under + symbol)

    pairs = {}
    for day in stock_price_set:
        #redis_db.zadd(day['Symbol'], pack_date(day['Date']), pack_data(day))
        pairs[pack_data(day)] = pack_date(day['Date'])

    redis_db.zadd(store_under + symbol, **pairs)

def pack_date(in_date):
    """
        Packs a date and converts it to an int format to put in redis. in_date is simply in format yyyy-mm-dd
    """
    out_date = ''
    for i in in_date.split('-'):
        out_date += i
    return int(out_date)

def pack_data(in_data):
    """
        Packs the other data into redis in the form of a csv of O, H, L, AC, Vol
        See here for adjustment algo for OHL: http://trading.cheno.net/downloading-yahoo-finance-historical-data-with-python/
    """

    adj_factor = in_data['AdjClose'] / in_data['Close']
    in_data['AdjOpen'] = str(round((in_data['Open'] * adj_factor), 2))
    in_data['AdjHigh'] = str(round ((in_data['High'] * adj_factor), 2))
    in_data['AdjLow'] = str(round((in_data['Low'] * adj_factor), 2))

    serializable = [in_data['Date'], in_data['Open'], in_data['High'], in_data['Low'], in_data['Close'],
                    in_data['Volume'], in_data['AdjOpen'], in_data['AdjHigh'], in_data['AdjLow'], in_data['AdjClose']]
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