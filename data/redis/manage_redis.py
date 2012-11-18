import redis

def flushall():
    redis_db = redis.StrictRedis(host='localhost', port=6379, db=0)
    redis_db.flushall()
    return

def fill_redis(stock_price_set):
    redis_db = redis.StrictRedis(host='localhost', port=6379, db=0)
    symbol = stock_price_set[0]['Symbol']
    redis_db.delete(symbol)

    pairs = {}
    for day in stock_price_set:
        #redis_db.zadd(day['Symbol'], pack_date(day['Date']), pack_data(day))
        pairs[pack_data(day)] = pack_date(day['Date'])

    redis_db.zadd('historical-D:' + day['Symbol'], **pairs)

def read_redis(stocks):
    redis_db = redis.StrictRedis(host='localhost', port=6379, db=0)

    list_of_stocks = []

    for s in stocks:
        all_days = []
        redis_symbol = "historical-D:" + s
        #raw = redis_db.zrangebyscore(s,'-inf','+inf')
        raw = redis_db.zrangebyscore(redis_symbol,'20121010','20121011')
        for line in raw:
            day_dict = {}
            print line
            for k, v in enumerate(line.split(',')):
                #print '************', k, v
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
                    day_dict['AdjClose'] = float(v)
                    continue
                if k == 5:
                    day_dict['Volume'] = float(v)
                    continue

            all_days.append(day_dict)

        list_of_stocks.append(all_days)

        #print all_days
    return list_of_stocks

def pack_date(in_date):
    """
        Packs a date and converts it to an int format to put in redis
    """
    out_date = ''
    for i in in_date.split('-'):
        out_date += i
    return int(out_date)

def pack_data(in_data):
    """
        Packs the other data into redis in the form of a csv of O, H, L, AC, Vol
    """

    serializable = [in_data['Date'], in_data['Open'], in_data['High'], in_data['Low'], in_data['AdjClose'],
                    in_data['Volume']]
    out_data = ','.join(str(p) for p in serializable)
    return out_data
        