
def fill_redis (stock_price_set):

    for day in stock_price_set:
        print len(stock_price_set), day['Symbol'], day['Date']
        #print item
        