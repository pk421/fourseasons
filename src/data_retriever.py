import os
import urllib2
import threading
import logging

def get_yahoo_data(s, **kwargs):

    logger = kwargs['log']
    logger.debug('Entering thread for: ' + s)
    file_path = "/home/wilmott/Desktop/fourseasons/fourseasons/tmp/" + s + ".csv"
    
    try:
        if (os.path.exists(file_path) == False) or os.path.getsize(file_path) < 3500:
            post_request = "http://table.finance.yahoo.com/table.csv?s=" + s + "&a=1&b=1&c=1950&d=10e=29&f=2012&ignore=.csv"
            test_file = urllib2.urlopen(post_request).read()
            file_path = "/home/wilmott/Desktop/fourseasons/fourseasons/tmp/" + s + ".csv"

            fout = open("/home/wilmott/Desktop/fourseasons/fourseasons/tmp/" + s + ".csv", 'w')
            fout.write(test_file)
            fout.close()
        else:
            return
    except:
        return


#os.system("curl --silent 'http://download.finance.yahoo.com/d/quotes.csv?s=SLV&f=l' > tmp/SLV.csv")
#http://table.finance.yahoo.com/table.csv?a=["fmonth","fmonth"]&b=["fday","fday"]&c=["fyear","fyear"]&d=["tmonth","tmonth"]&e=["tday","tday"]&f=["tyear","tyear"]&s=["ticker", "ticker"]&y=0&g=["per","per"]&ignore=.csv

def multithread_yahoo(thread_count = 10):
    #This procedure will work after the DOS line endings have already been converted to LINUX. Use: sed -i 's/\r//' filename
    

    logger = logging.getLogger(__name__)
    handler = logging.FileHandler('/home/wilmott/Desktop/fourseasons/fourseasons/log/' + __name__ + '.log')
    logger.error('etst')
    formatter = logging.Formatter('%(asctime)s %(threadName)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    stock_list = open('/home/wilmott/Desktop/fourseasons/fourseasons/data/large_universe', 'r')
    symbols = stock_list.read().split('\n')
    stock_list.close()
    print "Number of symbols to fetch: ", len(symbols)

    main_thread = threading.currentThread()
    for s in symbols:
        if len(threading.enumerate()) <= (thread_count + 1):
            print "if \t", s, "\t", len(threading.enumerate())
            d = threading.Thread(name='get_yahoo_data', target=get_yahoo_data, args=[s], kwargs={'log':logger})
            d.setDaemon(True)
            d.start()

        else:
            print "else \t", s, "\t", len(threading.enumerate())
            d.join()
            d = threading.Thread(name='get_yahoo_data', target=get_yahoo_data, args=[s], kwargs={'log':logger})
            d.setDaemon(True)
            d.start()

    d.join()






