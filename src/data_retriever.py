import os
import urllib2
import threading
import logging
import time

def get_yahoo_data():
    #This procedure will work after the DOS line endings have already been converted to LINUX. Use: sed -i 's/\r//' filename
    stock_list = open('/home/wilmott/Desktop/fourseasons/fourseasons/data/large_universe', 'r')
    symbols = stock_list.read().split('\n')
    stock_list.close()
    print"Number of symbols to fetch: ", len(symbols)

    for s in symbols:
        print s
        #single_quote_request = "curl --silent 'http://download.finance.yahoo.com/d/quotes.csv?s=" + s + "&f=l' > tmp/" + s + ".csv"
        file_path = "/home/wilmott/Desktop/fourseasons/fourseasons/tmp/" + s + ".csv"
        
        try:
            if (os.path.exists(file_path) == False) or os.path.getsize(file_path) < 3500:
                post_request = "http://table.finance.yahoo.com/table.csv?s=" + s + "&a=1&b=1&c=1950&d=10e=29&f=2012&ignore=.csv"
                print post_request, "\n\n"
                test_file = urllib2.urlopen(post_request).read()
                file_path = "/home/wilmott/Desktop/fourseasons/fourseasons/tmp/" + s + ".csv"

                fout = open("/home/wilmott/Desktop/fourseasons/fourseasons/tmp/" + s + ".csv", 'w')
                fout.write(test_file)
                fout.close()
            else:
                continue
        except:
            continue


#os.system("curl --silent 'http://download.finance.yahoo.com/d/quotes.csv?s=SLV&f=l' > tmp/SLV.csv")
#http://table.finance.yahoo.com/table.csv?a=["fmonth","fmonth"]&b=["fday","fday"]&c=["fyear","fyear"]&d=["tmonth","tmonth"]&e=["tday","tday"]&f=["tyear","tyear"]&s=["ticker", "ticker"]&y=0&g=["per","per"]&ignore=.csv

def yahoo_post():
    s= 'XOM'
    post_request = "http://table.finance.yahoo.com/table.csv?s=" + s + "&a=1&b=1&c=1950&d=10e=29&f=2012&ignore=.csv"
    test_file = urllib2.urlopen("http://table.finance.yahoo.com/table.csv?s=" + s + "&a=1&b=1&c=1950&d=10e=29&f=2012&ignore=.csv").read()
    fout = open("/home/wilmott/Desktop/fourseasons/fourseasons/tmp/" + s + ".csv", 'w')
    fout.write(test_file)

def multithread_yahoo():

    logging.basicConfig(level=logging.DEBUG,
                        format='%(threadName)-10s) %(message)s',
                        )

    def daemon(s):
        logging.debug('Starting ' + s)
        time.sleep(2)
        logging.debug('Exiting ' + s)

    d = threading.Thread(name='daemon', target=daemon)
    d.setDaemon(True)

    t = threading.Thread(name='non-daemon', target=non_daemon)

    stock_list = 'XOM\nMSFT\nAAPL\nGOOG\nJNJ\nPG\nF\nGM\nX\nAKS\nGLD\nSLV\nNEM\nABX'
    symbols = stock_list.split('\n')

    main_thread = threading.currentThread()
    for s in symbols:

        if len(threading.enumerate()) <= 3:
            print "if ", s, len(threading.enumerate())
            d = threading.Thread(name='daemon', target=daemon, args=[s])
            d.setDaemon(True)
            d.start()

        else:
            print "else ", s, len(threading.enumerate())
            d.join()
            d = threading.Thread(name='daemon', target=daemon, args=[s])
            d.setDaemon(True)
            d.start()

    d.join()






