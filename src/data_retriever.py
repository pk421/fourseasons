import os
import urllib2

mike = "mike!"

class bla(object):

    def __init__(self):

        print "in init"
        self.test = "hello"

    def get(self):
        print self.test


def get_yahoo_data():
    #This procedure will work after the DOS line endings have already been converted to LINUX. Use: sed -i 's/\r//' filename
    stock_list = open('/home/wilmott/Desktop/fourseasons/fourseasons/data/large_universe', 'r')
    symbols = stock_list.read().split('\n')
    stock_list.close()

    print symbols
    print"length", len(symbols)

    for s in symbols:
        print s
        #single_quote_request = "curl --silent 'http://download.finance.yahoo.com/d/quotes.csv?s=" + s + "&f=l' > tmp/" + s + ".csv"
        full_refresh_request = "curl 'http://table.finance.yahoo.com/table.csv?s=" + s + "&a=1&b=1&c=1950&d=10e=29&f=2012&ignore=.csv' > tmp/" + s + ".csv"
        print full_refresh_request
        try:
            file_path = "/home/wilmott/Desktop/fourseasons/fourseasons/tmp/" + s + ".csv"
            if (os.path.exists(file_path) == False) or os.path.getsize(file_path) < 3500:
                os.system(full_refresh_request)
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
    print test_file

    fout = open("/home/wilmott/Desktop/fourseasons/fourseasons/tmp/" + "ZZZZZ" + ".csv", 'w')
    #fout.write("This is a test")
    fout.write(test_file)
