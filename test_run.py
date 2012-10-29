from src.data_retriever import *

#get_yahoo_data()
#yahoo_post()
multithread_yahoo()


#import threading
#import logging
#import time
#
#logging.basicConfig(level=logging.DEBUG,
#                    format='%(threadName)-10s) %(message)s',
#                    )
#
#def daemon(s):
#    logging.debug('Starting ' + s)
#    time.sleep(2)
#    logging.debug('Exiting ' + s)
#
#d = threading.Thread(name='daemon', target=daemon)
#d.setDaemon(True)
#
#t = threading.Thread(name='non-daemon', target=non_daemon)
#
#stock_list = 'XOM\nMSFT\nAAPL\nGOOG\nJNJ\nPG\nF\nGM\nX\nAKS\nGLD\nSLV\nNEM\nABX'
#symbols = stock_list.split('\n')
#
#main_thread = threading.currentThread()
#for s in symbols:
#
#    if len(threading.enumerate()) <= 3:
#        print "if ", s, len(threading.enumerate())
#        d = threading.Thread(name='daemon', target=daemon, args=[s])
#        d.setDaemon(True)
#        d.start()
#
#    else:
#        print "else ", s, len(threading.enumerate())
#        d.join()
#        d = threading.Thread(name='daemon', target=daemon, args=[s])
#        d.setDaemon(True)
#        d.start()
#
#d.join()