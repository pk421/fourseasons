from src.data_retriever import *
import argparse
import os

parser = argparse.ArgumentParser()
#parser.add_argument("square",
#                    help="display a square of a given number",
#                    type=int)
#parser.add_argument("-v", "--verbosity",
#                    help="increase output verbosity",
#                    action="store_true")
parser.add_argument("--download_stocks",
                    help="run multithreaded download from yahoo",
                    action="store_true")
parser.add_argument("--extract_symbols_with_historical_data",
                    help="find symbols that have historical data",
                    action="store_true")
parser.add_argument("--load_redis",
                    help="move stock historical data in objects, and then into redis validating data at the same time",
                    action="store_true")
parser.add_argument("--read_redis",
                    help="read data from redis and put it in python objects in memory",
                    action="store_true")

args = parser.parse_args()
#print args.square**2
#if args.verbosity:
#    print "verbosity turned on"

if args.download_stocks:
    multithread_yahoo_download('large_universe.csv', thread_count=2, update_check=False, new_only=False)

if args.extract_symbols_with_historical_data:
    extract_symbols_with_historical_data()

if args.load_redis:
    load_redis()

if args.read_redis:
    read_redis()
