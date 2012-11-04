from src.data_retriever import *

multithread_yahoo_download('large_universe.csv', thread_count=2, update_check=False, new_only=False)
#extract_symbols_with_historical_data()

#objectify_data()
