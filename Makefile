#!/bin/sh

clean:
	find /home/wilmott/Desktop/fourseasons/fourseasons/src -name *.so | xargs rm
	find /home/wilmott/Desktop/fourseasons/fourseasons/src -name *.c | xargs rm
	find /home/wilmott/Desktop/fourseasons/fourseasons/src -name *.pyc | xargs rm

cython:
	python setup.py build_ext --inplace --pyrex-c-in-temp

kill_python:
	killall -9 python

mount_shares:
	sudo mount.vboxsf fourseasons Desktop/fourseasons/

start_services:
	#./start_services.sh
	supervisord --configuration etc/supervisord.conf

stop_services:
	supervisorctl --configuration etc/supervisord.conf stop all
	supervisorctl --configuration etc/supervisord.conf shutdown

tornado_start:
	supervisorctl --configuration etc/supervisord.conf start tornado
	# python src/web/http_processor.py

tornado_stop:
	supervisorctl --configuration etc/supervisord.conf stop tornado

poll_start:
	supervisorctl --configuration etc/supervisord.conf start poll_realtime_data

poll_stop:
	supervisorctl --configuration etc/supervisord.conf stop poll_realtime_data

redis_start:
	supervisorctl --configuration etc/supervisord.conf start redis

redis_stop:
	supervisorctl --configuration etc/supervisord.conf stop redis



cointegrations:
	python test_run.py --cointegrations

correlations:
	python test_run.py --correlations

download_stocks:
	python test_run.py --download_stocks

extract_symbols:
	python test_run.py --extract_symbols_with_historical_data

harding_seasonality:
	python test_run.py --harding_seasonality

indicator_system:
	python test_run.py --indicator_system

live_monitor:
	python test_run.py --live_monitor

live_portfolio_analysis:
	python test_run.py --live_portfolio_analysis

load_redis:
	python test_run.py --load_redis

poll_realtime_data:
	python test_run.py --poll_realtime_data

portfolio_analysis:
	python test_run.py --portfolio_analysis

read_redis:
	python test_run.py --read_redis

returns_analyzer:
	python test_run.py --returns_analyzer

stock_analyzer:
	python test_run.py --stock_analyzer

sweep_portfolios:
	python test_run.py --sweep_portfolios

vol_analyzer:
	python test_run.py --vol_analyzer



profile:
	# python -m cProfile -o AAAprofile.stats test_run.py --portfolio_analysis
	python -m cProfile -o AAAprofile.stats test_run.py --sweep_portfolios
	# python test_run.py --vol_analyzer;  python util/gprof2dot.py -n0.5 -e0.5 -f pstats AAAprofile.stats | dot -Tpng -o output.png

gprof:
	util/gprof2dot.py -n0.5 -e0.5 -f pstats AAAprofile.stats | dot -Tpng -o output.stats.png
