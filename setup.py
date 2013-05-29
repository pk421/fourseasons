from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# import Cython
# from Cython.Build import cythonize
# Cython.Compiler.Options.annotate = True

ext_modules = [
              Extension('src.toolsx', ['src/toolsx.pyx']),
              Extension('src.vol_analyzer', ['src/vol_analyzer.pyx']),
              # Extension('data.redis.manage_redis', ['data/redis/manage_redis.pyx'])
              ]

setup(
  name = 'modules to build',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules

)


# """
# from Cython.Build import cythonize

# setup(
#     name = "My hello app",
#     ext_modules = cythonize("*.pyx"),
# )

# # Can use this in say TA module where everything is a pyx file?
# # Or rather use it in Strategy
# import pyximport; pyximport.install()
# """
#######################################################################################################

# import os, sys
# from distutils.core import setup
# from distutils.extension import Extension

# try:
#     from Cython.Distutils import build_ext
#     from Cython.Build import cythonize
#     import Cython
#     Cython.Compiler.Options.annotate = True
# except ImportError:
#     print "You don't seem to have Cython installed. Please get a"
#     print "copy from www.cython.org and install it"
#     sys.exit(1)


# ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__)))
# BACKEND = os.path.join(ROOT, 'tradingclearly', 'backend')

# setup(
#       #name='TradingClearly Oculus',
#       name='tradingclearly',
#       version='0.01',
#       description='Trading Clearly Backend tools',
#       author='Kay Sackey, Michael Pilat',
#       author_email='admin@tradingclearly.com',

#     cmdclass = {'build_ext': build_ext},
#     packages=['tradingclearly',
#               'tradingclearly.backend',
#               'tradingclearly.backend.charting',
#               'tradingclearly.backend.data' ,
#               'tradingclearly.backend.models',
#               'tradingclearly.backend.seasonality',
#               'tradingclearly.backend.ta_screener_real_time',
#               'tradingclearly.backend.earnings_research',
#               'tradingclearly.backend.intraday_analysis',
#               'tradingclearly.backend.indicators.sharpe_ratio',
#               'tradingclearly.backend.indicators.sortino_ratio',
#               'tradingclearly.backend.indicators.macd',
#               'tradingclearly.backend.strategies.macd_cross',
#               'tradingclearly.backend.strategies.lin_reg_channel',
#               'tradingclearly.backend.strategies.fast_lin_reg_channel',
#               'tradingclearly.backend.strategies.lin_reg_chan_macd',
#               #'tradingclearly.backend.backtester.portfolio_manager',
#               #'tradingclearly.backend.backtester.time_simulator',],
#               'tradingclearly.backend.prob_engine.prob_simulator.pyx',
#               #'tradingclearly.backend.ta_screener_real_time.tools.pyx',
#               'tradingclearly.backend.comm_model.gold_model.pyx',
#               'tradingclearly.backend.returns_statistics.returns_calculator.pyx'
#               ],
#     ext_modules = [
#                     # Data models
#                     # Interfaces to Cassandra, Redis, Yahoo and Google
#                     # Todo: Add EarningsWhispers to that list
#                     Extension('tradingclearly.backend.data.models',
#                                 [BACKEND + '/data/models.pyx']),
#                     Extension('tradingclearly.backend.data.redis',
#                                 [BACKEND + '/data/redis.pyx']),
#                     # Charting (Trends)
#                     # Todo: This file should be in another directory
#                     Extension('tradingclearly.backend.seasonality.trends',
#                                 [BACKEND + '/seasonality/trends.pyx']),
#                     # Seasonality
#                     Extension('tradingclearly.backend.seasonality.seasonality',
#                                 [BACKEND + '/seasonality/seasonality.pyx']),
#                     # Real Time Screener
#                     Extension('tradingclearly.backend.ta_screener_real_time.real_time_screener',
#                                 [BACKEND + '/ta_screener_real_time/real_time_screener.pyx']),
#                     Extension('tradingclearly.backend.ta_screener_real_time.tools',
#                                 [BACKEND + '/ta_screener_real_time/tools.pyx']),
#                     #Earnings Research
#                     Extension('tradingclearly.backend.earnings_research.earnings_statistics',
#                                 [BACKEND + '/earnings_research/earnings_statistics.pyx']),
#                     #Intraday Analysis
#                     Extension('tradingclearly.backend.intraday_analysis.intraday_test',
#                                 [BACKEND + '/intraday_analysis/intraday_test.pyx']),

#                     #Backtester
#                     #Extension('tradingclearly.backend.backtester.portfolio_manager',
#                     #            [BACKEND + '/backtester/portfolio_manager.pyx']),
#                     #Extension('tradingclearly.backend.backtester.time_simulator',
#                     #            [BACKEND + '/backtester/time_simulator.pyx']),
#                     Extension('tradingclearly.backend.backtester.indicators.sharpe_ratio',
#                                 [BACKEND + '/backtester/indicators/sharpe_ratio.pyx']),

#                     Extension('tradingclearly.backend.backtester.indicators.sortino_ratio',
#                                 [BACKEND + '/backtester/indicators/sortino_ratio.pyx']),

#                     Extension('tradingclearly.backend.backtester.indicators.macd',
#                                 [BACKEND + '/backtester/indicators/macd.pyx']),

#                     Extension('tradingclearly.backend.backtester.indicators.seasonality_ind',
#                                 [BACKEND + '/backtester/indicators/seasonality_ind.pyx']),

#                     Extension('tradingclearly.backend.backtester.strategies.macd_cross',
#                                 [BACKEND + '/backtester/strategies/macd_cross.pyx']),

#                     Extension('tradingclearly.backend.backtester.indicators.lin_reg_channel',
#                                 [BACKEND + '/backtester/indicators/lin_reg_channel.pyx']),

#                     Extension('tradingclearly.backend.backtester.indicators.fast_lin_reg_channel',
#                                 [BACKEND + '/backtester/indicators/fast_lin_reg_channel.pyx']),

#                     Extension('tradingclearly.backend.backtester.strategies.lin_reg_chan_macd',
#                                 [BACKEND + '/backtester/strategies/lin_reg_chan_macd_hist.pyx']),

#                     Extension('tradingclearly.backend.prob_engine.prob_simulator',
#                                 [BACKEND + '/prob_engine/prob_simulator.pyx']),

#                     Extension('tradingclearly.backend.comm_model.gold_model',
#                                 [BACKEND + '/comm_model/gold_model.pyx']),

#                     Extension('tradingclearly.backend.returns_statistics.returns_calculator',
#                                 [BACKEND + '/returns_statistics/returns_calculator.pyx'])


#                 ],
# )

#######################################################################################################

# Build with:
# python setup.py build_ext --inplace --pyrex-c-in-temp

# Checkout: http://wiki.cython.org/enhancements/distutils_preprocessing