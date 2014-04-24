import numpy as np
from scipy import stats

def volatility_bs_annualized(price_data, v1, returns_period_length=1):

     #The function uses a formula very similar to the Bollinger Bands in which a "windowed" standard deviation is
     #calculated to ensure a high speed. The only major difference is that this formula does NOT feed in Simple Price
     #Data directly, but rather computes the day on day return of closing prices and then feeds that into the standard
     #deviation formula. At the end, the result is multiplied by the sqrt of 252 to give an annualized number. Also note
     #that using math.log is approx. 3 times faster than np.log

# #    cdef np.ndarray[np.float_t, ndim=1] ln_daily_returns, ma_returns
# #    cdef np.ndarray[np.float_t, ndim=1] price_sum, price_square_sum, variance, sigma, volatility
# #    cdef int x, len_data, warmup_factor
# #    cdef float root_252

     len_data = len(price_data)

     warmup_factor = v1 + returns_period_length
     # v_sum is the lookback period plus the number of periods we will be calculating the return over
     v_sum = v1 + returns_period_length - 1

     root_252 = np.sqrt(252)
     annualization_factor = root_252 / np.sqrt(returns_period_length)

     ln_daily_returns = np.empty(len_data)
     ma_returns = np.empty(len_data)

     price_sum = np.empty(len_data)
     price_square_sum = np.empty(len_data)
     variance = np.empty(len_data)
     sigma = np.empty(len_data)
     volatility = np.empty(len_data)

     ln_daily_returns[0:returns_period_length] = 0
     ma_returns[0:returns_period_length] = 0

     price_sum[0:v_sum] = 0
     price_square_sum[0:v_sum] = 0



     for x in xrange(returns_period_length, v_sum):

         if price_data[x] != price_data[x-returns_period_length]:
             ## ln_daily_returns[x] = np.log(np.abs(price_data[x] / price_data[x-1]))
             first_day = price_data[x-returns_period_length]
             ln_daily_returns[x] = np.log(np.abs(price_data[x] / first_day))
             #ln_daily_returns[x] = c_log(fabs(price_data[x] / price_data[x-1]))
             #ln_daily_returns[x] = c_log(price_data[x].close / price_data[x-1])
         else:
             ln_daily_returns[x] = 0

         ma_returns[x] = ma_returns[x-1] + (ln_daily_returns[x] / v1)

         price_sum[v_sum-1] += ln_daily_returns[x]

         price_square_sum[v_sum-1] += ln_daily_returns[x] * ln_daily_returns[x]   #faster than using an exponent

     variance[v_sum-1] = (v1 * (ma_returns[v_sum-1]**2)) - (2*ma_returns[v_sum-1]*(price_sum[v_sum-1]))+ price_square_sum[v_sum-1]



     for x in xrange(v_sum, len_data):

         if price_data[x] != price_data[x-returns_period_length]:
             ## ln_daily_returns[x] = np.log(np.abs(price_data[x] / price_data[x-1]))
             first_day = price_data[x-returns_period_length]
             ln_daily_returns[x] = np.log(np.abs(price_data[x] / first_day))
             #ln_daily_returns[x] = c_log(fabs(price_data[x] / price_data[x-1]))
             #ln_daily_returns[x] = c_log(price_data[x].close / price_data[x-1].close)
         else:
             ln_daily_returns[x] = 0

         ma_returns[x] = ma_returns[x-1] + ((ln_daily_returns[x] - ln_daily_returns[x-v1]) / v1)

         price_sum[x] = price_sum[x-1] + ln_daily_returns[x] - ln_daily_returns[x-v1]

         price_square_sum[x] = price_square_sum[x-1] + (ln_daily_returns[x]*ln_daily_returns[x]) - (ln_daily_returns[x-v1]*ln_daily_returns[x-v1])

         variance[x] = (v1 * ma_returns[x] * ma_returns[x]) - (2 * ma_returns[x] * price_sum[x]) + price_square_sum[x]

         sigma[x] = np.sqrt(variance[x] / v1)

         volatility[x] = sigma[x] * annualization_factor



     volatility[0:v_sum] = 0


     """
     for x in xrange(0, len_data):
         current_date = data[x].date
         print "%i \t %s \t %0.2f \t %0.4f \t %0.4f \t %0.4f \t %0.4f" % (x, current_date, price_data[x].close, ln_daily_returns[x], ma_returns[x], sigma[x], volatility[x])
     """

     return volatility

def simple_moving_average(price_data, s1):

    # cdef np.ndarray[np.float_t, ndim=1] sma
    # cdef int i, x, warmup_factor, len_data

    len_data = len(price_data)

    #Warmup Factor was determined from David's tests and varies for each function
    warmup_factor = s1 + 1

    sma = np.empty(len_data)

    #Use numpy to find the mean of the first set of elements and put it in position s1-1. This allows us to use the
    #windowed mean formula, below

    # Find the mean
    m = 0.0 # Mean
    for x in xrange(s1):
        m += price_data[x]
    m = m / s1
    sma[s1 - 1] = m

    for x in xrange(s1, len_data):
        sma[x] = sma[x-1] + ((price_data[x] - price_data[x-s1]) / s1)

    sma[0:warmup_factor] = 0

    """
    for x in xrange(0,len_data):
        current_date = data[x].date
        print "%i \t %s \t %0.2f \t %0.4f" % (x, current_date, price_data[x].close, ema[x])
    """

    return sma

def macd(price_data, m1, m2, m3):

     #Note that this macd does not return a histogram at the moment. It would be easy to adjust later and is probably
     #something we should do.

     # cdef np.ndarray[np.float_t, ndim=1] ema1, ema2, macd_line, signal_line
     # cdef int i, x, y, warmup_factor, len_data
     # cdef float k1, k2, k3

     len_data = len(price_data)

     #Warmup Factor was determined from David's tests and varies for each function
     warmup_factor = (5 * max(m1, m2, m3)) + 1

     #The k's are just constants based on the input args. They come from the definition of an ema
     k1 = 2/(float(m1+1))
     k2 = 2/(float(m2+1))
     k3 = 2/(float(m3+1))

     ema1 = np.empty(len_data)
     ema2 = np.empty(len_data)
     macd_line = np.empty(len_data)
     signal_line = np.empty(len_data)

     ema1[0]=price_data[0]
     ema2[0]=price_data[0]
     macd_line[0]=0
     signal_line[0]=0

     for y in xrange(1,len_data):
         ema1[y] = ((price_data[y]-ema1[y-1])*k1) + ema1[y-1]
         ema2[y] = ((price_data[y]-ema2[y-1])*k2) + ema2[y-1]
         macd_line[y] = ema1[y]-ema2[y]

         signal_line[y] = ((macd_line[y]-signal_line[y-1])*k3) + signal_line[y-1]

     ema1[0:warmup_factor] = 0
     ema2[0:warmup_factor] = 0
     macd_line[0:warmup_factor] = 0
     signal_line[0:warmup_factor] = 0

     """
     for x in xrange(0,len_data):
         current_date = data[x].date
         #print x, "\t", ema1[x], "\t", ema2[x], "\t", macd_line[x], "\t", signal_line[x]
         print "%i \t %s \t %0.2f \t %0.4f \t %0.4f \t %0.4f \t %0.4f" % (x, current_date, price_data[x].close, ema1[x], ema2[x], macd_line[x], signal_line[x])

     """
     return macd_line, signal_line

#cpdef np.ndarray[np.float_t, ndim=1] rsi(
#        np.ndarray[PriceSet, ndim=1] price_data,
#        int r1):

def rsi(price_data, r1):
#    cdef np.ndarray[np.float_t, ndim=1] gain_loss, gains, losses, avg_gain, avg_loss, rsi
#    cdef int i, x, len_data, warmup_factor

    len_data = len(price_data)

    #Warmup Factor was determined from David's tests and varies for each function
    warmup_factor = (7 * r1) + 1

    rsi = np.empty(len_data)
    gain_loss = np.empty(len_data)
    gains = np.empty(len_data)
    losses = np.empty(len_data)
    avg_gain = np.empty(len_data)
    avg_loss = np.empty(len_data)

    gain_loss[0] = 0
    avg_gain[0] = 0
    avg_loss[0] = 0


    for x in xrange (1, len_data):

        # gain_loss[x] = (price_data[x].close - price_data[x-1].close)
        gain_loss[x] = (price_data[x] - price_data[x-1])

        if gain_loss[x] > 0:
            gains[x] = gain_loss[x]
            losses[x] = 0
        elif gain_loss[x] < 0:
            gains[x] = 0
            losses[x] = -gain_loss[x]   #the minus is used in this case because it is faster than abs()
        elif gain_loss[x] == 0:
            gains[x] = 0
            losses[x] = 0

        avg_gain[x] = ((avg_gain[x-1]*(r1-1))+gains[x])/r1
        avg_loss[x] = ((avg_loss[x-1]*(r1-1))+losses[x])/r1


        rsi[x] = 100 - (100 * avg_loss[x]/(avg_gain[x]+avg_loss[x]))


    rsi[0:warmup_factor] = 0

    """
    for y in xrange(0, len_data):
        current_date = data[y].date
        #print "%i \t %s \t %0.2f \t %0.2f \t %0.2f \t %0.2f \t %0.2f" % (y, current_date, price_data[y].close, price_sum[y], price_square_sum[y], variance[y], sigma[y])
        #print "%i \t %s \t %0.2f \t %0.4f \t %0.4f \t %0.4f \t %0.4f \t %0.4f" % (y, current_date, price_data[y].close, gain_loss[y], avg_gain[y], avg_loss[y], rs[y], rsi[y])
        print "%i \t %s \t %0.2f \t %0.4f \t %0.4f \t %0.4f \t %0.4f" % (y, current_date, price_data[y].close, gain_loss[y], avg_gain[y], avg_loss[y], rsi[y])
    """

    return rsi

def sigma_prices(price_data, v1):

    #The function uses a formula very similar to the Bollinger Bands in which a "windowed" standard deviation is
    #calculated to ensure a high speed. The only major difference is that this formula does NOT feed in Simple Price
    #Data directly, but rather computes the day on day return of closing prices and then feeds that into the standard
    #deviation formula. At the end, the result is multiplied by the sqrt of 252 to give an annualized number. Also note
    #that using math.log is approx. 3 times faster than np.log

    len_data = len(price_data)

    warmup_factor = v1 + 1

    ma_prices = np.empty(len_data)

    price_sum = np.empty(len_data)
    price_square_sum = np.empty(len_data)
    variance = np.empty(len_data)
    sigma = np.empty(len_data)
    volatility = np.empty(len_data)

    ma_prices[0:v1] = 0

    price_sum[0:v1] = 0
    price_square_sum[0:v1] = 0

    m = 0
    for x in range(v1):
        m += price_data[x]
    m = m / v1
    ma_prices[v1-1] = m


    for x in xrange(0, v1):

        price_sum[v1-1] += price_data[x]
        price_square_sum[v1-1] += price_data[x] * price_data[x]

    variance[v1-1] = (v1 * (ma_prices[v1-1]**2)) - (2*ma_prices[v1-1] * (price_sum[v1-1])) + price_square_sum[v1-1]


    for x in xrange(v1, len_data):

        ma_prices[x] = ma_prices[x-1] + (price_data[x] - price_data[x-v1]) / v1
        price_sum[x] = price_sum[x-1] + price_data[x] - price_data[x-v1]
        price_square_sum[x] = price_square_sum[x-1] + (price_data[x] * price_data[x]) - (price_data[x-v1] * price_data[x-v1])
        variance[x] = (v1 * ma_prices[x] * ma_prices[x]) - (2 * ma_prices[x] * price_sum[x]) + price_square_sum[x]
        sigma[x] = np.sqrt(variance[x] / v1)

    sigma[0:warmup_factor] = 0
    """
    for x in xrange(0, len_data):
        current_date = data[x].date
        print "%i \t %s \t %0.2f \t %0.4f \t %0.4f \t %0.4f \t %0.4f" % (x, current_date, price_data[x].close, ln_daily_returns[x], ma_returns[x], sigma[x], volatility[x])
    """

    return sigma

def sigma_span(price_data, days, historical_sigma_lookback, sigma_input=None, sigma_average_range=0):

#    if sigma_input is not None:
#        historical_sigma = sigma_input

#    else:
#        # simply extract the most recent value
#        historical_sigma = sigma_prices(price_data, sigma_average_range)

    len_data = len(price_data)
    disp = np.empty(len_data)
    # disp becomes the percentage change in price, something like a momentum indicator
    for z in xrange(days + 1, len_data):
        disp[z] = (price_data[z] - price_data[z-days]) / price_data[z-days]
    disp[0:days+1] = 0

    # this is the historical std dev of the change in price over the past days
    historical_sigma = sigma_prices(disp, historical_sigma_lookback)

    warmup_factor = days + 1

    len_data = len(price_data)
    sigma_span = np.empty(len_data)

    for x in xrange(days + 1, len_data):
        if historical_sigma[x] == 0:
            sigma_span[x] = 0
            continue
        displacement = (price_data[x] - price_data[x-days]) / price_data[x-days]
        sigma_span[x] = displacement / historical_sigma[x]
        # sigma_span[x] = stats.percentileofscore(historical_sigma, displacement) - 50

#		print x, tr[x]['Date'], price_data[x], price_data[x-days], displacement, historical_sigma[x], sigma_span[x]

    sigma_span[0:warmup_factor] = 0

    return sigma_span, historical_sigma
