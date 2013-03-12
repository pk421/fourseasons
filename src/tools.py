import numpy as np

def volatility_bs_annualized(price_data, v1):

    #The function uses a formula very similar to the Bollinger Bands in which a "windowed" standard deviation is
    #calculated to ensure a high speed. The only major difference is that this formula does NOT feed in Simple Price
    #Data directly, but rather computes the day on day return of closing prices and then feeds that into the standard
    #deviation formula. At the end, the result is multiplied by the sqrt of 252 to give an annualized number. Also note
    #that using math.log is approx. 3 times faster than np.log

#    cdef np.ndarray[np.float_t, ndim=1] ln_daily_returns, ma_returns
#    cdef np.ndarray[np.float_t, ndim=1] price_sum, price_square_sum, variance, sigma, volatility
#    cdef int x, len_data, warmup_factor
#    cdef float root_252

    len_data = len(price_data)

    warmup_factor = v1 + 1

    root_252 = np.sqrt(252)

    ln_daily_returns = np.empty(len_data)
    ma_returns = np.empty(len_data)

    price_sum = np.empty(len_data)
    price_square_sum = np.empty(len_data)
    variance = np.empty(len_data)
    sigma = np.empty(len_data)
    volatility = np.empty(len_data)

    ln_daily_returns[0] = 0
    ma_returns[0] = 0

    price_sum[0:v1] = 0
    price_square_sum[0:v1] = 0



    for x in xrange(1, v1):

        if price_data[x] != price_data[x-1]:
            ln_daily_returns[x] = np.log(np.abs(price_data[x] / price_data[x-1]))
            #ln_daily_returns[x] = c_log(fabs(price_data[x] / price_data[x-1]))
            #ln_daily_returns[x] = c_log(price_data[x].close / price_data[x-1])
        else:
            ln_daily_returns[x] = 0

        ma_returns[x] = ma_returns[x-1] + (ln_daily_returns[x] / v1)

        price_sum[v1-1] += ln_daily_returns[x]

        price_square_sum[v1-1] += ln_daily_returns[x] * ln_daily_returns[x]   #faster than using an exponent

    variance[v1-1] = (v1 * (ma_returns[v1-1]**2)) - (2*ma_returns[v1-1]*(price_sum[v1-1]))+ price_square_sum[v1-1]



    for x in xrange(v1, len_data):

        if price_data[x] != price_data[x-1]:
            ln_daily_returns[x] = np.log(np.abs(price_data[x] / price_data[x-1]))
            #ln_daily_returns[x] = c_log(fabs(price_data[x] / price_data[x-1]))
            #ln_daily_returns[x] = c_log(price_data[x].close / price_data[x-1].close)
        else:
            ln_daily_returns[x] = 0

        ma_returns[x] = ma_returns[x-1] + ((ln_daily_returns[x] - ln_daily_returns[x-v1]) / v1)

        price_sum[x] = price_sum[x-1] + ln_daily_returns[x] - ln_daily_returns[x-v1]

        price_square_sum[x] = price_square_sum[x-1] + (ln_daily_returns[x]*ln_daily_returns[x]) - (ln_daily_returns[x-v1]*ln_daily_returns[x-v1])

        variance[x] = (v1 * ma_returns[x] * ma_returns[x]) - (2 * ma_returns[x] * price_sum[x]) + price_square_sum[x]

        sigma[x] = np.sqrt(variance[x] / v1)

        volatility[x] = sigma[x] * root_252



    volatility[0:warmup_factor] = 0


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
    