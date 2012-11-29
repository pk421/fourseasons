import urllib2
import os
import time
import datetime
import redis
from data.redis.manage_redis import pack_realtime_data
from data.redis.manage_redis import fill_realtime_redis

def query_realtime_data():
    """
    from here: http://www.forexpros.com/commodities/
    """

    base_path = '/home/wilmott/Desktop/fourseasons/fourseasons/'
    request = "http://www.forexpros.com/commodities/"
    #test_file = urllib2.urlopen(request, timeout=2).read()

    input_file = open((base_path + "src/scrape_output.html"), 'r')
    input_data = input_file.read()
    input_file.close()

    counter = 0
    start_time = datetime.datetime.now()
    while True:
        time.sleep(0.25)
        current_time = datetime.datetime.now()
        current_second = current_time.second
        
        if current_second > 0 and current_second < 5:
            try:
                test_file = urllib2.urlopen(request, timeout=2).read()
                timestamp = time.time()
                price_item = parse_data(test_file, timestamp)

                #This block deals with data that was retrieved after hours and contains a time of "day/month"
                if '/' in price_item['Time']:
                    continue

                fill_realtime_redis(price_item, store_under='realtime_1min:', delete_old_data=False)

                print counter, "\t", price_item['Last'], "\t", price_item['Time']
                time.sleep(10)
                counter += 1

            except:
                time.sleep(2)
                continue



    price_item = parse_data(input_data)
    print price_item

#    fout = open((base_path + "/src/scrape_output.html"), 'w')
#    fout.write(test_file)
#    fout.close()
    
    return

def parse_data(scraped_data, counter):

    find_gold = scraped_data.split('<nobr><a href="/commodities/gold" title="Gold" >Gold</a></nobr></td><td nowrap=\"nowrap\" class=\"m_t\">')
    #print len(find_gold)

    find_date = find_gold[1][0:6]
    #print find_date

    before_silver = find_gold[1].split("Silver")[0]
    #print before_silver

    gold_parts = before_silver.split("</td>")

    last = gold_parts[1].split(">")[1]
    previous = gold_parts[2].split(">")[1]
    high = gold_parts[3].split(">")[1]
    low = gold_parts[4].split(">")[1]
    change = gold_parts[5].split(">")[1]
    change_pct = gold_parts[6].split(">")[1]
    last_time = gold_parts[7].split(">")[1]

    realtime_price_data = {}
    realtime_price_data['Last'] = last
    realtime_price_data['Previous'] = previous
    realtime_price_data['High'] = high
    realtime_price_data['Low'] = low
    realtime_price_data['Change'] = change
    realtime_price_data['ChangePct'] = change_pct

    realtime_price_data['Time'] = last_time
    realtime_price_data['Date'] = str(datetime.datetime.today())

    realtime_price_data['Counter'] = counter
    realtime_price_data['Symbol'] = 'Gold_Spot'

    return realtime_price_data


    return price_item

#    <h3><a href="/commodities/real-time-futures">Real Time Commodities</a> <span
#    class="newSiteIconsSprite doubleArrowLink">&nbsp;</span></h3><h5 class="linkTitle subTitle">
#    <a href="/commodities/energies">Energy</a> <span class="newSiteIconsSprite doubleArrowLinkSmall">&nbsp;</span>
#    </h5><table id="energy" tablesorter class="genTable closedTable crossRatesTable" cellspacing="0" cellpadding="0" border="0">
#    <thead><tr><th class="icon" nowrap="nowrap">&nbsp;</th><th class="symbol left" nowrap="nowrap">
#    <nobr>Commodity</nobr></th><th nowrap="nowrap">Month</th><th nowrap="nowrap">Last</th><th nowrap="nowrap">Prev.</th>
#    <th nowrap="nowrap">High</th><th nowrap="nowrap">Low</th><th nowrap="nowrap">Chg.</th><th nowrap="nowrap">Chg. %</th>
#    <th nowrap="nowrap">Time</th></tr></thead><tbody><tr id="pair_8833"><td class="center">
#    <span class="newSiteIconsSprite  greenArrowIcon">&nbsp;</span></td><td class="bold left" nowrap="nowrap"><nobr>
#    <a href="/commodities/brent-oil" title="Brent Oil" >Brent Oil</a></nobr></td><td nowrap="nowrap" class="m_t">Jan 13</td>
#    <td id="energycl_last8833">111.70</td><td>111.64</td><td>111.78</td><td class="">111.47</td><td class="bold greenFont">0.06</td>
#    <td class="bold greenFont">0.05%</td><td id="energycl_date8833">2:40:43</td></tr><tr id="pair_8849"><td class="center">
#    <span class="newSiteIconsSprite  redArrowIcon">&nbsp;</span></td><td class="bold left" nowrap="nowrap"><nobr>
#    <a href="/commodities/crude-oil" title="Crude Oil" >Crude Oil</a></nobr></td><td nowrap="nowrap" class="m_t">Jan 13</td>
#    <td id="energycl_last8849">89.05</td><td>89.28</td><td>89.28</td><td class="">88.92</td><td class="bold redFont">-0.23</td>
#    <td class="bold redFont">-0.26%</td><td id="energycl_date8849">2:41:40</td></tr><tr id="pair_8862"><td class="center">
#    <span class="newSiteIconsSprite  greenArrowIcon">&nbsp;</span></td><td class="bold left" nowrap="nowrap"><nobr>
#    <a href="/commodities/natural-gas" title="Natural Gas" >Natural Gas</a></nobr></td><td nowrap="nowrap" class="m_t">Dec 12</td>
#    <td id="energycl_last8862">3.744</td><td>3.738</td><td>3.744</td><td class="">3.725</td><td class="bold greenFont">0.005</td>
#    <td class="bold greenFont">0.15%</td><td id="energycl_date8862">2:41:40</td></tr><tr id="pair_8988"><td class="center">
#    <span class="newSiteIconsSprite  greenArrowIcon">&nbsp;</span></td><td class="bold left" nowrap="nowrap"><nobr>
#    <a href="/commodities/heating-oil" title="Heating Oil" >Heating Oil</a></nobr></td><td nowrap="nowrap" class="m_t">
#    Dec 12</td><td id="energycl_last8988">3.0822</td><td>3.0821</td><td>3.0848</td><td class="">3.0779</td>
#    <td class="bold greenFont">0.0001</td><td class="bold greenFont">0.00%</td><td id="energycl_date8988">2:39:46</td>
#    </tr></tbody></table><h5 class="linkTitle"><a href="/commodities/metals">Metals</a>
#    <span class="newSiteIconsSprite doubleArrowLinkSmall">&nbsp;</span></h5>
#    <table id="metals" tablesorter class="genTable closedTable crossRatesTable" cellspacing="0" cellpadding="0" border="0">
#    <thead><tr><th class="icon" nowrap="nowrap">&nbsp;</th><th class="symbol left" nowrap="nowrap"><nobr>Commodity</nobr>
#    </th><th nowrap="nowrap">Month</th><th nowrap="nowrap">Last</th><th nowrap="nowrap">Prev.</th><th nowrap="nowrap">High</th>
#    <th nowrap="nowrap">Low</th><th nowrap="nowrap">Chg.</th><th nowrap="nowrap">Chg. %</th><th nowrap="nowrap">Time</th>
#    </tr></thead><tbody><tr id="pair_8830"><td class="center"><span class="newSiteIconsSprite  greenArrowIcon">&nbsp;</span>
#    </td><td class="bold left" nowrap="nowrap"><nobr><a href="/commodities/gold" title="Gold" >Gold</a></nobr></td>
#    <td nowrap="nowrap" class="m_t">Dec 12</td><td id="metalscl_last8830">1733.65</td><td>1734.40</td><td>1733.95</td>
#    <td class="">1729.35</td><td class="bold redFont">-0.75</td><td class="bold redFont">-0.04%</td><td id="metalscl_date8830">
#    2:41:28</td></tr><tr id="pair_8836"><td class="center"><span class="newSiteIconsSprite  redArrowIcon">&nbsp;</span>
#    </td><td class="bold left" nowrap="nowrap"><nobr><a href="/commodities/silver" title="Silver" >Silver</a></nobr></td>
#    <td nowrap="nowrap" class="m_t">Dec 12</td><td id="metalscl_last8836">33.165</td><td>33.189</td><td>33.183</td>
#    <td class="">33.013</td><td class="bold redFont">-0.024</td><td class="bold redFont">-0.07%</td><td id="metalscl_date8836">
#    2:41:40</td></tr><tr id="pair_8831"><td class="center"><span class="newSiteIconsSprite  greenArrowIcon">&nbsp;</span>
#    </td><td class="bold left" nowrap="nowrap"><nobr><a href="/commodities/copper" title="Copper" >Copper</a></nobr></td>
#    <td nowrap="nowrap" class="m_t">Dec 12</td><td id="metalscl_last8831">3.518</td><td>3.527</td><td>3.529</td><td class="">
#    3.514</td><td class="bold redFont">-0.009</td><td class="bold redFont">-0.26%</td><td id="metalscl_date8831">2:41:25
#    </td></tr><tr id="pair_8910"><td class="center"><span class="newSiteIconsSprite  greenArrowIcon">&nbsp;</span></td>
#    <td class="bold left" nowrap="nowrap"><nobr><a href="/commodities/platinum" title="Platinum" >Platinum</a></nobr></td>
#    <td nowrap="nowrap" class="m_t">Jan 13</td><td id="metalscl_last8910">1581.55</td><td>1579.15</td><td>1581.90</td>
#    <td class="">1575.45</td><td class="bold greenFont">2.40</td><td class="bold greenFont">0.15%</td><td id="metalscl_date8910">
#    2:40:55</td></tr></tbody></table><h5 class="linkTitle"><a href="/commodities/softs">Agriculture</a>
#    <span class="newSiteIconsSprite doubleArrowLinkSmall">&nbsp;</span></h5>
#    <table id="agriculture" tablesorter class="genTable closedTable crossRatesTable" cellspacing="0" cellpadding="0" border="0">
#    <thead><tr><th class="icon" nowrap="nowrap">&nbsp;</th><th class="symbol left" nowrap="nowrap"><nobr>Commodity</nobr></th>
#    <th nowrap="nowrap">Month</th><th nowrap="nowrap">Last</th><th nowrap="nowrap">Prev.</th><th nowrap="nowrap">High</th>
#    <th nowrap="nowrap">Low</th><th nowrap="nowrap">Chg.</th><th nowrap="nowrap">Chg. %</th><th nowrap="nowrap">Time</th>
#    </tr></thead><tbody><tr id="pair_8832"><td class="center"><span class="newSiteIconsSprite  greenArrowIcon">&nbsp;</span>
#    </td><td class="bold left" nowrap="nowrap"><nobr><a href="/commodities/us-coffee-c" title="US Coffee C" >US Coffee C</a>
#    </nobr></td><td nowrap="nowrap" class="m_t">Mar 13</td><td id="agriculturecl_last8832">156.35</td><td>156.35</td>
#    <td>157.78</td><td class="">153.03</td><td class="bold greenFont">3.98</td><td class="bold greenFont">2.61%</td>
#    <td id="agriculturecl_date8832">19/11</td></tr><tr id="pair_8918"><td class="center">
#    <span class="newSiteIconsSprite  greenArrowIcon">&nbsp;</span></td><td class="bold left" nowrap="nowrap"><nobr
#    ><a href="/commodities/us-corn" title="US Corn" >US Corn</a></nobr></td><td nowrap="nowrap" class="m_t">Dec 12</td>
#    <td id="agriculturecl_last8918">738.88</td><td>739.88</td><td>739.38</td><td class="">738.12</td>
#    <td class="bold redFont">-1.00</td><td class="bold redFont">-0.14%</td><td id="agriculturecl_date8918">2:41:07</td>
#    </tr><tr id="pair_8917"><td class="center"><span class="newSiteIconsSprite  greenArrowIcon">&nbsp;</span></td>
#    <td class="bold left" nowrap="nowrap"><nobr><a href="/commodities/us-wheat" title="US Wheat" >US Wheat</a></nobr>
#    </td><td nowrap="nowrap" class="m_t">Dec 12</td><td id="agriculturecl_last8917">844.38</td><td>841.88</td>
#    <td>844.62</td><td class="">842.62</td><td class="bold greenFont">2.50</td><td class="bold greenFont">0.30%</td>
#    <td id="agriculturecl_date8917">2:40:28</td></tr><tr id="pair_8834"><td class="center">
#    <span class="newSiteIconsSprite  greenArrowIcon">&nbsp;</span></td><td class="bold left" nowrap="nowrap"><nobr>
#    <a href="/commodities/london-sugar" title="London Sugar" >London Sugar</a></nobr></td>
#    <td nowrap="nowrap" class="m_t">Mar 13</td><td id="agriculturecl_last8834">526.05</td><td>526.05</td><td>526.75</td>
#    <td class="">516.00</td><td class="bold greenFont">17.10</td><td class="bold greenFont">3.36%</td>
#    <td id="agriculturecl_date8834">19/11</td></tr><tr id="pair_8851"><td class="center">
#    <span class="newSiteIconsSprite  greenArrowIcon">&nbsp;</span></td><td class="bold left" nowrap="nowrap"><nobr>
#    <a href="/commodities/us-cotton-no.2" title="US Cotton No.2" >US Cotton No.2</a></nobr></td>
#    <td nowrap="nowrap" class="m_t">Mar 13</td><td id="agriculturecl_last8851">71.81</td><td>71.98</td><td>72.00</td>
#    <td class="">71.72</td><td class="bold redFont">-0.17</td><td class="bold redFont">-0.23%</td>
#    <td id="agriculturecl_date8851">2:41:28</td></tr></tbody></table>
