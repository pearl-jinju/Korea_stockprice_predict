from pykrx import stock
from pykrx import bond
import matplotlib
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm


# data 조회일자 기준 x일 이전부터 현재까지==============
# 영업일 기준 최근일자 호출함수
def date_from_now(day:int=0):        
    edate_str = stock.get_nearest_business_day_in_a_week()
    edate_dt = datetime.datetime.strptime(edate_str, '%Y%m%d')
    sdate_dt = edate_dt-datetime.timedelta(days=day)
    sdate_str = sdate_dt.strftime('%Y%m%d')
    return sdate_str

# 종목 기초정보 제공 함수
# 티커/종목명/시가/종가/변동폭(오늘)/등락률(오늘)/거래량(오늘)/거래대금(오늘)/상장주식수/보유수량/지분율/한도수량/한도소진률/BPS/PER/PBR/EPS/DIV/DPS
def get_stock_basic_info(day:int=0, market:str="ALL"):
    df_stake_info = stock.get_exhaustion_rates_of_foreign_investment(date_from_now(), market).reset_index()
    df_change = stock.get_market_price_change(date_from_now(day),date_from_now(), market=market).reset_index()
    df_result = pd.merge(df_change,df_stake_info, on='티커',how='left')
    
    df_fundamental = stock.get_market_fundamental( date_from_now(), market=market).reset_index()
    df_result = pd.merge(df_result,df_fundamental, on='티커',how='left')
    return df_result

df_stock_basic_info = get_stock_basic_info()


#종목명을 받으면 기간동안의 주가/거래량/수급/기본Fundamental 지표 변동을 출력해주는 함수(day default 종목 전체기간 데이터 호출)
def get_stock_price_info(ticker:str, day:int=50000, market:str="ALL"):
    df_stock_ohlcv = stock.get_market_ohlcv(date_from_now(day), date_from_now(), ticker)
    cond = df_stock_basic_info['티커']==ticker
    stock_name = df_stock_basic_info[cond]['종목명'].values[0]
    print(stock_name)
    df_stock_volume_detail = stock.get_market_trading_value_by_date(date_from_now(day), date_from_now(), ticker)
    df_result = pd.merge(df_stock_ohlcv, df_stock_volume_detail, left_index=True, right_index=True, how='left')
    df_stock_fundamental = stock.get_market_fundamental(date_from_now(day), date_from_now(), ticker)
    df_result = pd.merge(df_result, df_stock_fundamental, left_index=True, right_index=True, how='left').reset_index()
    df_result['종목명'] = stock_name
    return df_result

# 티커 호출
tickers = stock.get_market_ticker_list(date_from_now(), market="ALL")

result_df = pd.DataFrame()
for ticker in tqdm(tickers):
    temp_df = get_stock_price_info(ticker)
    result_df = pd.concat([result_df,temp_df])
    
result_df.columns = temp_df.columns

result_df.to_csv("../data/get_stock_price_info.csv",index=False, encoding='utf-8')
     