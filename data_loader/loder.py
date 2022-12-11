from pykrx import stock
from pykrx import bond
import matplotlib
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import pickle


# 영업일 기준 최근일자 호출함수(L= linked_date_type ex- %Y%m%d , S=seperate_date_type ex- %Y-%m-%d)
# date_from_now() 는 오늘날짜 date_from_now(365)은 365일 전 날짜를 str형태로 출력한다.
def date_from_now(day:int=0, type="L"):        
    edate_str = stock.get_nearest_business_day_in_a_week()
    if type == "L":
        edate_dt = datetime.datetime.strptime(edate_str, '%Y%m%d')
        sdate_dt = edate_dt-datetime.timedelta(days=day)
        sdate_str = sdate_dt.strftime('%Y%m%d')
    elif type == "S":
        edate_dt = datetime.datetime.strptime(edate_str, '%Y%m%d')
        sdate_dt = edate_dt-datetime.timedelta(days=day)
        edate_str = edate_dt.strftime('%Y-%m-%d')
        sdate_str = sdate_dt.strftime('%Y-%m-%d')
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



if __name__=='__main__':   
    for market in ["KOSPI","KOSDAQ"]:    
        result_df = pd.DataFrame()
        # 티커 호출
        tickers = stock.get_market_ticker_list(date_from_now(), market=market)
        for ticker in tqdm(tickers[:1]):  # tickers[1270] 오류 있음  
                temp_df = get_stock_price_info(ticker)
                result_df = pd.concat([result_df,temp_df])
        result_df.columns = temp_df.columns
        result_df['market'] = market
        
        with open(f'../data/stock_price_data_all_period_{market}.pickle', 'wb') as f:
            pickle.dump(result_df, f)

    # 임시저장 파일 load
    with open('../data/stock_price_data_all_period_KOSPI.pickle', 'rb') as f:
        kospi_df = pickle.load(f)
    with open('../data/stock_price_data_all_period_KOSDAQ.pickle', 'rb') as f:
        kosdaq_df = pickle.load(f)
        
    all_market_df = pd.concat([kospi_df,kosdaq_df])
    # 최종 파일 생성
    with open(f'../data/stock_price_data_all_period_ALL.pickle', 'wb') as f:
        pickle.dump(all_market_df, f)