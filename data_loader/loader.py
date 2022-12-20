from pykrx import stock
import pandas as pd
import datetime
from tqdm import tqdm
import pickle
import os


def date_from_now(day:int=0, type="L"):
    '''
    영업일 기준 최근일자 호출함수(L= linked_date_type ex- %Y%m%d , S=seperate_date_type ex- %Y-%m-%d)\n
    date_from_now() 는 오늘날짜 date_from_now(365)은 365일 전 날짜를 str형태로 출력한다.
    '''        
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


def get_stock_basic_info(day:int=0, market:str="ALL"):
    """ 종목 기초정보 제공 함수
        티커/종목명/시가/종가/변동폭(오늘)/등락률(오늘)/\n
        거래량(오늘)/거래대금(오늘)/상장주식수/보유수량/\n
        지분율/한도수량/한도소진률/BPS/PER/PBR/EPS/DIV/DPS_

    Args:
        day (int, optional): _description_. Defaults to 0.
        market (str, optional): _description_. Defaults to "ALL".

    Returns:
        _type_: DataFrame
    """
    df_stake_info = stock.get_exhaustion_rates_of_foreign_investment(date_from_now(), market).reset_index()
    df_change = stock.get_market_price_change(date_from_now(day),date_from_now(), market=market).reset_index()
    df_result = pd.merge(df_change,df_stake_info, on='티커',how='left')
    df_fundamental = stock.get_market_fundamental( date_from_now(), market=market).reset_index()
    df_result = pd.merge(df_result,df_fundamental, on='티커',how='left')
    return df_result

df_stock_basic_info = get_stock_basic_info()




def get_stock_price_info(ticker, market="ALL", info="ALL", day=50000):
    """
    종목명을 받으면 기간동안의 주가/거래량/수급/기본Fundamental 지표 변동을 출력해주는 함수(day default 종목 전체기간 데이터 호출)
    info="ALL" 거래량과 펀더멘탈 정보 포함
    info="BASIC" 기본 정보만 포함
    """
    df_stock_ohlcv = stock.get_market_ohlcv(date_from_now(day), date_from_now(), ticker)
    cond = df_stock_basic_info['티커']==ticker
    stock_name = df_stock_basic_info[cond]['종목명'].values[0]
    if info=="ALL":
        df_stock_volume_detail = stock.get_market_trading_value_by_date(date_from_now(day), date_from_now(), ticker)
        df_result = pd.merge(df_stock_ohlcv, df_stock_volume_detail, left_index=True, right_index=True, how='left')
        df_stock_fundamental = stock.get_market_fundamental(date_from_now(day), date_from_now(), ticker)
        df_result = pd.merge(df_result, df_stock_fundamental, left_index=True, right_index=True, how='left').reset_index()
    elif info=="BASIC":
        df_result = df_stock_ohlcv
    df_result['종목명'] = stock_name
    return df_result

def ticker_to_name(ticker): 
    """티커를 받으면 종목명을 반환하는 함수 """
    name = stock.get_market_ticker_name(ticker)
    return name

def get_index_fundamental_info(stock_ticker):
    ''' 종목의 티커를 입력하면 그 종목이 소속된 시장의 funadamental 지표를 반환하는 함수 \n
        당일의 자료는 pkl 형식으로 파일을 저장하며,\n
        이미 pkl이 있다면 파일을 불러오는 함수를 호출한다.'''
    # 파일이 없다면
    if os.path.isfile(f'../data/market_fundamental_{date_from_now()}.pkl')==False:
        market_tickers_kospi = stock.get_index_ticker_list(market='KOSPI')
        market_tickers_kosdaq = stock.get_index_ticker_list(market='KOSDAQ')
        result_df = pd.DataFrame()
        maket_tickers_all = market_tickers_kospi+market_tickers_kosdaq
        for maket_ticker in tqdm(maket_tickers_all):
            # 시장명 str
            maket_ticker = maket_ticker
            market_name = stock.get_index_ticker_name(maket_ticker)
            stocks_in_market = stock.get_index_portfolio_deposit_file(str(maket_ticker))
            if stocks_in_market!=[]:
                # 당일은 지수의 등락률이 0.0으로 나옴 funda만 사용할것!
                index_fundamental = stock.get_index_fundamental(date_from_now(1), date_from_now(1), maket_ticker)
                index_fundamental_rate = index_fundamental['등락률'].values[-1]
                index_fundamental_per = index_fundamental['PER'].values[-1]
                index_fundamental_pbr = index_fundamental['PBR'].values[-1]
                index_fundamental_div = index_fundamental['배당수익률'].values[-1]
                temp_df = pd.DataFrame([[market_name,index_fundamental_rate,index_fundamental_per,index_fundamental_pbr,index_fundamental_div,stocks_in_market]])
                result_df = pd.concat([result_df,temp_df])
        result_df.columns = ['시장명','등락률','PER','PBR', "DIV","편입종목"]
        with open(f'../data/market_fundamental_{date_from_now()}.pkl', 'wb') as f:
            pickle.dump(result_df, f)
            
        # 리스트를 str로
        result_df['편입종목'] = result_df['편입종목'].str.join("_")
        cond1 = result_df['편입종목'].str.contains(stock_ticker)
        result_df = result_df[cond1]
         # 시장이 있는 경우
        if list(result_df['시장명'].values):
            market_name = result_df['시장명'].values[-1]
            market_rate = result_df['등락률'].values[-1]
            market_per = result_df['PER'].values[-1]
            market_pbr = result_df['PBR'].values[-1]
            market_div = result_df['DIV'].values[-1]
        else:
            market_name = "-"
            market_rate = "-"
            market_per = "-"
            market_pbr = "-"
            market_div = "-"
        res = {'market_name':market_name,
               'rate':market_rate,
               'PER':market_per,
               'PBR':market_pbr,
               'DIV':market_div}
        return res
    
    
    
    else:
        # load
        with open(f'../data/market_fundamental_{date_from_now()}.pkl', 'rb') as f:
            result_df = pickle.load(f)
        # 리스트를 str로
        result_df['편입종목'] = result_df['편입종목'].str.join("_")
        cond1 = result_df['편입종목'].str.contains(stock_ticker)
        result_df = result_df[cond1]
         # 시장이 있는 경우
        if list(result_df['시장명'].values):
            market_name = result_df['시장명'].values[-1]
            market_rate = result_df['등락률'].values[-1]
            market_per = result_df['PER'].values[-1]
            market_pbr = result_df['PBR'].values[-1]
            market_div = result_df['DIV'].values[-1]
        else:
            market_name = "-"
            market_rate = "-"
            market_per = "-"
            market_pbr = "-"
            market_div = "-"
        res = {'market_name':market_name,
               'rate':market_rate,
               'PER':market_per,
               'PBR':market_pbr,
               'DIV':market_div}
        return res


if __name__=='__main__':   
    for market in ["KOSPI","KOSDAQ"]:    
        result_df = pd.DataFrame()
        # 티커 호출
        tickers = stock.get_market_ticker_list(date_from_now(), market=market)
        for ticker in tqdm(tickers[2:3]):  # tickers[1270] 오류 있음  
                temp_df = get_stock_price_info(ticker)
                result_df = pd.concat([result_df,temp_df])
        result_df.columns = temp_df.columns
        result_df['market'] = market
        
        with open(f'../data/stock_price_data_all_period_{market}.pkl', 'wb') as f:
            pickle.dump(result_df, f)

    # 임시저장 파일 load
    with open('../data/stock_price_data_all_period_KOSPI_2.pkl', 'rb') as f:
        kospi_df = pickle.load(f)
    with open('../data/stock_price_data_all_period_KOSDAQ_2.pkl', 'rb') as f:
        kosdaq_df = pickle.load(f)
        
    all_market_df = pd.concat([kospi_df,kosdaq_df])
    # 최종 파일 생성
    with open(f'../data/stock_price_data_all_period_ALL_2.pkl', 'wb') as f:
        pickle.dump(all_market_df, f)