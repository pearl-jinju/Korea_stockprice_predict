from pykrx import stock
import pandas as pd
import datetime
from tqdm import tqdm
import time
import requests
from bs4 import BeautifulSoup
import numpy as np
import re
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


def get_stock_basic_info(day=0, market="ALL", detail="ALL"):
    """ 종목 기초정보 제공 함수
        티커/종목명/시가/종가/변동폭(오늘)/등락률(오늘)/\n
        거래량(오늘)/거래대금(오늘)/상장주식수/보유수량/\n
        지분율/한도수량/한도소진률/BPS/PER/PBR/EPS/DIV/DPS_
        
        All = 모든 정보
        BASIC = 기초 정보

    Args:
        day (int, optional): _description_. Defaults to 0.
        market (str, optional): _description_. Defaults to "ALL".
        detail (str, optional): _description_. Defaults to "ALL".

    Returns:
        _type_: DataFrame
    """
    if detail=="ALL":
        df_stake_info = stock.get_exhaustion_rates_of_foreign_investment(date_from_now(), market).reset_index()
        df_change = stock.get_market_price_change(date_from_now(day),date_from_now(), market=market).reset_index()
        df_result = pd.merge(df_change,df_stake_info, on='티커',how='left')
        df_fundamental = stock.get_market_fundamental( date_from_now(), market=market).reset_index()
        df_result = pd.merge(df_result,df_fundamental, on='티커',how='left')
        return df_result
    if detail=="BASIC":
        df_change = stock.get_market_price_change(date_from_now(day),date_from_now(), market=market).reset_index()
        return df_change



def get_stock_price_info(ticker, market="ALL", info="ALL", day=50000):
    """
    종목명을 받으면 기간동안의 주가/거래량/수급/기본Fundamental 지표 변동을 출력해주는 함수(day default 종목 전체기간 데이터 호출)
    info="ALL" 거래량과 펀더멘탈 정보 포함
    info="BASIC" 기본 정보만 포함
    """
    df_stock_basic_info = get_stock_basic_info()
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

def get_market_basket():
        market_tickers_kospi = stock.get_index_ticker_list(market='KOSPI')
        market_tickers_kosdaq = stock.get_index_ticker_list(market='KOSDAQ')
        result_df = pd.DataFrame()
        market_tickers_all = market_tickers_kospi+market_tickers_kosdaq
        for market_ticker in tqdm(market_tickers_all):
            # 시장명 str
            market_name = stock.get_index_ticker_name(market_ticker)
            stocks_in_market = stock.get_index_portfolio_deposit_file(str(market_ticker))
            temp_df = pd.DataFrame([[market_ticker,market_name,stocks_in_market]])
            result_df = pd.concat([result_df,temp_df])
        result_df.columns = ['시장티커','시장명','편입종목']
        result_df.reset_index(inplace=True,drop=True)
        # save
        with open('market_basket.pkl', 'wb') as f:
            pickle.dump(result_df, f)


def get_stocks_fundamental_info(stock_ticker):
    """_ 
    종목의 티커를 입력하면 그 종목이 소속된 시장의 funadamental 지표를 반환하는 함수 \n

    Args:
        stock_ticker (_type_): _description_
    """
    with open('market_basket.pkl', 'rb') as f:
        raw_basket_df = pickle.load(f)
        raw_basket_df['편입종목'] = raw_basket_df['편입종목'].str.join("_")

    # 코스피/코스닥 관련 바스켓이름이 나오지 않도록 (너무 큰 바스켓이므로)
    cond = (raw_basket_df['시장명'].str.contains("코스피")==False) & (raw_basket_df['시장명'].str.contains("코스닥")==False)
    basket_df = raw_basket_df[cond]

    # 없다면, 코스피/코스닥 관련 바스켓이 나오도록
    if basket_df.empty==True:
        basket_df = raw_basket_df[cond]

    cond = basket_df['편입종목'].str.contains(stock_ticker)
    cond_basket_df = basket_df.loc[cond][['시장티커','시장명']]
    basket_names = cond_basket_df['시장명'].values
    basket_tickers = cond_basket_df['시장티커'].values
    # 맨마지막 시장만 사용 (추후 수정)0
    try:
        basket_name = basket_names[0]
        basket_ticker = basket_tickers[0]
        basket_funadamental = stock.get_index_fundamental(date_from_now(1), date_from_now(1), basket_ticker)
        basket_funadamental_rate = basket_funadamental['등락률'].values[-1]
        basket_funadamental_per = basket_funadamental['PER'].values[-1]
        basket_funadamental_pbr = basket_funadamental['PBR'].values[-1]
        basket_funadamental_div = basket_funadamental['배당수익률'].values[-1]
        result_df = pd.DataFrame([[basket_name,basket_funadamental_rate,basket_funadamental_per,basket_funadamental_pbr,basket_funadamental_div]])
        result_df.columns = ['시장명','등락률','PER','PBR', "DIV"]
        
        res = {'market_name':basket_name,
                'rate':basket_funadamental_rate,
                'PER':basket_funadamental_per,
                'PBR':basket_funadamental_pbr,
                'DIV':basket_funadamental_div}
        return res

    except:
        res = {'market_name':"-",
                'rate':"-",
                'PER':"-",
                'PBR':"-",
                'DIV':"-"}
        return res
    

def get_thema():
    #크롤링 차단 막기
    seed = np.random.randint(100)
    np.random.seed(seed)
    delay = np.random.randint(5)/100
    
    theama_idx=1
    idx_max = 750
    stock_idx_max = 100

    result_df = pd.DataFrame(columns=['테마명',"종목명", "테마 사유"])
    while theama_idx < idx_max:
        # 테마 조회
        search_url = f'https://finance.naver.com/sise/sise_group_detail.naver?type=theme&no={theama_idx}'
        resp = requests.get(search_url)
        soup = BeautifulSoup(resp.text, 'html.parser')
        s = soup.select_one
        time.sleep(delay)

        try:
            thema_name = s('head > title').text.split(':')[0] # re.sub(r"/", "_", s('head > title').text.split(':')[0])
            theama_idx+=1
            stock_idx=1
            
            while stock_idx<stock_idx_max:
                try:
                    stock_name = re.sub(r"&",r"&",re.sub(r"\s", "", re.sub(r"\t", "",s(f'#contentarea > div:nth-child(5) > table > tbody > tr:nth-child({stock_idx}) > td:nth-child(2) > div > div > strong').text)).split('\n')[0])
                    # print(stock_name)
                    stock_detail = re.sub(r"\n\n", "", re.sub(r"\t", "",s(f'#contentarea > div:nth-child(5) > table > tbody > tr:nth-child({stock_idx}) > td:nth-child(2) > div > div > p').text))
                    data = pd.DataFrame([[thema_name,stock_name, stock_detail]])
                    data.columns = ['테마명',"종목명", "테마 사유"]
                    stock_idx+=1
                    result_df = pd.concat([result_df, data])
                except AttributeError:
                    break

        except AttributeError:
            print(f"pass{theama_idx}/{idx_max}")
            theama_idx+=1
            continue
        
    result_df = result_df.reset_index(drop=True)
        
    # save
    with open('thema_data.pkl', 'wb') as f:
        pickle.dump(result_df, f)
        
        
def get_included_thema_stocks_in_thema(name:str):
    """종목명을 입력하면,\n
       그 종목이 포함된 테마종류와,\n
       테마 내에 속한 종목들의 테마사유, 종가, 등락률을 반환합니다.

    Args:
        name (str): _description_
    """
    stock_name = name

    with open('thema_data.pkl', 'rb') as f:
        df = pickle.load(f)
    #종목 기초정보
    df_change = get_stock_basic_info(day=0, market="ALL", detail="BASIC")
    df_change = df_change.iloc[:,1:6]
    del df_change['시가']
    del df_change['변동폭']
        
    # 인덱스 재정렬    
    df = df.reset_index(drop=True)
    # 입력된 종목명으로 필터링
    cond = df['종목명']==stock_name
    # 종목이 포함된 테마 출력
    thema_df = df[cond]
    included_thema =  thema_df['테마명'].values
    
    stocks_in_thema_result =pd.DataFrame()
    for thema in included_thema:
        cond = df['테마명']==thema
        stocks_in_thema = df[cond]
        stocks_in_thema = pd.merge(stocks_in_thema,df_change, on="종목명", how='left')
        stocks_in_thema = stocks_in_thema.sort_values(by='등락률', ascending=False)
        stocks_in_thema_result = pd.concat([stocks_in_thema_result,stocks_in_thema])
        
    stocks_in_thema_result.columns = stocks_in_thema.columns
    stocks_in_thema_result = stocks_in_thema_result[['테마명','종목명','종가','등락률','테마 사유']]


    return [included_thema, stocks_in_thema_result]


if __name__=='__main__': 
    """
    테마 데이터
    마켓 바스켓 정보를 최신화하기 위해 한번씩 실행
    """
      
    # 테마 정보 받아오기
    get_market_basket()
    # get_thema()
    # get_index_fundamental_info('005930')
    
    # for market in ["KOSPI","KOSDAQ"]:    
    #     result_df = pd.DataFrame()
    #     # 티커 호출
    #     tickers = stock.get_market_ticker_list(date_from_now(), market=market)
    #     for ticker in tqdm(tickers[2:3]):  # tickers[1270] 오류 있음  
    #             temp_df = get_stock_price_info(ticker)
    #             result_df = pd.concat([result_df,temp_df])
    #     result_df.columns = temp_df.columns
    #     result_df['market'] = market
        
    #     with open(f'.\data\stock_price_data_all_period_{market}.pkl', 'wb') as f:
    #         pickle.dump(result_df, f)

    # # 임시저장 파일 load
    # with open(r'.\\data\\raw_data\\stock_price_data_all_period_KOSPI_2.pkl', 'rb') as f:
    #     kospi_df = pickle.load(f)
    # with open(r'.\\data\\raw_data\\stock_price_data_all_period_KOSDAQ_2.pkl', 'rb') as f:
    #     kosdaq_df = pickle.load(f)
        
    # all_market_df = pd.concat([kospi_df,kosdaq_df])
    # # 최종 파일 생성
    # with open(f'.\\data\\raw_data\\stock_price_data_all_period_ALL_2.pkl', 'wb') as f:
    #     pickle.dump(all_market_df, f)
        
