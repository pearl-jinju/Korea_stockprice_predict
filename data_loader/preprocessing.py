import pickle
import pandas as pd
import numpy as np
from pykrx import stock
from pykrx import bond
from datetime import datetime
from loder import date_from_now, get_stock_basic_info
from tqdm import tqdm



analysis_day=20
period_yeild_day=20
year_to_day = 252
moving_average_day=252
# 신규상장 감안 분석제거 년수
new_listing_reduction_year = 1

def patten_to_vector(analysis_day:int=20, period_yeild_day:int=20, year_to_day:int=252,moving_average_day:int=252,new_listing_reduction_year:int=1):
    # 데이터 로드
    data_path =  "../data/stock_price_data_all_period_20221209.pickle"

    with open(data_path, 'rb') as f:
        df_raw_data_all = pickle.load(f)

    # 주가 형태 분석 랜덤 추출 pivot(index='종목명', columns='날짜', values='종가')
    df_price_period = df_raw_data_all.pivot(index='종목명', columns='날짜', values='종가').transpose()

    # 임베딩 함수
    result_df = pd.DataFrame()
    # 모든 종목 조회!
    for idx in tqdm(range(len(df_price_period))):
        try:
            # 종목 추출
            nan_row = df_price_period.iloc[:, idx:idx+1]
            nan_row_copy = nan_row.copy()
            # 컬럼명 변경, X일 이동평균 컬럼 추가
            nan_row_copy.columns = ["종가"]
            nan_row_copy[f'{moving_average_day}일평균'] = nan_row_copy.copy()['종가'].rolling(moving_average_day).mean()
            # 이동평균 적용 후, 결측기간 제거
            row = nan_row_copy.dropna(axis=0)
            business_year = int((len(row)/year_to_day))
            # 년단위로 조절하기 위해 252 나머지를 trim(1~year_to_day)
            # 신규상장 효과를 배제하기 위해 new_listing_reduction_year년을 추가 trim
            # 이동평균계산을 위해 추가 trim                     
            date_trim = int((len(row)%year_to_day+year_to_day*new_listing_reduction_year))
            row = row.iloc[date_trim:,:]
            # 종목명 저장
            # stockname = row.columns[0]
            # 결측치제거
            row = row.dropna(axis=0)
            # 1년 단위로 반복!
            temp_df = pd.DataFrame()
            for year in range(business_year-3,business_year-1):
                # 1일 단위로 반복! 1년 기준일 -(분석범위 + 예측할 수익률기간)
                for day in range(1,year_to_day-(analysis_day+period_yeild_day)+1):    
                    row_per_year = row.iloc[(year*year_to_day+day):(year*year_to_day+day)+(analysis_day+period_yeild_day),:]
                    # features(price, average_price)
                    analysis_range = row_per_year['종가'].iloc[:analysis_day]
                    moving_average_range = row_per_year[f'{moving_average_day}일평균'].iloc[:analysis_day]
                    # make target
                    buy_price = row_per_year['종가'].values[analysis_day]
                    sell_price = row_per_year['종가'].values[-1]  
                    period_yeild = round(((sell_price-buy_price)/buy_price)*100,2)
                    # list로(vector) 변환: features + target  
                    price_value_list = analysis_range.values
                    moving_average_value_list = moving_average_range.values
                    price_vector = list(price_value_list) + list(moving_average_value_list)
                    # Nomalize
                    price_vector = price_vector/analysis_range.mean()
                    result_vector = np.append(price_vector, period_yeild)
                    vector_df = pd.DataFrame([result_vector])        
                    temp_df = pd.concat([temp_df, vector_df])


            result_df = pd.concat([result_df,temp_df])
        except:
            continue

    print(result_df)
    # save
    with open(f'../data/Korea_stock_Dataset_{analysis_day}_{period_yeild_day}_{year_to_day}_{moving_average_day}_{new_listing_reduction_year}.pickle', 'wb') as f:
        pickle.dump(result_df, f)
    print("vectorizing_COMPLETE")
    
patten_to_vector(20,5,252,224)

# 허스트 지수 산출