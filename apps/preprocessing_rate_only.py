import pickle
import pandas as pd
import numpy as np
from pykrx import stock
from pykrx import bond
from datetime import datetime
from loader import date_from_now, get_stock_basic_info, get_stock_price_info
from tqdm import tqdm
import params






def patten_to_vector_rate(analysis_day:int=params.ANALYSIS_DAY, period_yeild_day:int=params.PERIOD_YEILD_DAY, year_to_day:int=params.YEAR_TO_DAY,new_listing_reduction_year:int=params.NEW_LISTING_REDUCTION_YEAR):
    # 데이터 로드
    data_path =  "..\\data\\stock_price_data_all_period_20221209.pickle"

    with open(data_path, 'rb') as f:
        df_raw_data_all = pickle.load(f)

    # 주가 형태 분석 랜덤 추출 pivot(index='종목명', columns='날짜', values='종가')
    df_price_period = df_raw_data_all.pivot(index='종목명', columns='날짜', values='등락률').transpose()
    
    stock_name_list = df_price_period.columns
    # 임베딩 함수
    result_df = pd.DataFrame()
    
    # 모든 종목 정보조회!
    # all_stock_df = get_stock_basic_info()
    #모든 종목 조회
    for idx in tqdm(range(len(df_price_period.columns))):
    # for idx in tqdm(range(1)):
        try:
            # 종목 추출
            nan_row = df_price_period.iloc[:, idx:idx+1]
            nan_row_copy = nan_row.copy()
            # 컬럼명 변경, X일 이동평균 컬럼 추가
            nan_row_copy.columns = ["등락률"]
            # 이동평균 적용 후, 결측기간 제거
            row = nan_row_copy.dropna(axis=0)
            business_year = int((len(row)/year_to_day))
            # 년단위로 조절하기 위해 252 나머지를 trim(1~year_to_day)
            # 신규상장 효과를 배제하기 위해 new_listing_reduction_year년을 추가 trim
            # 이동평균계산을 위해 추가 trim                     
            date_trim = int(year_to_day*new_listing_reduction_year)
            row = row.iloc[date_trim:,:]
            # 종목명 저장
            stockname = stock_name_list[idx]
            # 결측치제거
            row = row.dropna(axis=0)
            # # 종목의 기본 정보 가져오기
            # cond = all_stock_df['종목명'] == stockname
            # ticker = all_stock_df.loc[cond,'티커'].values[0]
            # stock_fundamental_info = get_stock_price_info(ticker,"ALL","ALL", 50000)
            # stock_fundamental_info = stock_fundamental_info.iloc[:,-6:-2]
            # del stock_fundamental_info['EPS']
                       
            
            # 1년 단위로 반복!
            temp_df = pd.DataFrame()
            for year in tqdm(range(business_year-1)):
                # 1일 단위로 반복! 1년 기준일 -(분석범위 + 예측할 수익률기간)
                for day in range(1,year_to_day-(analysis_day+period_yeild_day)+1):    
                    row_per_year = row.iloc[(year*year_to_day+day):(year*year_to_day+day)+(analysis_day+period_yeild_day),:]
                    # features(price, average_price)
                    analysis_range = row_per_year['등락률'].iloc[:analysis_day]
                    # fundamental info 붙이기
                    # stock_fundamental_info_per_year = stock_fundamental_info.iloc[(year*year_to_day+day):(year*year_to_day+day)+(analysis_day+period_yeild_day),:].iloc[:analysis_day]
                    # per_list = stock_fundamental_info_per_year['PER'].values
                    # pbr_list = stock_fundamental_info_per_year['PBR'].values
                    # make target
                    period_yeild = 100
                    for i in range(len(analysis_range)):
                        yeild = analysis_range.iloc[i]
                        period_yeild = period_yeild*(1+yeild/100) 
                    period_yeild = period_yeild-100
                    # list로(vector) 변환: features + target  
                    price_value_list = analysis_range.values
                    price_vector = list(price_value_list) 
                    price_vector = np.round(price_vector,2)
                    # per_vector = list(per_list)
                    # per_vector = np.round(per_vector,2)
                    # pbr_vector = list(pbr_list)
                    # pbr_vector = np.round(pbr_vector,2)
                    # mp.array에 값 붙이기
                    result_vector = np.append(price_vector, period_yeild)
                    # result_vector = np.append(price_vector, per_vector)
                    # result_vector = np.append(result_vector, pbr_vector)
                    # result_vector = np.append(result_vector, period_yeild)
                    vector_df = pd.DataFrame([result_vector])        
                    temp_df = pd.concat([temp_df, vector_df])
                temp_df['종목명'] = stockname


            result_df = pd.concat([result_df,temp_df])
            
            with open(f'..\\data\\Korea_stock_Dataset_{analysis_day}_{period_yeild_day}_{year_to_day}_{new_listing_reduction_year}_RATE_only.pkl', 'wb') as f:
                pickle.dump(result_df, f)
            print(idx)
            
            
        except :
            print("오류")
            continue

    print(result_df)
    # save
    with open(f'..\\data\\Korea_stock_Dataset_{analysis_day}_{period_yeild_day}_{year_to_day}_{new_listing_reduction_year}_RATE_only.pkl', 'wb') as f:
        pickle.dump(result_df, f)
    
if __name__=='__main__':
    # 정상 용도
    # patten_to_vector_rate(50,5,252,1)
    # 불나방 용도
    patten_to_vector_rate(5,1,252,1)

# # 허스트 지수 산출