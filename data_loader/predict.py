from pykrx import stock
from pykrx import bond
from loader import date_from_now, get_stock_basic_info, get_stock_price_info
from tqdm import tqdm
import params
import pandas as pd
import numpy as np
import model

# df = bond.get_otc_treasury_yields(date_from_now(), date_from_now(), "국고채2년")
# print(df.head())


# 종목명을 넣으면 vector로 변환하여 day동안의 모델을 적용한 수익률과 실제수익률을 비교하는 함수를 출력
def get_backtest_yeild_with_name(name, market ="ALL"  ,day=params.YEAR_TO_DAY*3):
    print("get_stock_basic_info....")
    all_stock_df = get_stock_basic_info()
    cond = all_stock_df['종목명'] == name
    ticker = all_stock_df.loc[cond,'티커'].values[0]
    print("get_stock_price_info....")
    result_df = get_stock_price_info(ticker,market,"BASIC", day)
    result_df = result_df['종가']
    print("initializing simulation....")
    # 현재 종목보유상태 초기화
    trading_position = "N"
    # 시작 보유 자산
    start_trading_price = 100
    trading_price = 100
    # 현재 보유 주수
    trading_amount = 0
    # 현재 수익률 초기화
    current_yeild = 0
    print("Back_test....")
    for idx in tqdm(range(len(result_df)-(params.ANALYSIS_DAY+params.PERIOD_YEILD_DAY)+1)):
        row_vector = result_df.iloc[idx:idx+params.ANALYSIS_DAY+params.PERIOD_YEILD_DAY]
        vector_df = row_vector.iloc[:params.ANALYSIS_DAY]
        vector_df_nomalized = vector_df/(vector_df).mean()
        buy_price = row_vector.values[params.ANALYSIS_DAY]
        sell_price = row_vector.values[-1]  
        period_yeild = round(((sell_price-buy_price)/buy_price)*100,2)
        # list로(vector) 변환: features + target  
        vector_df_nomalized = vector_df_nomalized.values
        vector_df_nomalized = np.append(vector_df_nomalized, period_yeild)
        vector_df_nomalized = pd.DataFrame([vector_df_nomalized])  
        yeild_prediction = model.final_lgb_model.predict(vector_df_nomalized.iloc[:,:20])[0] 
        
        # 매매 트레이딩 시뮬레이션
        # 현재 주식이 없다면
        if trading_position=="N":
            # 만약 매수 허들을 넘었다면 매수,
            if yeild_prediction > params.TRADING_HURDLE[0]:
                # 보유 상태로 변경
                trading_position = "Y"
                trading_amount = trading_price / buy_price
                trading_price = 0
            # 만약 매도 허들 이하라면 무시,
            elif yeild_prediction < params.TRADING_HURDLE[1]:
                continue
            # 만약 중립구간이라면 무시
            else:
                continue
        elif trading_position=="Y":
            # 만약 매수 허들을 넘었다면 무시,
            if yeild_prediction > params.TRADING_HURDLE[0]:
                continue
            # 만약 매도 허들 이하라면 매도,
            elif yeild_prediction < params.TRADING_HURDLE[1]:
                trading_position = "N"
                trading_price = trading_amount*buy_price
                trading_amount = 0
            # 만약 중립구간이라면 무시
            else:
                continue
            
        # 현재 투자전략 수익률
        # 현재 보유중이라면
        if trading_position=="Y":
            current_yeild = (trading_amount*buy_price)/start_trading_price - 1
        # 현재 보유중이지 않다면
        elif trading_position=="N":
            current_yeild = trading_price/start_trading_price - 1

    # Backtest_yeild 연평균으로 환산
    backtest_yeild = (1+current_yeild)**(1/(day/params.YEAR_TO_DAY))
    # Backtest_yeild
    backtest_yeild = round(current_yeild*100,2)

    print("=============================================")
    print(f"{name}_Back_test 연환산수익률 {backtest_yeild:.2f}%")
    print(f"당일기준 {params.ANALYSIS_DAY}일 후 예상 보유수익률 {yeild_prediction:.2f}%")
    
    # 추천 매매포지션
    recommend_position = ""
    # 투자전략 효과성
    invest_efficiency = 0
    
    if (backtest_yeild > 0) and (yeild_prediction >params.TRADING_HURDLE[0]):
        recommend_position = "매수"
        invest_efficiency = 1
        print("BUY!!! 매수전략이 유리합니다.")
    elif (backtest_yeild >= 0):
        recommend_position = "홀딩/관망"
        invest_efficiency = 1
        print("투자전략이 효과적이지만 현재 매수적기는 아닙니다.")
    elif (backtest_yeild < 0) and (yeild_prediction <params.TRADING_HURDLE[1]):
        recommend_position = "매도"
        print("SELL!!! 매도전략이 유리합니다.")
    elif (backtest_yeild < 0):
        recommend_position = "홀딩/관망"
        print("투자하기에 현재 투자전략이 부적절합니다.")
    print("=============================================")
    return [backtest_yeild, yeild_prediction, recommend_position, invest_efficiency]

# invest_efficiency 가 1이면서, recommend_position이 '매수'인것만 추천해줄것!
print(get_backtest_yeild_with_name("푸른저축은행"))

