from pykrx import stock
from pykrx import bond
from loader import get_stock_basic_info
import joblib
from tqdm import tqdm
import params
import pandas as pd
import numpy as np
# from mplfinance.original_flavor import candlestick_ohlc
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.dates as mdates


def ratio_judge(x, critical_point, diretion="F"):
    ''' 지표를 판단해주는 함수
    현재 지표값, 판단의 임계점, 해석의 방향(높을수록 좋다면 F, 낮을수록 좋다면 R))
    '''
    judge_high = ""
    judge_low = ""
    if diretion == "F":
        judge_high = "고평가"
        judge_low = "저평가"
    elif diretion == "R":
        judge_high = "저평가"
        judge_low = "고평가"
        
    if x>= critical_point:
        return judge_high
    elif x<=-critical_point:
        return judge_low
    else:
        return "보통"


def get_high_low_info():
    """
    주식의 모든 정보를 가져와 정렬 후, 상위/하위 종목을 반환하는 함수
    """
    all_stock_df = get_stock_basic_info()
    # 등락률 상위종목
    top10 = all_stock_df.sort_values(by='등락률',ascending=False)[['종목명','종가','등락률']].iloc[:10,:]
    bottom10 = all_stock_df.sort_values(by='등락률',ascending=True)[['종목명','종가','등락률']].iloc[:10,:]
    return [top10, bottom10]


# # 주가 dataframe을 넣으면 차트를 반환하는 함수
# def get_chart(name, price_info_df, buy_date=[], sell_date=[]):
#     price_df = price_info_df
#     # date2num으로 날짜 변경
#     price_df["날짜"] = mdates.date2num(price_df["날짜"].values)
    
#     fig = plt.figure(figsize=(8, 5))
#     fig.set_facecolor('w')
#     gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
#     axes = []
#     axes.append(plt.subplot(gs[0]))
#     axes.append(plt.subplot(gs[1], sharex=axes[0]))
#     axes[0].get_xaxis().set_visible(False)
#     x = np.arange(len(price_df.index))
#     ohlc = price_df[['시가', '고가', '저가', '종가']].astype(int).values
#     dohlc = np.hstack((np.reshape(x, (-1, 1)), ohlc))
#     # 봉차트
#     candlestick_ohlc(axes[0], dohlc, width=0.5, colorup='r', colordown='b')

#     plt.tight_layout()
#     plt.show()




# 종목명을 넣으면 vector로 변환하여 day동안의 모델을 적용한 수익률과 실제수익률을 비교하는 함수를 출력
def get_backtest_yeild_with_name(name, buy_cond, sell_cond, year, price_info_df,model):
    result_df = price_info_df
    # get_chart(name,price_info_df)
    
    stock_price_info_close = result_df.iloc[:,1:5]['종가']
    stock_price_info_2days = result_df.iloc[-2:,1:5]
    stock_fundamental_info = result_df.iloc[-1:,-6:-2]
    stock_fundamental_info.round(2)
    del stock_fundamental_info['EPS']

    result_df = result_df['등락률']
    
    # print("initializing simulation....")
    # 현재 종목보유상태 초기화
    trading_position = "N"
    # 시작 보유 자산
    start_trading_price = 1000000
    trading_price = 1000000
    # 현재 보유 주수
    trading_amount = 0
    # 현재 수익률 초기화
    current_yeild = 0
    # 매수 횟수
    buy_cnt = 0
    # 매도 횟수
    sell_cnt = 0
    # 매수 가격 리스트
    buy_list = []
    # 매도 가격 리스트
    sell_list = []
    # 예측수익률 리스트
    yeild_prediction_list = []
    # 보유일자 리스트
    holding_day_ls = []
    # 보유일자 초기화
    holding_day = 0
    # print("Back_test....")

    if model =="naive":
        final_lgb_model = joblib.load(".\\model\\lgbm_model_0.60_0.60_iter_2169_day_5.pkl")
    elif model=="deep":
        final_lgb_model = joblib.load(".\\model\\lgbm_model_0.28_0.27_iter_2749_day_5.pkl")
    elif model=="robust":    
        final_lgb_model = joblib.load(".\\model\\lgbm_model_0.30_0.30_iter_2651_day_5.pkl")
     

    for idx in tqdm(range(len(result_df)-(params.ANALYSIS_DAY+params.PERIOD_YEILD_DAY)+1)):
        row_vector = result_df.iloc[idx:idx+params.ANALYSIS_DAY+params.PERIOD_YEILD_DAY]
        price_vector = stock_price_info_close.iloc[idx:idx+params.ANALYSIS_DAY+params.PERIOD_YEILD_DAY]
        vector_df = row_vector.iloc[:params.ANALYSIS_DAY]
        buy_price = price_vector.values[params.ANALYSIS_DAY]
        sell_price = price_vector.values[-1]  
        period_yeild = round(((sell_price-buy_price)/buy_price)*100,2)
        # list로(vector) 변환: features + target  
        vector_df = vector_df.values
        vector_df = np.append(vector_df, period_yeild)
        vector_df = pd.DataFrame([vector_df])  
        # print(vector_df)

        yeild_prediction = final_lgb_model.predict(vector_df.iloc[:,:params.ANALYSIS_DAY])[0] 
        yeild_prediction_list.append(yeild_prediction)
        # print(f"예상 수익률: {yeild_prediction}")
        


        # 매매 트레이딩 시뮬레이션
        # 현재 주식이 없다면
        if trading_position=="N":
            # 만약 매수 허들을 넘었다면 매수,
            if yeild_prediction >= buy_cond:
                # 보유 상태로 변경
                
                trading_position = "Y"
                holding_day = 1
                trading_amount = trading_price / buy_price
                # print(f"{trading_amount:.1f}주 주당{buy_price:.1f}원 매수 , 평가금액: {trading_price:.1f}원")
                buy_list.append(buy_price)
                trading_price = 0
                buy_cnt+=1
                
            # 만약 매도 허들 이하라면 무시,
            elif yeild_prediction <=sell_cond:
                if holding_day >=1:
                    holding_day += 1
                continue
            # 만약 중립구간이라면 무시
            else:
                if holding_day >=1:
                    holding_day += 1
                continue
        elif trading_position=="Y":
            # 만약 매수 허들을 넘었다면 무시,
            if yeild_prediction >= buy_cond:
                if holding_day >=1:
                    holding_day += 1
                continue
            # 만약 매도 허들 이하라면 매도,
            elif yeild_prediction <= sell_cond:
                # 보유일자 저장
                holding_day_ls.append(holding_day)
                # 보유일자 초기화
                holding_day = 0
                trading_position = "N"
                trading_price = trading_amount*sell_price
                # print(f"{trading_amount:.1f}주 주당{sell_price:.1f}원 매도 , 평가금액: {trading_price:.1f}원")
                sell_list.append(buy_price)
                trading_amount = 0
                sell_cnt+=1
                
            # 만약 중립구간이라면 무시
            else:
                if holding_day >=1:
                    holding_day += 1
                continue

        # 현재 투자전략 수익률
        # 현재 보유중이라면
        if trading_position=="Y":
            current_yeild = (trading_amount*sell_price)/start_trading_price - 1
        # 현재 보유중이지 않다면
        elif trading_position=="N":
            current_yeild = trading_price/start_trading_price - 1
        # print(f"현재 수익률은 {current_yeild*100:.2f}%입니다.")
    
    # Backtest_yeild 연평균으로 환산
    backtest_yeild = ((1+current_yeild)**(1/(year))-1)*100
    
    # Backtest_yeild
    backtest_yeild = round(backtest_yeild,2)

    # print(f"{name}_Back_test 연환산수익률 {backtest_yeild:.2f}%")
    # print(f"{params.PERIOD_YEILD_DAY}일 후 예상 보유수익률 {yeild_prediction:.2f}%")
    # print(f"매수횟수 : {buy_cnt}", f"매도횟수 : {sell_cnt}", f"매매횟수 : {buy_cnt+sell_cnt}")
    
    # 투자전략 효율성 체크 초기화
    invest_efficiency = 0
    
    # backtest 수익률이 기준금리보다 높고, 시뮬레이션 조건 매수수익률보다 높다면,
    if (backtest_yeild >= params.KOLIBOR) and (yeild_prediction > buy_cond):
        invest_efficiency = 1
        # print("BUY!!! 매수 추천")
    # backtest 수익률이 기준금리보다 높지만, 시뮬레이션 조건 매수수익률보다 낮다면,    
    elif (backtest_yeild >= params.KOLIBOR):
        invest_efficiency = 1
        # print("투자전략이 효과적이지만 현재 매수시기는 아닙니다.")
    # backtest 수익률이 기준금리보다 낮고, 시뮬레이션 조건 매도수익률보다 낮다면, 
    elif (backtest_yeild < params.KOLIBOR) and (yeild_prediction < sell_cond):
        # print("SELL!!! 매도 추천")
        invest_efficiency = 0
    # backtest 수익률이 기준금리보다 낮지만, 시뮬레이션 조건 매도수익률보다 높다면, 
    elif (backtest_yeild < params.KOLIBOR):
        invest_efficiency = 0
        # print("투자하기에 현재 투자전략이 부적절합니다.")
    
    return [backtest_yeild, yeild_prediction, invest_efficiency, stock_price_info_2days, stock_fundamental_info, buy_cond, sell_cond, buy_list, sell_list, result_df.values,yeild_prediction_list,holding_day_ls]