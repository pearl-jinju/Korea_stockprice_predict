import streamlit as st
from predict import get_backtest_yeild_with_name
import pandas as pd
import params

st.title('지금 투자해도될까?')

stock_name = st.text_input('종목명을 입력하세요', value="삼성전자")
analysis_day = st.number_input('분석기간(단위 : 년)을 입력하세요', value=3)
# 버튼 설계
if st.button("결과확인"):
    con = st.container()
    con.caption("예측결과")

    result = get_backtest_yeild_with_name(stock_name, analysis_day)
    backtest_yeild = result[0]
    yeild_prediction = result[1]
    recommend_position = result[2]
    invest_efficiency = result[3]

    if invest_efficiency==1:
        invest_efficiency = "투자전략 유효"
    elif invest_efficiency==0:
        invest_efficiency = "투자전략 유효하지 않음"



    result_ls = [[backtest_yeild, yeild_prediction, recommend_position, invest_efficiency]]

    result_df = pd.DataFrame(result_ls)

    result_df.columns =[ "Back_test 연환산수익률(%)" , "5일 이후 예측수익률(%)",  "추천 매매포지션" , "투자전략 유효성"]

    con.dataframe(result_df)

    # 간단 진단
    # 초기화
    summary = ""    
    if (backtest_yeild > 0) and (yeild_prediction >params.TRADING_HURDLE[0]):
        recommend_position = "매수"
        summary = "BUY!!! 매수전략이 유리합니다."
    elif (backtest_yeild >= 0):
        recommend_position = "홀딩/관망"
        summary = "투자전략이 효과적이지만 현재 매수적기는 아닙니다."
    elif (backtest_yeild < 0) and (yeild_prediction <params.TRADING_HURDLE[1]):
        recommend_position = "매도"
        summary = "SELL!!! 매도전략이 유리합니다."
    elif (backtest_yeild < 0):
        recommend_position = "홀딩/관망"
        summary = "투자하기에 현재 투자전략이 부적절합니다."
        
    con.write(summary)
    con.write("* Back_test 연환산수익률(%)은 현재 모델의 매매패턴을 적용했을 때 투자전략 수익률로 실제 수익률과 다릅니다.")
    
    
    
    # 예측수익률을 기준으로 한 예상 주가 차트
    # 투자전략이 효과적인 오늘의 종목
    # 투자
    #streamlit run deploy.py