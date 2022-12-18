import streamlit as st
import pandas as pd
from predict import get_backtest_yeild_with_name, get_info
import random


st.title('지금 투자해도 될까?')

stock_ls = ['삼성전자','현대차','NAVER','카카오']
rand_stock = random.choice(stock_ls)

stock_name = st.text_input('종목명을 입력하세요', value=rand_stock)
st.markdown('----')
buy_cond = st.number_input('매수조건 수익률을 입력하세요. (단위 : %)', value=4)
st.write(f"{buy_cond}% 이상 수익이 예측되는 경우 매수하도록 설정합니다.")
st.markdown('----')
sell_cond = st.number_input('매도조건 수익률을 입력하세요. (단위 : %)', value=-4)
st.write(f"{sell_cond}% 이상 손실이 예측되는 경우 매도하도록 설정합니다.")
st.markdown('----')
analysys_year = st.number_input('시뮬레이션 기간을 입력하세요. (단위 : 년)', value=1)
st.write(f"설정된 투자 전략으로 {analysys_year}년의 기간동안 시뮬레이션을 실행합니다.")


# 버튼 설계
if st.button("시뮬레이션"):
    con = st.container()
    st.markdown("---------")
    with st.spinner('Wait for it...'):
        result = get_backtest_yeild_with_name(stock_name, buy_cond, sell_cond, analysys_year)
        
        backtest_yeild = result[0]
        yeild_prediction = result[1]
        recommend_position = result[2]
        invest_efficiency = result[3]
        stock_price_info = result[4]
        stock_fundamental_info = result[5]
        invest_alert = ""
        if invest_efficiency==1:
            invest_efficiency = "투자전략 적합"
            invest_alert = "적합"
            
        elif invest_efficiency==0:
            invest_efficiency = "투자전략 부적합"
            invest_alert = "부적합"


        
        result_ls = [[backtest_yeild, yeild_prediction, recommend_position, invest_efficiency]]    
        result_df = pd.DataFrame(result_ls)
        result_df.columns =[ f"{analysys_year}년 Back_test 연환산수익률(%)" , "5일 이후 예측수익률(%)",  "추천 매매포지션" , "투자전략 유효성"]
        
        col1,col2 = st.columns(2)
        col1.metric("투자전략 유효성",f"{result_df['투자전략 유효성'].values[0]}")
        col1.write(">연환산수익률이 기준금리보다 높다면, 효과적인 전략으로 판단합니다.")
        col2.metric(f"{analysys_year}년 Back_test 연환산수익률(%)",f"{result_df[f'{analysys_year}년 Back_test 연환산수익률(%)'].values[0]:.2f}")
        col2.write(">Back_test 연환산수익률(%)은 현재 모델의 매매 시, 얻은 수익률로 실제 수익률과 다릅니다.")

        st.markdown("---------")
        if invest_alert=='부적합':
            st.write('투자전략이 부적합한 경우에는 예측수익률을 제공하지 않습니다.')
        else:  
            col1,col2 = st.columns(2)
            col1.metric(f"5일 이후 예측수익률(%)",f"{result_df['5일 이후 예측수익률(%)'].values[0]:.2f}")
            col1.write(">시뮬레이션에 사용된 모델을 활용한 예측결과입니다.  투자전략이 부적합하다면 신뢰할 수 없습니다.")
            col2.metric("추천 매매포지션",f"{result_df['추천 매매포지션'].values[0]}")        
        
        # 기타 정보제공
        st.markdown("---------")
        st.caption("주가 관련정보")
        o_price_yesterday = stock_price_info['시가'].iloc[0]
        h_price_yesterday = stock_price_info['고가'].iloc[0]
        l_price_yesterday = stock_price_info['저가'].iloc[0]
        c_price_yesterday = stock_price_info['종가'].iloc[0]
        o_price = stock_price_info['시가'].iloc[1]
        h_price = stock_price_info['고가'].iloc[1]
        l_price = stock_price_info['저가'].iloc[1]
        c_price = stock_price_info['종가'].iloc[1]
        
        col1, col2  = st.columns(2)
        col1.metric("종가", str(c_price), str(f"{((c_price-c_price_yesterday)/c_price_yesterday)*100:.2f}%"))
        
        col1, col2, col3 = st.columns(3)
        col1.metric("시가", str(o_price), str(o_price-o_price_yesterday))
        col2.metric("고가", str(h_price), str(h_price-h_price_yesterday))
        col3.metric("저가", str(l_price), str(l_price-l_price_yesterday))

        
        st.markdown("---------")

        st.caption("펀더멘탈 정보")
        per = stock_fundamental_info.iloc[:,0:1].values[0][0]
        pbr = stock_fundamental_info.iloc[:,1:2].values[0][0]

        per_result = "좋다"
        pbr_result = "-나쁘다"
        
        col1, col2 = st.columns(2)
        col1.metric("PER", f"{per:.2f}", per_result)
        col2.metric("PBR", f"{pbr:.2f}", pbr_result)
        
        top_bottom = get_info()
        top10 = top_bottom[0]
        bottom10 = top_bottom[1]
        

        st.markdown("---------")
        st.markdown('상승률 상위')
        st.dataframe(top10)
        st.markdown("하락률 상위")
        st.dataframe(bottom10)
        
              

    # 간단 진단
    # 초기화
    # summary = "" 
    # if    
    # if (backtest_yeild > 0) and (yeild_prediction >params.TRADING_HURDLE[0]):
    #     recommend_position = "매수"
    #     summary = "BUY!!! 매수전략이 유리합니다."
    # elif (backtest_yeild >= 0):
    #     recommend_position = "홀딩/관망"
    #     summary = "투자전략이 효과적이지만 현재 매수적기는 아닙니다."
    # elif (backtest_yeild < 0) and (yeild_prediction <params.TRADING_HURDLE[1]):
    #     recommend_position = "매도"
    #     summary = "SELL!!! 매도전략이 유리합니다."
    # elif (backtest_yeild < 0):
    #     recommend_position = "홀딩/관망"
    #     summary = "투자하기에 현재 투자전략이 부적절합니다."
        
    # con.write(summary)
    
    
    
    
    # 예측수익률을 기준으로 한 예상 주가 차트
    # 투자전략이 효과적인 오늘의 종목
    # 투자
    #streamlit run deploy.py