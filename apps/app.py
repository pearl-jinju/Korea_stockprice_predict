import streamlit as st
import pandas as pd
from predict import get_backtest_yeild_with_name, get_high_low_info, ratio_judge
from loader import date_from_now, get_stock_basic_info, get_stock_price_info, get_index_fundamental_info, get_included_thema_stocks_in_thema
import random
from itertools import combinations
import params
import numpy as np
import time


# ========================================
# 매수 판단 예측수익률
buy_cond = 6
# 매도 판단 예측수익률
sell_cond = -2
# 목표 수익률
target_cond = 3.25
# 시뮬레이션 기간
analysys_year = 2

# ========================================



st.sidebar.header('MENU')
side_menu_name = st.sidebar.selectbox('사용할 기능을 선택하세요.',['주식 수익률 예측','종목관련 테마 조회','상승률/하락률 상위종목'])


st.title('지금 투자해도 될까?')
st.markdown('----')

if side_menu_name=='주식 수익률 예측':
    st.header("종목명만 입력하세요!")
    stock_name = st.text_input('  ', value="카카오")
    st.markdown('----')
    # st.header("Step 2 : 목표수익률 입력")
    # target_cond = st.number_input('목표 연간 수익률을 입력하세요. (단위 : %)', value=3.25)
    # st.markdown(f"* 현재 기준금리, 5년물 국채 , 또는 다른 지표들이 기준이 될수 있습니다.")
    # st.markdown('----')
    # st.header("Step 3 : 시뮬레이션 기간 입력")
    # analysys_year = st.number_input('시뮬레이션 기간을 입력하세요. (단위 : 년)', value=3)
    # st.write(f"* :green[학습된 투자 전략으로 {analysys_year}년의 기간동안 시뮬레이션을 실행]합니다. 기간이 길어질수록 분석시간이 늘어납니다.")
   
    button1, button2, button3 = st.columns(3)
    

    # 버튼 설계
    if button2.button("시뮬레이션"):
        
        success_msg = st.success('분석을 시작합니다. 잠시만 기다려주세요...')

        # 컨테이너 생성
        con = st.container()
        
        
        # 종목 기본정보 불러오기
        all_stock_info_df = get_stock_basic_info(0, market="ALL", detail="BASIC")
        # 티커 추출
        cond = all_stock_info_df['종목명'] == stock_name   
        ticker = all_stock_info_df.loc[cond,'티커'].values[0]
        # 종목 가격 관련 정보 불러오기
        result_df = get_stock_price_info(ticker,"ALL","ALL", params.YEAR_TO_DAY*analysys_year)     
        success_msg.empty()
        success_msg = st.success('금방 끝납니다!  (8초 이내)')
        
        with st.spinner('Wait for it...'):
            # 종료조건, target_cond 보다 높은 backtest_yeild가 발견되거나, iter를 다 돌때 까지
            start = time.perf_counter()
            result = get_backtest_yeild_with_name(stock_name, buy_cond, sell_cond, analysys_year,result_df)
            end = time.perf_counter()
            total_time =  end - start
            backtest_yeild =  result[0]
            yeild_prediction = result[1]
            invest_efficiency = result[2]
            stock_price_info = result[3]
            stock_fundamental_info = result[4]
            sell_cond = result[6]
            buy_cond = result[5]
            buy_list = result[7]
            sell_list = result[8]
            
            success_msg.empty()
            st.success(f'분석 완료! \t Completed : {total_time:.1f}초 ')
            
            
            # 매수 매도 가격과 수익률 출력
            trading_history = pd.DataFrame([buy_list,sell_list]).transpose()
            trading_history.columns = ['매수가격','매도가격']
            trading_history.dropna(inplace=True, axis=0)
            trading_history['수익률'] = round(trading_history['매도가격']/trading_history['매수가격']-1,3)*100 
            trading_history['수익률'] = trading_history['수익률'].apply(lambda x : str(x)[:6]+"%")
         
            if invest_efficiency==1:
                invest_efficiency = "적합"

            elif invest_efficiency==0:
                invest_efficiency = "부적합"


            result_ls = [[backtest_yeild, yeild_prediction, invest_efficiency]]    
            result_df = pd.DataFrame(result_ls)
            result_df.columns =[ f"{analysys_year}년 Back_test 연환산수익률(%)" , "5일 이후 예측수익률(%)", "투자전략 유효성"]
            
        st.markdown("---------")
        yeild_prediction = result_df['5일 이후 예측수익률(%)'].values[0]
        col1,col2 = st.columns(2)
        # 조건부 서식
        col1.header("투자전략 평가")
        if result_df['투자전략 유효성'].values[0]=="적합": 
            col1.header(":green[AI 신뢰도 높음]")
        else:
            col1.header(":red[AI 신뢰도 낮음]")
        col2.header("추천 매매포지션")
        
        if invest_efficiency=="적합":
            if yeild_prediction>buy_cond:
                col2.header(":green[매수]") 
            else:
                col2.header(":yellow[관망]")
        else:
            if yeild_prediction<sell_cond:
                col2.header(":red[매도]") 
            else:
                col2.header(":yellow[관망]")
                
                
        st.markdown("---------")
        st.header(f"AI 매매일지")
        st.text(f" {analysys_year}년 중 매매횟수: {len(trading_history)*2}회**")
        st.dataframe(trading_history,use_container_width=True)
        
        
        col1,col2,col3 = st.columns(3)
        col1.metric("","")
        col2.metric(f"{analysys_year}년 Back_test 연환산수익률(%)",f"{result_df[f'{analysys_year}년 Back_test 연환산수익률(%)'].values[0]:.2f}%")
        col3.metric("","")
        
        st.text("")
        st.markdown(f"* Back_test :green[연환산수익률이 {target_cond}% 이상]일 때 효과적인 매매전략입니다.  \n  (2022.11.24 기준금리)")
        st.markdown("* :red[Back_test 연환산수익률(%)은 현재 모델을 기준으로 계산된 시뮬레이션 결과로로 실제 수익률과 다릅니다.]")     
      
        # 기타 정보제공

        st.markdown("---------")
        st.header("종목 관련 정보")
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
        
        col1, col2, col3= st.columns(3)
        col1.metric("시가", str(o_price), str(o_price-o_price_yesterday))
        col2.metric("고가", str(h_price), str(h_price-h_price_yesterday))
        col3.metric("저가", str(l_price), str(l_price-l_price_yesterday))

        
        st.markdown("---------")

        st.header("종목 펀더멘탈 정보")
        st.markdown("* 지표가 시장에 비해 15% 이상 좋은경우 저평가, 나쁜경우 고평가로 판단합니다.")

        market_fundamental = get_index_fundamental_info(ticker)
        per = stock_fundamental_info.iloc[:,0:1].values[0][0]
        pbr = stock_fundamental_info.iloc[:,1:2].values[0][0]
        div = stock_fundamental_info.iloc[:,2:3].values[0][0]
        
        market_name = market_fundamental['market_name']
        market_per = market_fundamental['PER']
        market_pbr = market_fundamental['PBR']
        market_div = market_fundamental['DIV']
        
        
        # 시장대비 x% 이상인 경우 저평가
        # 시장대비 X% 이하인 경우 저평가
        # -x%~x% 정도 지표가 
        
        
        # per은 낮을수록 저평가 (시장-종목)
        market_stock_diff_per = market_per-per 
        market_stock_diff_rate_per = market_stock_diff_per/market_per
        per_judge = ratio_judge(market_stock_diff_rate_per,0.15)
        
        # pbr은 낮을수록 저평가 (시장-종목)
        market_stock_diff_pbr = market_pbr-pbr
        market_stock_diff_rate_pbr = market_stock_diff_pbr/market_pbr
        pbr_judge = ratio_judge(market_stock_diff_rate_pbr,0.15)
        
        # div는 높을수록 좋음
        market_stock_diff_div = market_div-div
        market_stock_diff_rate_div = market_stock_diff_div/market_div
        div_judge = ratio_judge(market_stock_diff_rate_div, 0.15,"R")

        per_result = f"{market_stock_diff_per:.2f} ({market_name} {per_judge})"
        pbr_result = f"{market_stock_diff_pbr:.2f} ({market_name} {pbr_judge})"
        div_result = f"{market_stock_diff_div:.2f} ({market_name} {div_judge})"
        
    
        col1, col2 = st.columns(2)
        col1.metric("PER(주가/순이익)", f"{per:.2f}", per_result)
        col2.metric("PBR(주가/장부가치)", f"{pbr:.2f}", pbr_result)
        
        col1, col2 = st.columns(2)
        col1.metric("DIV(배당/주가)", f"{div:.2f}", div_result)
        col2.metric("", "", "")
            
        st.markdown("---------")


                
elif side_menu_name=='상승률/하락률 상위종목':

    # 시장을 조회해서 각 업종정보를 가져와서 종합?
    top_bottom = get_high_low_info()
    top10 = top_bottom[0]
    top10_name = top10['종목명']
    top10_close = top10['종가']
    top10_rate = top10['등락률']
    
    bottom10 = top_bottom[1]
    bottom10_name = bottom10['종목명']
    bottom10_close = bottom10['종가']
    bottom10_rate = bottom10['등락률']
    
    with st.container():
        # 종목명 종가 등락률
        col1,col2 = st.columns(2)
        col1.header(f"상승률 상위")
        col2.header(f"하락률 상위")
        for i in range(10):
            # 종목명 종가 등락률 1~10위
            col1,col2 = st.columns(2)
            col1.metric( f" ", f"{top10_name.values[i]}", f"{top10_rate.values[i]:.2f}%  / {top10_close.values[i]}")
            col2.metric( f" ",f"{bottom10_name.values[i]}", f"{bottom10_rate.values[i]:.2f}%  /  {bottom10_close.values[i]}")

    
elif side_menu_name=="종목관련 테마 조회":
    stock_name = st.text_input('종목명을 입력하세요', value="카카오")
    thema = get_included_thema_stocks_in_thema(stock_name)
    thema_list = thema[0]
    thema_stocks_df = thema[1]
        
    st.header("관련 테마")    
    options = st.multiselect(
                            f'{stock_name} 이/가 포함된 테마입니다.',
                            thema_list,
                            thema_list)            
    st.markdown("---------")
    st.header("테마별 관련 종목")
    for thema_name in options:
        cond = thema_stocks_df['테마명']==thema_name
        # 테마명 분리
        result_df = thema_stocks_df[cond].iloc[:,1:]
        # 테마 평균 수익률 계산
        thema_average_rate = 1
        thema_average_rate = result_df['등락률'].mean()
        # 소수점 문제로 인해 str 형태로 임시 변경
        result_df = result_df.astype('str')
        result_df['종가'] = result_df['종가'].apply(lambda x: x.split(".")[0])
        result_df['등락률'] = result_df['등락률'].apply(lambda x: x+"%")    
        
        st.subheader(f"{thema_name}테마 /  평균 등락률 : {thema_average_rate:.2f}%")
        st.dataframe(result_df)
        

elif side_menu_name=='코스피/코스닥 달력':
    st.markdown("개발중입니다.")