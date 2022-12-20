import streamlit as st
import pandas as pd
from predict import get_backtest_yeild_with_name, get_high_low_info, ratio_judge
from loader import date_from_now, get_stock_basic_info, get_stock_price_info, get_index_fundamental_info, get_included_thema_stocks_in_thema
import random
from itertools import combinations
import params
import numpy as np


st.sidebar.header('MENU')
side_menu_name = st.sidebar.selectbox('사용할 기능을 선택하세요.',['주식 수익률 예측','종목관련 테마 조회','상승률/하락률 상위종목'])


st.title('지금 투자해도 될까?')
st.markdown('----')

if side_menu_name=='주식 수익률 예측':
    st.header("Step 1 : 종목명 입력")
    stock_name = st.text_input('종목명을 입력하세요', value="카카오")
    st.markdown('----')
    st.header("Step 2 : 목표수익률 입력")
    target_cond = st.number_input('목표 연간 수익률을 입력하세요. (단위 : %)', value=3.25)
    st.markdown(f"* Back_test :green[연환산수익률이 {target_cond}% 이상 되도록]하는 매수/매도 조건을 찾습니다.")
    st.markdown(f"* 매매전략에서의 요구수익률을 의미합니다. 기준금리, 5년물 국채 , 또는 다른 지표들이 기준이 될수 있습니다.")
    st.markdown('----')
    st.header("Step 3 : 시뮬레이션 기간 입력")
    analysys_year = st.number_input('시뮬레이션 기간을 입력하세요. (단위 : 년)', value=3)
    st.write(f"* :green[학습된 투자 전략으로 {analysys_year}년의 기간동안 시뮬레이션을 실행]합니다. 기간이 길어질수록 분석시간이 늘어납니다.")
   
    # ========================================
    #매수 예측 최소범위
    buy_min =  1
    #매수 예측 최대범위
    buy_max = 10
    #매도 예측 최소범위
    sell_min = -1
    #매도 예측 최대범위
    sell_max = -10
    #model mae 값으로 변경하며 추정
    model_mae = 3
    # ========================================
    

    # 버튼 설계
    if st.button("시뮬레이션"):
        st.success('분석을 시작합니다.')
        # 로딩바 초기화
        my_bar = st.progress(0)

        # 컨테이너 생성
        con = st.container()
        st.markdown("---------")
        
        # 종목 기본정보 불러오기
        all_stock_info_df = get_stock_basic_info(0, market="ALL", detail="BASIC")
        # 티커 추출
        cond = all_stock_info_df['종목명'] == stock_name   
        ticker = all_stock_info_df.loc[cond,'티커'].values[0]
        # 종목 가격 관련 정보 불러오기
        result_df = get_stock_price_info(ticker,"ALL","ALL", params.YEAR_TO_DAY*analysys_year)
        
        # 목표 수익률 
        # 반복 횟수 (최대 매수예측 조건 10% 최저 매도조건 예측 -1%)
        # 매수는 덜민감하게 반응하고, 매도는 민감하게 반응하여 수익률을 올림

        # 매수매도조건 생성기
        combinations_ls = combinations(list(range(sell_max,sell_min+1,model_mae))+list(range(buy_min,buy_max+1,model_mae)),2)
        combinations_for_use = []
        for comb in combinations_ls:
            if (comb[0]<=0) and (comb[1]>=0):
                combinations_for_use.append(comb)
        #최적 조합 초기화
        best_result = -np.inf
        best_idx = np.inf
        backtest_yeild = -np.inf
        
        # 인덱스 
        idx = 0
        len_idx = len(combinations_for_use)
            
        # 로딩바 기초변수 설정        
        percent_complete = 0
        percent_tick = 1/(len_idx+1)
         

        # 만약 목표로 하는 수익률 이하로 계속 backtest 수익률이 나온다면 반복하되,
        with st.spinner('Wait for it...'):
            # 종료조건, target_cond 보다 높은 backtest_yeild가 발견되거나, iter를 다 돌때 까지
            while (backtest_yeild <= target_cond):
                percent_complete += percent_tick
                my_bar.progress(percent_complete)
                if percent_complete>=1:
                    percent_complete = 1
                  
                # 종료조건, 모든 조합이 다 돈 경우 후보중 최적 조합을 뽑는다.
                if  (idx==len_idx):
                    percent_complete = 1
                    break
                    
                # 수익률 조합 꺼내기
                comb = combinations_for_use[idx]
                sell_cond = comb[0]
                buy_cond = comb[1]
                result = get_backtest_yeild_with_name(stock_name, buy_cond, sell_cond, analysys_year,result_df)
                backtest_yeild = result[0]
                # 최고 수익률 달성시 최고결과 갱신
                if backtest_yeild >= best_result:
                    best_result = backtest_yeild
                    best_idx = idx
                idx +=1
            percent_complete = 1
                
            comb = combinations_for_use[best_idx]
        result = get_backtest_yeild_with_name(stock_name, comb[1], comb[0], analysys_year, result_df)                    
        backtest_yeild =  result[0]
        yeild_prediction = result[1]
        recommend_position = result[2]
        invest_efficiency = result[3]
        stock_price_info = result[4]
        stock_fundamental_info = result[5]
        sell_cond = result[7]
        buy_cond = result[6]
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
        # 응답 성공 메세지
        st.success('분석 완료!')
        
        
        st.header("투자전략 총평")
        col1,col2 = st.columns(2)
        col1.metric("투자전략 유효성",f"{result_df['투자전략 유효성'].values[0]}")
        col2.metric(f"{analysys_year}년 Back_test 연환산수익률(%)",f"{result_df[f'{analysys_year}년 Back_test 연환산수익률(%)'].values[0]:.2f}%")
        col2.write(">:red[Back_test 연환산수익률(%)은 현재 모델을 기준으로 매매 했을 때, 얻은 수익률로 실제 수익률과 다릅니다.]")
        st.write(">일반적으로 연환산수익률이 요구수익률(ex 기준금리)보다 높다면, 효과적인 전략으로 판단합니다.")

        st.markdown("---------")

        st.write('* :green[투자전략이 부적합한 경우에는 기간을 변경해보세요]')
        st.write('* :red[[주 의] 투자전략이 부적합한 경우에는 예측수익률을 신뢰하지 마십시오!!!]')
        st.markdown("---------")
        
        st.header("예측수익률 및 추천 매매포지션")
        col1,col2 = st.columns(2)
        col1.metric(f"5일 이후 예측수익률(%)",f"{result_df['5일 이후 예측수익률(%)'].values[0]:.2f}%")
        col2.metric("추천 매매포지션",f"{result_df['추천 매매포지션'].values[0]}")        
        st.write(">시뮬레이션에 사용된 모델을 활용한 예측결과입니다.  투자전략이 부적합하다면 신뢰할 수 없습니다.")
        
        
        st.markdown("---------")
        if invest_alert=='부적합':
            st.write(">현재 종목은 예측모델을 활용한 매매전략을 제시하기 어렵습니다!")
            st.write('>:red[투자전략이 부적합한 경우에는 최적 매매기준점을 제공하지 않습니다.]')
        else:
            st.header("매매 타이밍 진단")
            col1,col2 = st.columns(2)
            col1.metric("매수 기준 예측수익률(%)",f"{buy_cond}%",f"+ 예측값이 {buy_cond}% 이상일 때, 매수 시기입니다.")
            col2.metric("매도 기준 예측수익률(%)",f"{sell_cond}%",f"- 예측값이 {sell_cond}% 이하일 때, 매도 시기입니다.")



        # 기타 정보제공
        with st.spinner('Wait for it...'):
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
            with st.spinner('Wait for it...'):
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
                div_judge = ratio_judge(market_stock_diff_rate_pbr,0.15,"R")

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
        col1,col2= st.columns(2)
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