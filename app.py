import streamlit as st
import pandas as pd
from predict import get_backtest_yeild_with_name, get_high_low_info, ratio_judge
import params
from loader import get_stock_basic_info, get_stock_price_info, get_included_thema_stocks_in_thema ,get_stocks_fundamental_info
from sklearn.metrics import mean_absolute_error
import numpy as np
import time
from streamlit.components.v1 import html

# ========================================
# 매수 판단 예측수익률
buy_cond = 6
# 매도 판단 예측수익률
sell_cond = -2
# 목표 수익률
target_cond = 3.25
# 시뮬레이션 기간
analysys_year = 2
# 초기 모델값
model_radio = '조금 복잡한 모델 (기본 모델)'

# 초기 매수 민감도
buy_sensitivity = buy_cond
# 초기 손절 민감도
sell_sensitivity = sell_cond
# ========================================



st.sidebar.header('MENU')
side_menu_name = st.sidebar.selectbox('사용할 기능을 선택하세요.',['매매타이밍 추천 프로그램','종목관련 테마 조회','상승률/하락률 상위종목'])


st.title('지금 투자해도 될까?')
st.markdown('----')
        
html1 = html("""
        <ins class="kakao_ad_area" style="display:none;"
        data-ad-unit = "DAN-PDZakg9aJMvjy92N"
        data-ad-width = "320"
        data-ad-height = "100"></ins>
        <script type="text/javascript" src="//t1.daumcdn.net/kas/static/ba.min.js" async></script>   
    """)

if side_menu_name=='매매타이밍 추천 프로그램':
    st.header("종목명을 입력하세요!")
    stock_name = st.text_input('  ', value="카카오")
    st.markdown("----")
    
    detail1 = st.checkbox('상세설정')
    if detail1:
        # 라디오 버튼 추가(모델 선택)
        st.subheader("모델 선택")
        model_radio = st.radio(
        "모델을 선택하세요",
        ('조금 복잡한 모델 (기본 모델)','빠른 분석' ,'심층 분석'),)   
        
        st.markdown("----")
        st.subheader("매수 타이밍 선택")
        # 슬라이더 추가(손절 민감도)
        buy_sensitivity = st.slider('AI가 매수할 타이밍을 선택해주세요. 단위(%)', 1, 12, 6)
        st.markdown("----")
        st.info(f"AI 추정값이 {buy_sensitivity}% 이상 예상되는 경우 매수하도록 설정합니다. +6% 추천")
        buy_cond = buy_sensitivity
    
        st.markdown("----")
        st.subheader("손절 타이밍 선택")
        # 슬라이더 추가(손절 민감도)
        sell_sensitivity = st.slider('AI가 매도할 타이밍을 선택해주세요. 단위(%)', -12, -1, -3)
        st.markdown("----")
        st.info(f"AI 추정값이 {sell_sensitivity}% 이상 예상되는 경우 매도하도록 설정합니다. -3% 추천")
        sell_cond = sell_sensitivity

    col1, col2 = st.columns(2)
    
    #매수 민감도 인터페이스
    if buy_sensitivity <= 3:
        info_text = col1.error('매수를 자주 실행합니다. 현금 보유일수가 짧아집니다.')
    elif buy_sensitivity <= 6:
        info_text = col1.success("일반적인 매수를 실행합니다.")
    elif buy_sensitivity <= 9:
        info_text = col1.warning("매수를 가끔 실행합니다. 현금 보유일수가 길어집니다.")
    elif buy_sensitivity <= 12:
        info_text = col1.error("매수를 거의 실행하지 않습니다. 현금 보유일수가 매우 길어집니다.")

        
    #매도 민감도 인터페이스
    if sell_sensitivity >= 0:
        info_text = col2.error('매도를 자주 실행합니다. 보유주기가 짧아집니다.')
    elif sell_sensitivity >= -4:
        info_text = col2.success("일반적인 매도를 실행합니다.")
    elif sell_sensitivity >= -8:
        info_text = col2.warning("매도를 가끔 실행합니다. 보유주기가 길어집니다.")
    elif sell_sensitivity >= -12:
        info_text = col2.error("매도를 거의 실행하지 않습니다. 보유주기가 매우 길어집니다.")
        
    # 모델 인터페이스
    if model_radio == '빠른 분석':
        model_radio = 'naive'
        analysys_year = 1
        st.success('간단한 추정모델입니다.')
    elif model_radio == '조금 복잡한 모델 (기본 모델)':
        model_radio = 'deep'
        analysys_year = 3
        st.success('조금 복잡한 모델입니다. 시간이 조금 더 걸립니다. 30초 이내')
    elif model_radio == '심층 분석':
        analysys_year = 10
        model_radio = 'robust'
        st.error('매우 복잡한 모델입니다. 시간이 오래 걸립니다. 1분 이내')
    
    # st.header("Step 2 : 목표수익률 입력")
    # target_cond = st.number_input('목표 연간 수익률을 입력하세요. (단위 : %)', value=3.25)
    # st.markdown(f"* 현재 기준금리, 5년물 국채 , 또는 다른 지표들이 기준이 될수 있습니다.")
    # st.markdown('----')
    # st.header("Step 3 : 시뮬레이션 기간 입력")
    # analysys_year = st.number_input('시뮬레이션 기간을 입력하세요. (단위 : 년)', value=3)
    # st.write(f"* :green[학습된 투자 전략으로 {analysys_year}년의 기간동안 시뮬레이션을 실행]합니다. 기간이 길어질수록 분석시간이 늘어납니다.")
   

    html2 = html("""
                <ins class="kakao_ad_area" style="display:none;"
                data-ad-unit = "DAN-Nk8wkCPUr6a9ITSc"
                data-ad-width = "300"
                data-ad-height = "250"></ins>
                <script type="text/javascript" src="//t1.daumcdn.net/kas/static/ba.min.js" async></script>
                """)

    button1, button2, button3 = st.columns(3)
    # 버튼 설계    
    if button2.button("매매 타이밍 찾기!"):
        st.markdown('----')
        progress = 0
        my_bar = st.progress(0.0+progress)
        start = time.perf_counter()
        success_msg = st.success('분석을 시작합니다. 잠시만 기다려주세요...')

        # 컨테이너 생성
        con = st.container()
        
        # 종목 기본정보 불러오기
        all_stock_info_df = get_stock_basic_info(0, market="ALL", detail="BASIC")
        # 로딩바 진행
        my_bar.progress(0.15)
        
        # 티커 추출
        cond = all_stock_info_df['종목명'] == stock_name   
        ticker = all_stock_info_df.loc[cond,'티커'].values[0]
        my_bar.progress(0.25)
        # 종목 가격 관련 정보 불러오기
        price_df = get_stock_price_info(ticker,"ALL","ALL", params.YEAR_TO_DAY*analysys_year)     
        my_bar.progress(0.3)
        success_msg.empty()
        # 로딩바 진행
        my_bar.progress(0.4)
        success_msg = st.success('잠시만 기다려주세요')
        
        with st.spinner('Wait for it...'):
            # 종료조건, target_cond 보다 높은 backtest_yeild가 발견되거나, iter를 다 돌때 까지
            my_bar.progress(0.45)
            result = get_backtest_yeild_with_name(stock_name, buy_cond, sell_cond, analysys_year,price_df,model_radio)
            my_bar.progress(0.65)
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
            price_pattern = result[9]
            pridict_pattern = result[10]
            holding_day_ls = result[11]
            trading_position = result[12]
            now_price = result[13]
            
            my_bar.progress(0.85)
           
            success_msg.empty()
            st.success(f' Completed : {total_time:.1f}sec ')
            my_bar.empty()
            
            my_bar.progress(1)
            # 현재 포지션 수익률 초기화
            curr_yeild =0
        # 보유중이라면
        if trading_position =="Y":
            # 매수 매도 가격과 수익률 출력
            trading_history = pd.DataFrame([buy_list,sell_list,holding_day_ls]).transpose()
            trading_history.columns = ['매수가격','매도가격','보유일자(일)']
            trading_history.dropna(inplace=True, axis=0)
            last_buy_price = trading_history['매수가격'].values[-1]
            curr_yeild = round(((now_price-last_buy_price)/last_buy_price)*100,2)
        # 미보유중이라면
        elif trading_position =="N":
            trading_history = pd.DataFrame([buy_list,sell_list,holding_day_ls]).transpose()
            trading_history.columns = ['매수가격','매도가격','보유일자(일)']
            trading_history.dropna(inplace=True, axis=0)
            curr_yeild = 0

        trading_history['수익률'] = round(trading_history['매도가격']/trading_history['매수가격']-1,5)*100 
        # 이익횟수와 손실횟수
        profit_cnt = len(trading_history[trading_history['수익률']>=0])
        loss_cnt = len(trading_history[trading_history['수익률']<0])
        # 적중률 0오류 처리
        if profit_cnt==0:
            correct_rate =0
        else:
            correct_rate = round((profit_cnt/(profit_cnt+loss_cnt))*100,2)
        # 소수점 문제로 인해 str 형태로 임시 변경
        trading_history['매수가격'] = trading_history['매수가격'].apply(lambda x: round(int(x),0))
        trading_history['수익률'] = trading_history['수익률'].apply(lambda x: round(float(x),2))
        trading_history['수익률'] = trading_history['수익률'].apply(lambda x: str(x)[:-1]+"%")
        
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
        col1.header("현재 AI 포지션")
        
        if trading_position =="Y":
            col2.header(":green[보유 중]")
            col2.header(f"현재 수익률 :{curr_yeild:.2f}%")
        elif  trading_position =="N":
            col2.header(":red[관망 중]")
        st.line_chart(price_df['종가'],use_container_width=True)
        
        st.markdown("---------")
        
        
        
        col1,col2 = st.columns(2)
        # 조건부 서식
        col1.header("투자전략 평가")
        if (result_df['투자전략 유효성'].values[0]=="적합") and(correct_rate>60): 
            col1.subheader(":green[AI 신뢰도 높음]")
            st.info('  AI 신뢰도가 높습니다!     매매타이밍추천을 참고하세요!', icon="ℹ️")
        else:
            col1.subheader(":red[AI 신뢰도 낮음]")
            st.warning('  AI 신뢰도가 낮습니다!    매매타이밍추천을 무시하세요!', icon="ℹ️")
        col2.header("추천 매매포지션")
        
        if invest_efficiency=="적합":
            if yeild_prediction>buy_cond:
                col2.subheader("매수") 
            else:
                col2.subheader("관망")
        else:
            if yeild_prediction<sell_cond:
                col2.subheader(":red[매도]") 
            else:
                col2.subheader("관망")
                
                
        st.markdown("---------")

        st.header(f"AI 매매일지")
        col1, col2 = st.columns(2)
        col1.info(f" {analysys_year}년 중 손익실현: {len(trading_history)}회")
        col1.success(f" {analysys_year}년 중 이익 횟수: {profit_cnt}회")
        col2.info(f" 평균 보유일 수: {np.mean(holding_day_ls):.1f}일")
        col2.error(f" {analysys_year}년 중 손실 횟수: {loss_cnt}회")
        
        st.info(f"적중률 : {correct_rate:.2f}%")
       
        
        st.dataframe(trading_history,use_container_width=True)
        
        
        st.markdown("---------")
        st.header(f" {analysys_year}년 Back_test 연환산수익률(%)")
        col1,col2,col3 = st.columns(3)
        
        col1.header("","")
        col2.header(f"{result_df[f'{analysys_year}년 Back_test 연환산수익률(%)'].values[0]:.2f}%")
        col3.header("","")
        
        st.text("")
        st.markdown(f"* Back_test :green[연환산수익률이 {target_cond}% 이상]일 때 효과적인 매매전략입니다.  \n  (2022.11.24 기준금리)")
        st.markdown("* :red[Back_test 연환산수익률(%)은 현재 모델을 기준으로 계산된 시뮬레이션 결과로로 실제 수익률과 다릅니다.]")     
        
        
        st.markdown("---------")
        con2 = st.container()
        con2.header(f" 내일 수익률 예측(%)")
        col1,col2,col3 = con2.columns(3)
        
        col1.header("","")
        col2.header(f"{result_df[f'5일 이후 예측수익률(%)'].values[0]:.2f}%")
        col3.header("","")
        
        con2.error("""
                   주가 예측은 현실적으로 어려운 일이며, 이론적으로는 추정할 수 없습니다.\n
                   제공하는 예측수익률은 학습된 모델을 기준으로 추정된 결과일 뿐입니다. \n
                   예측값은 오차가 항상 있습니다. 이점 유의하여 주시고, 투자의 근거로 사용하지 않길 바랍니다.\n
                   모든 투자에 대한 책임은 투자자 본인에게 있습니다.
                   """)
        st.markdown("---------")
        
        
        
        real_chart_data = pd.DataFrame(
                np.array(price_pattern[-5:]),
                columns=['수익률']
                )
        predict_chart_data=pd.DataFrame(
                np.array(pridict_pattern[-6:-1]),
                columns=['예측수익률']
                )
              
        chart_data = pd.concat([real_chart_data,predict_chart_data],axis=1)
        mae_5 = mean_absolute_error(price_pattern[-5:],pridict_pattern[-6:-1])
        st.header(f"최근 5일 등락률 패턴  :green[오차 ± {mae_5:.2f}%]")
        st.line_chart(chart_data,use_container_width=True)  
        
        
        real_chart_data = pd.DataFrame(
                np.array(price_pattern[-100:]),
                columns=['수익률']
                )
        predict_chart_data=pd.DataFrame(
                np.array(pridict_pattern[-101:-1]),
                columns=['예측수익률']
                )
        chart_data = pd.concat([real_chart_data,predict_chart_data],axis=1)
        mae_100 = mean_absolute_error(price_pattern[-100:],pridict_pattern[-101:-1])
        st.header(f"최근 100일 등락률 패턴  :green[오차 ± {mae_100:.2f}%]")
        st.line_chart(chart_data,use_container_width=True)     
        

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



        market_fundamental = get_stocks_fundamental_info(ticker)
        per = stock_fundamental_info.iloc[:,0:1].values[0][0]
        pbr = stock_fundamental_info.iloc[:,1:2].values[0][0]
        div = stock_fundamental_info.iloc[:,2:3].values[0][0]
        
        market_name = market_fundamental['market_name']
        market_per = market_fundamental['PER']
        market_pbr = market_fundamental['PBR']
        market_div = market_fundamental['DIV']
        
        st.header(f"종목 펀더멘탈 정보")
        st.subheader(f"벤치마크 : {market_name}")
        st.markdown("* 지표가 시장에 비해 15% 이상 좋은경우 저평가, 나쁜경우 고평가로 판단합니다.")
        
        
        # 시장대비 x% 이상인 경우 저평가
        # 시장대비 X% 이하인 경우 저평가
        # -x%~x% 정도 지표가 
        
        
        # per은 낮을수록 저평가 (시장-종목)
        market_stock_diff_per = per - market_per 
        market_stock_diff_rate_per = market_stock_diff_per/market_per
        per_judge = ratio_judge(market_stock_diff_rate_per,0.15)
        
        # pbr은 낮을수록 저평가 (시장-종목)
        market_stock_diff_pbr = pbr - market_pbr
        market_stock_diff_rate_pbr = market_stock_diff_pbr/market_pbr
        pbr_judge = ratio_judge(market_stock_diff_rate_pbr,0.15)
        
        # div는 높을수록 좋음
        market_stock_diff_div = div - market_div
        market_stock_diff_rate_div = market_stock_diff_div/market_div
        div_judge = ratio_judge(market_stock_diff_rate_div, 0.15,"R")

        per_result = f"펀더멘탈 평가 : {per_judge}"
        pbr_result = f"펀더멘탈 평가 : {pbr_judge}"
        div_result = f"펀더멘탈 평가 : {div_judge}"
        
        
            
        
        col1, col2 = st.columns(2)
        if per_judge =="고평가":
            col1.header("PER")
            col1.info(f"종목 PER {per:.2f}   /   시장 PER {market_per:.2f}  /   {market_stock_diff_rate_per*100:.2f}%")
            col1.error(per_result)
        elif per_judge=="보통":
            col1.header("PER")
            col1.info(f"종목 PER {per:.2f}   /   시장 PER {market_per:.2f}  /   {market_stock_diff_rate_per*100:.2f}%")
            col1.info(per_result)
        elif per_judge=="저평가":
            col1.header("PER")
            col1.info(f"종목 PER {per:.2f}   /   시장 PER {market_per:.2f}  /   {market_stock_diff_rate_per*100:.2f}%")
            col1.success(per_result)
            
        if pbr_judge =="고평가":
            col2.header("PBR")
            col2.info(f"종목 PBR {pbr:.2f}   /   시장 PBR {market_pbr:.2f}  /   {market_stock_diff_rate_pbr*100:.2f}%")
            col2.error(pbr_result)
        elif pbr_judge=="보통":
            col2.header("PBR")
            col2.info(f"종목 PBR {pbr:.2f}   /   시장 PBR {market_pbr:.2f}  /   {market_stock_diff_rate_pbr*100:.2f}%")
            col2.info(pbr_result)
        elif pbr_judge=="저평가":
            col2.header("PBR")
            col2.info(f"종목 PBR {pbr:.2f}   /   시장 PBR {market_pbr:.2f}  /   {market_stock_diff_rate_pbr*100:.2f}%")
            col2.success(pbr_result)

        col1, col2 = st.columns(2)
        
        if div_judge =="고평가":
            col1.header("DIV")
            col1.info(f"종목 DIV {div:.2f}   /   시장 DIV {market_div:.2f}  /   {market_stock_diff_rate_div*100:.2f}%")
            col1.error(div_result)
        elif div_judge=="보통":
            col1.header("DIV")
            col1.info(f"종목 DIV {div:.2f}   /   시장 DIV {market_div:.2f}  /   {market_stock_diff_rate_div*100:.2f}%")
            col1.info(div_result)
        elif div_judge=="저평가":
            col1.header("DIV")
            col1.info(f"종목 DIV {div:.2f}   /   시장 DIV {market_div:.2f}  /   {market_stock_diff_rate_div*100:.2f}%")
            col1.success(div_result)
        
        # 펀더멘탈 총평 작성    
        result_ls = [per_result, pbr_result, div_result]
        high = [i for i in result_ls if "고평가" in i]
        normal = [i for i in result_ls if "보통" in i]
        low = [i for i in result_ls if "저평가" in i]
        result_fundamental = len(high)  + len(normal)*2  +len(low)*3
        
        if result_fundamental>=7:
            result_fundamental = "저평가"
        elif result_fundamental>= 5:
            result_fundamental = "보통"
        else:
            result_fundamental = "고평가"
            
        if result_fundamental =="고평가":
            col2.header("총평")
            col2.header(f":red[{result_fundamental}]")
        elif result_fundamental=="보통":
            col2.header("총평")
            col2.header(f"{result_fundamental}")

        elif result_fundamental=="저평가":
            col2.header("총평")
            col2.header(f":green[{result_fundamental}]")
        
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
            col1,col2,col3,col4 = st.columns(4)
            col1.metric( f" ", f"{top10_name.values[i]}", f"{top10_rate.values[i]:.2f}%  / {top10_close.values[i]}")
            col3.metric( f" ",f"{bottom10_name.values[i]}", f"{bottom10_rate.values[i]:.2f}%  /  {bottom10_close.values[i]}")

    
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
        
        st.subheader(f"{thema_name}테마 / 평균 등락률 : {thema_average_rate:.2f}%")
        st.dataframe(result_df,use_container_width=True)
        
    
        

elif side_menu_name=='코스피/코스닥 달력':
    st.markdown("개발중입니다.")