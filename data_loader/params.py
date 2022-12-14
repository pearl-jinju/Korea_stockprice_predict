# 벡터화할 기간(분석기간)
ANALYSIS_DAY =45
# 수익률 산정기간(test값 산출 기간)
PERIOD_YEILD_DAY=5
# 1년으로 볼 영업일 수 
YEAR_TO_DAY = 252
# 이동평균일수
MOVING_AVERAGE_DAY=244
# 신규상장 감안 분석제거 년수
NEW_LISTING_REDUCTION_YEAR = 1

#사용할 DATASET(분석기간, 수익률기간,1년기준치, 평균가격, 신규상장효과 제거년수)
DATA_SET = "../data/Korea_stock_Dataset_45_5_252_224_1.pkl"

# 매수/매도를 하기위한 최소한의 수익률
TRADING_HURDLE = [0.3,-0.3]
# TRADING_HURDLE = [500,500]