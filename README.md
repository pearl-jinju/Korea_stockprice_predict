# Korea_stockprice_predict
## 프로젝트 목표
  > 미래의 주가는 과거의 주가의 행태(behavior)에 의해 결정된다는 기술적분석(technical analysis)은 주식 시장을 비롯한 금융 시장을 분석하고 예측하는 중요한 기법 가운데 하나입니다.
  > 이 프로젝트의 목표는 시장에서 주로 사용되는 기술적분석(차트, 지지선 등)을 x일간의 주가의 패턴을 vector화하여 분석하고자했으며, **기술적분석의 효과성을 확인하기 위함입니다**
   > * 기술적 분석은 다양한 지표를 사용하지만, 대부분의 지표가 주가로부터 계산됩니다. 이에 모델학습시에 다중공선성 문제가 지속적으로 발생했고, 이 프로젝트에서는 오직 주가에만 초점을 맞췄습니다.
   > * 현재 시점(t)을 기준으로 과거 X일 전까지의 주가 패턴을 LightGBM을 train_data로, 현재 시점(t)으로부터 y일 후 보유수익률을 test로 사용하였습니다. 
   

## Usage
TODO

## 결과 화면 예시
TODO



## TODO_LIST
> *  일별 자동 데이터 수집 및 DB 최신화
> *  일별 자동 학습 및 최적 파라미터 도출
> *  년별, 중대한 사건별 chunk 분리 및 개별 학습 적용  + 종목의 지표별(수익지표-BPS,EPS 등), 시장별(KOSPI, KOSDAQ) chunk분리 및 개별 학습 적용
> * flask 기반 웹페이지 설계
