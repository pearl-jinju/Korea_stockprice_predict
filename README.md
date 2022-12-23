# Korea_stockprice_predict
## 프로젝트 목표
  > 미래의 주가는 과거의 주가의 행태(behavior)에 의해 결정된다는 기술적분석(technical analysis)은 주식 시장을 비롯한 금융 시장을 분석하고 예측하는 중요한 기법 가운데 하나입니다.
  > 이 프로젝트의 목표는 시장에서 주로 사용되는 기술적분석(차트, 지지선 등)을 x일간의 주가의 패턴을 vector화하여 분석하고자했으며, **기술적분석의 효과성을 확인하기 위함입니다**
   > * 기술적 분석은 다양한 지표를 사용하지만, 대부분의 지표가 주가로부터 계산됩니다. 이에 모델학습시에 다중공선성 문제가 지속적으로 발생했고, 이 프로젝트에서는 오직 주가에만 초점을 맞췄습니다.
   > * 현재 시점(t)을 기준으로 과거 X일 전까지의 주가 패턴을 LightGBM을 train_data로, 현재 시점(t)으로부터 Z일 후 보유수익률을 test로 사용하였습니다. 
   
## Dataset의 정의
![image](https://user-images.githubusercontent.com/97038372/207221677-b3a95f85-3d4f-4aee-8e8e-f78ee41888f6.png)
![image](https://user-images.githubusercontent.com/97038372/207221712-ffb92880-44d5-40ed-b6bd-7289fc39546c.png)

## 발견한 점

![image](https://user-images.githubusercontent.com/97038372/208244219-47fe7971-b2a9-4ee6-84a7-19e0f4c5d2ca.png)

> 20221217
> 50일의 이전 주가를 x값으로 5일 이후의 수익률을 y값으로 두었을 때, 수익률의 분포이다.
> 놀랍게도 이론에서 가정하는 것처럼 정규분포의 형태를 보였다.

## Usage
TODO

## 실행 예시
### 1. 매매타이밍 추천 프로그램
#### 1)종목명/모델 선택/손절 민감도 선택
![image](https://user-images.githubusercontent.com/97038372/209274458-644ae4a6-464a-4905-8fce-d015c08acc32.png)

#### 2) 투자 전략평가 /  추천매매포지션
> - AI가 추정한 투자전략이 유효한지 확인합니다.
> - 현재 위 종목에 대한 매매포지션을 추천합니다.

![image](https://user-images.githubusercontent.com/97038372/209274609-f1552060-8f93-4783-a99f-5d379c449c5c.png)

#### 3) 연환산수익률 확인
![image](https://user-images.githubusercontent.com/97038372/209274698-784a0517-c734-429e-85b7-c553124b350d.png)

#### 4) 내일의 수익률 예측
> - 학습된 모델이 추정한 내일의 수익률입니다. regression 모델이 사용된 추정입니다.
![image](https://user-images.githubusercontent.com/97038372/209274722-7ed912da-8801-4d30-aa09-64ad7ac71f8f.png)

#### 5) 추정 오차확인
> - 학습된 모델이 추정한 수익률과 실제 수익률간 오차를 차트로 표현합니다.
![image](https://user-images.githubusercontent.com/97038372/209274806-7225f940-7c0f-463d-a45d-84fe689a8bda.png)

#### 6) 종목 관련 정보
> - 당일 주가의 변동을 나타냅니다.
![image](https://user-images.githubusercontent.com/97038372/209274873-26d35acf-d817-430a-b89b-b79ed7de4d24.png)

#### 7) 종목 펀더멘탈 정보
> - 당일 기준 종목의 펀더멘탈을 벤치마크(해당 종목이 편입된 벤치마크 시장 참고)와 비교하여 상대적 고평가/보통/저평가를 구분합니다.
![image](https://user-images.githubusercontent.com/97038372/209274910-cf9ee73e-5310-4b28-8aac-bebed312138c.png)


### 2. 종목관련 테마 조회
#### 1) 관련테마
> - 종목과 관련된 테마를 출력합니다.
![image](https://user-images.githubusercontent.com/97038372/209275147-a33f3151-42a8-4002-81bc-a1cc72a2ddbe.png)
> - 아래처럼 일부의 테마만 선택할 수 있습니다.
![image](https://user-images.githubusercontent.com/97038372/209275264-69787bd7-efdd-4830-a57b-c7e8bbbce99a.png)
 
#### 2) 테마 관련 다른 종목/ 테마 평균 수익률/ 테마사유
> - 테마 내에 속한 다른 종목들을 확인해 볼 수 있습니다. 테마의 평균 수익률 및 테마로 선정된 사유까지 확인할 수 있습니다.
![image](https://user-images.githubusercontent.com/97038372/209275465-e5d9d696-5cc2-4bed-9a57-c40ecc91b148.png)


### 3. 상승률/하락률 상위종목
> -  상승률/하락률 TOP10 종목을 출력합니다
![image](https://user-images.githubusercontent.com/97038372/209275616-d2d669a2-7f0c-4ee0-89d0-e815094020a3.png)


## TODO_LIST
1) 웹페이지 제작
> * streamlit 기반 웹페이지 설계
> * 일별 투자 적기 종목리스트 정보 제공 함수 (웹 서비스 항목) (적용완료 20221221)
> * 최고 backtest 수익률 종목리스트 제공 함수 (웹 서비스 항목) (적용완료 20221221)
> * 종목검색 기능 및 그 종목의 투자 적기판단 제공 함수 (웹 서비스 항목) (적용완료 20221221)
> * 검색기록 DB연동해서 인기검색어 표출
----
2) 예측모델 보강
> * 일별 자동 데이터 수집 및 DB 최신화
> * 일별 자동 학습 및 최적 파라미터 도출
> * 년별, 중대한 사건별 chunk 분리 및 개별 학습 적용  + 종목의 지표별(수익지표-BPS,EPS 등), 시장별(KOSPI, KOSDAQ) chunk분리 및 개별 학습 적용
> * 종목별 연환산 수익률 대비 1년국채금리를 벤치마크지표로서 비교하여 초과수익률을 계산하고, 이를 투자 시기로 활용하도록 할것
