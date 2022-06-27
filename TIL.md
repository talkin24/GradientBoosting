# Part1. Bagging & Boosting

## Ch1. Gradient Boosting in ML

XGBoost는 표 형식 데이터를 활용한 예측에서 가장 뛰어난 머신러닝 알고리즘이다.

부스팅의 일반적인 아이디어: 약한 학습기(weak learner)를 반복적으로 오차를 개선하여 강한 학습기로 바꾸는 것

Gradient Boosting의 핵심 아이디어: 경사 하강법을 사용해 잔여 오차를 최소화 하는 것

`df_bikes.isna().any(axis=1)`: 열을 따라 누락된 값이 하나 이상인 모든 행을 찾음

`df_bikes.iloc[[56, 81]]`: 두개 이상의 인덱스를 찾을 때는 대괄호 두번 사용

`df_bikes.groupby('season')['hum'].transform('median')`: transform()메서드는 첫번째 매개 변수로 전달된 함수를 적용한 다음 원본과 동일한 길이의 시리즈나 데이터프레임을 반환함

`df_bikes['dteday'].apply(pd.to_datetime, infer_datetime_format=True, errors='coerce')`: infer_datetime_format=True로 지정하면 판다스가 datetime 객체의 종류를 결정하며, 대부분의 경우 안전함

`df_bikes.loc[730, 'yr'] = 1.0`loc 메서드를 사용하여 행과 열을 지정하여 특정 값을 변경할 수 있음

`fillna` 메서드에 원본과 동일한 길이의 시리즈 객체를 전달하면 누락된 위치에 있는 값만 채우는데 사용함