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

사이킷런의 `cross_val_score` 함수의 `scoring='neg_mean_squared_error'` 라고 쓰는 이유: 사이킷런은 점수가 높은것을 좋은 것으로 판단하기 때문

`df.info()` 는 메모리 사용량도 확인시켜줌



## Ch2. Decision Tree

gini impurity는 낮을수록 분류가 잘 된 것. 0.5인 경우 무작위 추측보다 더 낫다고 볼 수 없음

딱 한번만 분할된 트리는 stump라고 부른다. 이 스텀프도 부스터로 사용되면 강력해질 수 있음

훈련된 reg 객체의 tree_ 속성에 훈련된 트리 객체가 저장되어 있음

​	children_left, children_right 속성은 자식 노드의 인덱스를 담고 있음. 따라서 이들이 -1이면 리프노드라는 의미

GridSearchCV의 핵심은 매개변수 값의 딕셔너리를 만드는 것

`max_features` 옵션

- 0~1: 전체 feature의 비율
- 정수: 사용할 feature 개수
- None, auto: 전체 feature
- sort: 전체 feature의 제곱근
- log2: 전체 feature의 로그(밑이2)

`splitter` 옵션을 'random'으로 하면 과대적합을 막고 다양한 트리를 만들 수 있음. 'best' 옵션은 정보 이득이 가장 큰 특성만을 선택함

회귀와 분류의 criterion은 다름

- 회귀: squared_error, friedman_mse, absolute_error, poisson
- 분류: gini, entropy

`RandomizedSearchCV` 는 모든 조합을 테스트하는 대신 랜덤한 조합을 테스트함. 제한된 시간 내에 최적의 조합을 찾게 해줌. 또한 분포를 이용하여 연속적 매개변수 샘플링 가능. randint, uniform, loguniform 등

모델을 결정한 후에는 테스트셋을 포함하여 모델을 훈련하는것이 정확도를 더 높일 수 있음. 단 모델을 실전에 투여했을 때 얻을 수 있는 성능을 추정하기 어려워짐

`operator.itemgetter` 는 sorted 같은 함수의 key 매개변수에 적용하여 다양한 기준으로 정렬할 수 있도록 하는 모듈

사이킷런에서 추천하는 특성 중요도 측정 방법은 `permutation_importance()`함수임. 이 함수는 기존 모델에 특성 하나를 랜덤하게 섞은 후 모델을 훈련하여 성능을 비교



## Ch3. Bagging & Randomforest

RF는 XGB와 마찬가지로 결정트리의 앙상블. 차이점은 RF는 Bagging을 통해 트리를 연결, XGB는 Boosting을 통해 트리를 연결

앙상블 방법은 크게 2가지

1. VotingClassifier처럼 사용자가 선택한 여러 종류의 머신러닝 모델을 연결하는 방식
2. XGBoost나 RF처럼 같은 종류의 모델을 여러개 합치는 앙상블

Bagging = Bootstrap Aggregation

Bootstraping: 중복을 허용한 샘플링

원본데이터 -> 부트스트래핑(샘플링) -> 애그리게이팅

중복허용한 샘플링을 통해 원래 가방에 있는 것보다 더 많은 샘플링이 가능함!

RF는 원본 데이터셋과 같은 크기의 부트스트래핑 샘플을 사용해 각 트리를 만듦. 수학적으로 평균적으로 각 트리의 샘플은 전체 샘플의 2/3 정도만 사용함.

분류일 경우, 다수결. 회귀일 경우 평균.

RF의 `n_estimator` 인자가 트리 개수

기본적으로 RF Classifier는 노드를 분할할 때 특성 개수의 제곱근을 사용함 => 중복 샘플을 가진 두 트리의 분할이 매우 달라져 매우 다른 예측을 만들게 됨. 분산을 줄이는 포인트

RF회귀모델은 부트스트랩샘플을 이용하지만 노드 분할에 특성의 제곱근이 아니라 feature를 전부 사용함

`oob_score` : True이면 각 트리에서 훈련시 사용되지 않은 샘플을 사용해 개별 트리의 예측 점수를 누적하여 평균을 냄

트리의 개수가 많아야 더 많은 oob sample이 나오게 됨. 당연히 각 개별 트리는 각각 다른 oob 샘플을 갖게됨

`warm_start`: 해당 매개변수는 트리 개수를 결정하는데 도움이 됨. True 시 이전 모델에 이어서 트리를 추가하게 됨

seaborn의 set()메서드는 set_theme()의 별칭. style 매개변수의 기본값이 dark grid 임

RF는 일반적으로 부트스트래핑을 사용하지만 boostrap 매개변수를 False로 지정할 수도 있음. 이 경우 oob_score_ 속성이 만들어지지 않음

verbose 매개변수 사용 시 유용한 정보를 얻을 수 있음

RF 구조 자체가 분산을 줄이도록 고안되었기 때문에 결정트리 매개변수가 RF에서 아주 중요하지는 않음

교차검증함수는 훈련된 모델을 반환하지 않기 때문에 oob_score_ 속성을 사용할 수 없음

CV 중 특정 셋만 값이 튄다면 shuffle을 통해 해결 가능할 수 있음

RF의 단점: 개별 트리에 제약이 된다. 모든 트리가 동일한 실수를 저지르면 RF도 실수를 저지름. 개별 트리가 해결할 수 없는 데이터 내의 문제 때문에 RF의 성능이 향상될 수 없었음

부스팅은 트리가 저지른 실수에서 배우도록 설계됨



## Ch4. From GB to XGB

부스팅의 기본 아이디어: 이전 트리의 오차를 기반으로 새로운 트리를 훈련. 즉 개별 트리가 이전 트리를 기반으로 만들어짐

GB의 새로운 트리는 올바르게 예측된 값에는 영향을 받지 않음. 따라서 잔차를 활용.

GB 모델 만들기

1. 결정트리 훈련. (기본학습기)
2. 훈련 세트에 대한 예측 수행
3. 잔차 계산(잔차는 다음 트리의 타깃이 됨)
4. 새로운 트리를 잔차에 대해 훈련
5. 2~4를 반복. 앙상블에 추가할 트리 개수만큼 반복이 계속됨
6. 각 트리의 테스트셋 예측 결과를 더함

learning rate는 개별 트리의 기여를 결정함. 일반적으로 트리 개수를 늘리면 lr을 줄여야 함. 따라서 최적의 lr은 트리 개수에 따라 다름

GBRegressor의 기본학습기는 DT임. 이 기본학습기의 매개변수도 조정할 수 있음