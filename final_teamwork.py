import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. 데이터 불러오기
df = pd.read_csv("aihub_018_감성대화.csv") 

# 2. 텍스트와 라벨 지정
texts = df['text']    
labels = df['types'] 

# 3. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels)

# 4. TF-IDF 벡터화
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. 로지스틱 회귀 모델 학습
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train_vec, y_train)

# 6. 테스트 예측 및 평가
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# 7. 결과 출력
print("정확도:", accuracy)
print("분류 리포트:\n", report)