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

#8. 학습된 모델과 벡터 저장 (지피티의 도움을 받아 진행)
import joblib

# 모델 저장
joblib.dump(model, 'model.pkl')

# 벡터라이저 저장
joblib.dump(vectorizer, 'vectorizer.pkl')

import joblib

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# 테스트용 문장 넣어보기
test_sentence = ["오늘 날씨가 정말 좋아요"]
test_vector = vectorizer.transform(test_sentence)
prediction = model.predict(test_vector)

print("예측 결과:", prediction)

#리뷰 넣어 돌리기
import joblib

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

with open('리뷰 랜덤 추출.txt', 'r', encoding='cp949') as f:
    reviews = f.readlines()

reviews = [review.strip() for review in reviews if review.strip()]
review_vectors = vectorizer.transform(reviews)
predictions = model.predict(review_vectors)

for review, prediction in zip(reviews, predictions):
    print(f"리뷰: {review} ➡ 감정: {prediction}")

from collections import defaultdict

emotion_dict = defaultdict(list)

with open('리뷰 랜덤 추출.txt', 'r', encoding='cp949') as f:
    reviews = f.readlines()

for review in reviews:
    review = review.strip()
    if review:
        vec = vectorizer.transform([review])
        pred = model.predict(vec)[0]
        emotion_dict[pred].append(review)

# 감정별로 파일로 저장
for emotion, emotion_reviews in emotion_dict.items():
    filename = f"{emotion}_리뷰.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        for r in emotion_reviews:
            f.write(r + '\n')

print("감정별 리뷰 저장 완료 ")
