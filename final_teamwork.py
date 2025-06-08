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
from collections import Counter
import matplotlib.pyplot as plt
from collections import defaultdict

# 모델과 벡터 불러오기
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# 리뷰 파일 불러오기
with open('리뷰 랜덤 추출.txt', 'r', encoding='cp949') as f:
    reviews = f.readlines()

# 전처리
reviews = [review.strip() for review in reviews if review.strip()]

# 벡터화
review_vectors = vectorizer.transform(reviews)

# 감정 예측
predictions = model.predict(review_vectors)

# 결과 출력
for review, prediction in zip(reviews, predictions):
    print(f"리뷰: {review} ➡ 감정: {prediction}")



# 첫 번째 막대그래프_감정 개수 
emotion_counts = Counter(predictions)
total_reviews = sum(emotion_counts.values())
emotion_percentages = {emotion: (count / total_reviews) * 100 for emotion, count in emotion_counts.items()}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1행 2열 (좌/우)

axes[0].bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
axes[0].set_title('감정별 리뷰 개수')
axes[0].set_xlabel('감정')
axes[0].set_ylabel('리뷰 수')
axes[0].tick_params(axis='x', rotation=45)

# 두 번째 그래프_감정별 비율
axes[1].bar(emotion_percentages.keys(), emotion_percentages.values(), color='lightgreen')
axes[1].set_title('감정별 리뷰 비율 (%)')
axes[1].set_xlabel('감정')
axes[1].set_ylabel('비율 (%)')
axes[1].set_ylim(0, 100)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

for i, (emotion, percentage) in enumerate(emotion_percentages.items()):
    axes[1].text(i, percentage + 1, f'{percentage:.1f}%', ha='center')

plt.show()

# 그래프 저장
fig.savefig('감정_리뷰_개수_비율_한화면.png', dpi=300)



# 감정별로 리뷰 파일로 저장하기
emotion_dict = defaultdict(list)

with open('리뷰 랜덤 추출.txt', 'r', encoding='cp949') as f:
    reviews = f.readlines()

for review in reviews:
    review = review.strip()
    if review:
        vec = vectorizer.transform([review])
        pred = model.predict(vec)[0]
        emotion_dict[pred].append(review)

for emotion, emotion_reviews in emotion_dict.items():
    filename = f"{emotion}_리뷰.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        for r in emotion_reviews:
            f.write(r + '\n')

print("감정별 리뷰 저장 완료")
