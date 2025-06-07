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

print("감정별 리뷰 저장 완료 ✅")
