import joblib

# model.pkl 불러오기
model = joblib.load('model.pkl')

# vectorizer.pkl 불러오기
vectorizer = joblib.load('vectorizer.pkl')

# 테스트용 문장 넣어보기
test_sentence = ["오늘 날씨가 정말 좋아요"]

# 벡터화
test_vector = vectorizer.transform(test_sentence)

# 예측하기
prediction = model.predict(test_vector)

print("예측 결과:", prediction)


import joblib

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


from collections import defaultdict

# 감정별로 리뷰를 담을 딕셔너리
emotion_dict = defaultdict(list)

# 리뷰 파일 다시 불러오기
with open('리뷰 랜덤 추출.txt', 'r', encoding='cp949') as f:
    reviews = f.readlines()

# 각 리뷰에 대해 감정 예측
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
