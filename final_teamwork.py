#1. 리뷰 텍스트 불러오기기
import pandas as pd
with open("random_review.txt", "r", encoding = "cp949") as f:
    lines = f.readlines()

#줄 번호 제거 및 양쪽 공백 제거
reviews = []
for line in lines:
    parts = line.strip().split("\t", 1)
    if len(parts) == 2:
        review_text = parts[1]
        reviews.append(review_text)

df = pd.DataFrame(reviews, columns = ["review"])
print(df.head())

#2. 텍스트 전처리
import re

def clean_text(text):
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", "", text) #특수 문자 제거하기
    text = re.sub(r"\s+", " ", text) #공백 정리
    return text.strip()

df["cleaned"] = df["review"].apply(clean_text)

#3. 임베딩
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["cleaned"])

#4. 감정 분석 (transformer사용)
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="beomi/KcELECTRA-base") #지피티의 도움을 받음

def get_sentiment(text):
    try:
        result = classifier(text[:512])[0]
        return "긍정" if result["label"] == "LABEL_1" else "부정"
    except Exception as e:
        return "오류"
    
df["sentiment"] = df["cleaned"].apply(get_sentiment)

print(df[["review", "sentiment"]].head(10))
print("전체 리뷰 개수:", len(df))
print("긍정 리뷰 개수:", len(df[df["sentiment"] == "긍정"]))
print("부정 리뷰 개수:", len(df[df["sentiment"]=="부정"]))
print("오류 리뷰 개수:", len(df[df["sentiment"] == "오류"]))

#5. 감정 분석 결과 저장
df.to_csv("감정분석_결과.csv", index=False, encoding="utf-8-sig")


