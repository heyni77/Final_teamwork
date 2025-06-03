#1. 리뷰 텍스트 불러오기기
import pandas as pd
with open("random_review.txt", "r", encoding = "cp949") as f:
    lines = f.readlines()

#줄 번호 제거 및 양쪽 공백 제거
reviews = [line.strip().split("\t",1)[-1] for line in lines if "\t" in line]

df = pd.DataFrame(reviews, columns = ["review"])
print(df.head())

#2. 텍스트 전처리
import re

def clean_text(text):
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text) #공백 정리
    return text.strip()

df["cleaned"] = df["review"].apply(clean_text)

#3. 임베딩
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["cleaned"])

#4. 감정 분석 모델

