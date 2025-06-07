import openai
import time
import os

# OpenAI API 키 설정
openai.api_key = "" 

# 감정별 파일명
emotion_files = [
    '분노_리뷰.txt',
    '기쁨_리뷰.txt',
    '슬픔_리뷰.txt',
    '불안_리뷰.txt',
    '상처_리뷰.txt',
    '당황_리뷰.txt'
]

# 프롬프트 템플릿
base_prompt = """너는 뷰티 브랜드 마케팅 전문가야.
다음은 감정이 '{emotion}'인 고객 리뷰들이야.
리뷰 내용을 바탕으로 마케팅 인사이트 3가지와 이를 활용한 구체적인 솔루션을 제안해줘.
직관적이고 actionable한 결과를 원해. 친절하고 명확하게 작성해줘."""

# 감정별 처리
for file_name in emotion_files:
    try:
        print(f"\n===== {file_name} 분석 중 =====")

        with open(file_name, 'r', encoding='utf-8') as f:
            reviews = f.readlines()

        # 필요한 만큼만 자르기
        selected_reviews = reviews[:10] 

        # 리뷰 합치기
        reviews_text = '\n'.join([f"- {review.strip()}" for review in selected_reviews if review.strip()])
        emotion = file_name.split('_')[0]
        full_prompt = base_prompt.format(emotion=emotion) + "\n\n" + reviews_text

        # ChatGPT API 호출
        response = openai.ChatCompletion.create(
            model="gpt-4o",  
            messages=[
                {"role": "system", "content": "당신은 마케팅 전문가입니다."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )

        result = response['choices'][0]['message']['content']

        output_file = f"{emotion}_인사이트.txt"
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.write(result)

        print(f"===== {output_file} 저장 완료 ✅ =====")
        time.sleep(10)

    except Exception as e:
        print(f"\n===== {file_name} 에서 오류 발생 =====")
        print(e)
        continue



