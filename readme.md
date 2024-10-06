
# 파이썬으로 RAG 웹서비스 만들기 : 청약 FAQ 챗봇

### 가이드 영상 링크 : TBD
### 청약 FAQ 문서 링크 : [링크바로가기](https://www.molit.go.kr/USR/policyData/m_34681/dtl.jsp?search=&srch_dept_nm=&srch_dept_id=&srch_usr_nm=&srch_usr_titl=Y&srch_usr_ctnt=&search_regdate_s=&search_regdate_e=&psize=10&s_category=&p_category=&lcmspage=1&id=4765)

## 활용 기술
![image](https://github.com/user-attachments/assets/1fd16d1a-cf58-4922-b5db-be521110d0b0)

## 예시 화면

![image](https://github.com/user-attachments/assets/a6a2f9fe-029e-4808-a153-712d528bad09)



## 사전 준비사항

### 파이썬 가상 환경 설정  
Python 가상환경 설정 가이드 : https://github.com/dabidstudio/dabidstudio_guides/blob/main/python-set-venv.md

### OpenAI API Key 발급
OpenAI API Key 발급 방법 : https://github.com/dabidstudio/dabidstudio_guides/blob/main/get-openai-api-key.md


## 시작 방법  

1. 다음 명령어로 필요한 패키지 설치:
    ```bash
    pip install -r requirements.txt
    ```
2. `.env` 파일을 생성하고 OpenAI API Key 입력:
    ```
    OPENAI_API_KEY=your-openai-api-key-here
    ```
3. 다음 명령어로 웹 서비스 실행:
    ```bash
    streamlit run start.py
    ```

## 코드 설명  
- **시작 코드**: `start.py` 
- **완성 코드**: `completed.py` 

## 예시 질문
1. 30세대 미만의 주택을 공급하려는 경우에도 전매제한을 적용받는지
2. 주택청약종합저축은 증여 또는 명의변경이 가능한가요?
3. 무순위 청약 시에도 부부 중복신청이 가능한가요?
4. 혼인신고일이 2018. 9. 1.인 경우 언제까지 신혼부부 주택 특별공급의 신청이  가능한가요? 

## 기타
- FAISS 뷰어 : https://faissviewer-hu2g6bbyxgcdjjumbsfysz.streamlit.app
