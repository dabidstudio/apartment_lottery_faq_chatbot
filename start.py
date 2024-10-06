
## streamlit 관련 모듈 불러오기
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain_community.vectorstores import FAISS #벡터 DB
from langchain_openai import ChatOpenAI, OpenAIEmbeddings #AI모델, 임베딩
from langchain_community.document_loaders import PyMuPDFLoader # PDF Parser
from langchain.text_splitter import RecursiveCharacterTextSplitter # 청크 쪼개기 
from langchain.prompts import PromptTemplate # 프롬프트 템플릿 
from langchain.schema.output_parser import StrOutputParser # 
from langchain_core.documents.base import Document # Document (Type)
from langchain_core.runnables import Runnable # Runnable (Type)

from typing import List
import os
import fitz  # PyMuPDF

## 환경변수 불러오기
from dotenv import load_dotenv, dotenv_values
load_dotenv()



############################### 1단계 : PDF 문서를 벡터DB에 저장하는 함수들 ##########################

## 1: 임시폴더에 파일 저장
def save_uploadedfile(uploadedfile: UploadedFile) -> str : 
    temp_dir = "PDF_임시폴더"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.read()) 
    return file_path

## 2: 저장된 PDF 파일을 Document로 변환
def pdf_to_documents(pdf_path:str) -> List[Document]:
    documents = []
    loader = PyMuPDFLoader(pdf_path)
    doc = loader.load()
    for d in doc:
        d.metadata['file_path'] = pdf_path
    documents.extend(doc)
    return documents

## 3: Document를 더 작은 document로 변환
def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(documents)

## 4: Document를 벡터DB로 저장
def save_to_vector_store(documents: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")



############################### 2단계 : PDF 문서를 벡터DB에 저장하는 함수들 ##########################


## 사용자 질문에 대한 RAG 처리
@st.cache_data
def process_question(user_question):


    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    ## 벡터 DB 호출
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    ## 관련 문서 3개를 호출하는 Retriever 생성
    retriever = new_db.as_retriever(search_kwargs={"k": 3})
    ## 사용자 질문을 기반으로 관련문서 3개 검색 
    retrieve_docs : List[Document] = retriever.invoke(user_question)

    ## RAG 체인 선언
    chain = get_rag_chain()
    ## 질문과 문맥을 넣어서 체인 결과 호출
    response = chain.invoke({"question": user_question, "context": retrieve_docs})

    return response, retrieve_docs



def get_rag_chain() -> Runnable:
    template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘
    - 질문에 대한 응답을 해줘
    - 간결하게 5줄 이내로 해줘
    - 곧바로 응답결과를 말해줘

    컨텍스트 : {context}

    질문: {question}

    응답:"""

    custom_rag_prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o-mini")

    return custom_rag_prompt | model | StrOutputParser()



############################### 3단계 : 응답결과와 문서를 함께 보도록 도와주는 함수 ##########################


@st.cache_data(show_spinner=False)
## dpi 조정으로 해상도 조정 가능
def convert_pdf_to_images(pdf_path: str, dpi: int = 400) -> List[bytes]:
    doc = fitz.open(pdf_path)  # open document
    images = []
    for page in doc:
        zoom = dpi / 72  # 디폴트 dpi : 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat) # type: ignore
        img_bytes = pix.tobytes("png")  # PNG 형태로 이미지
        images.append(img_bytes)
    return images

def display_pdf_page(image_bytes: bytes, page_number: int, total_pages: int) -> None:
    st.image(image_bytes, caption=f"Page {page_number}", output_format="PNG", width=600)


############################### 메인함수 ##########################


def main():
    print(dotenv_values(".env"))
    print("셋팅완료")


if __name__ == "__main__":
    main()