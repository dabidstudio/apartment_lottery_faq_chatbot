#2024-10-16
#여러 pdf파일 업로드를 허용하는 경우

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
import fitz  # PyMuPDF
import re

from dotenv import load_dotenv, dotenv_values
load_dotenv()

############################### 1단계 : PDF 문서를 벡터DB에 저장하는 함수들 ##########################

## 1: 임시폴더에 파일 저장
def save_uploadedfiles(uploadedfiles: List[UploadedFile]) -> List[str]: 
    temp_dir = "PDF_임시폴더"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_paths = []
    for uploadedfile in uploadedfiles:
        file_path = os.path.join(temp_dir, uploadedfile.name)
        with open(file_path, "wb") as f:
            f.write(uploadedfile.read()) 
        file_paths.append(file_path)
    return file_paths

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

def process_pdfs(file_paths: List[str]) -> None:
    all_documents = []
    for pdf_path in file_paths:
        documents = pdf_to_documents(pdf_path)
        smaller_documents = chunk_documents(documents)
        all_documents.extend(smaller_documents)
    save_to_vector_store(all_documents)

############################### 2단계 : RAG 기능 구현과 관련된 함수들 ##########################

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
def convert_pdfs_to_images(pdf_paths: List[str], dpi: int = 250) -> dict:
    images_dict = {}
    for pdf_path in pdf_paths:
        doc = fitz.open(pdf_path)  # 문서 열기
        image_paths = []
        
        # 이미지 저장용 폴더 생성 per PDF
        file_base = os.path.splitext(os.path.basename(pdf_path))[0]
        output_folder = os.path.join("PDF_이미지", file_base)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for page_num in range(len(doc)):  #  각 페이지를 순회
            page = doc.load_page(page_num)  # 페이지 로드

            zoom = dpi / 72  # 72이 디폴트 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat) # type: ignore

            image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")  # 페이지 이미지 저장 page_1.png, page_2.png, etc.
            pix.save(image_path)  # PNG 형태로 저장
            image_paths.append(image_path)  # 경로를 저장
            
        images_dict[file_base] = image_paths
    return images_dict

def display_pdf_page(image_path: str, page_number: int) -> None:
    image_bytes = open(image_path, "rb").read()  # 파일에서 이미지 인식
    st.image(image_bytes, caption=f"Page {page_number}", use_column_width=True)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def main():
    st.set_page_config("청약 FAQ 챗봇", layout="wide")

    # Initialize session state
    if 'images_dict' not in st.session_state:
        st.session_state.images_dict = {}
    if 'selected_file' not in st.session_state:
        st.session_state.selected_file = None
    if 'page_number' not in st.session_state:
        st.session_state.page_number = None

    left_column, right_column = st.columns([1, 1])
    with left_column:
        st.header("청약 FAQ 챗봇")

        pdf_docs = st.file_uploader("PDF Uploader", type="pdf", accept_multiple_files=True)
        button = st.button("PDF 업로드하기")
        if pdf_docs and button:
            with st.spinner("PDF문서 저장중"):
                pdf_paths = save_uploadedfiles(pdf_docs)
                process_pdfs(pdf_paths)
            # (3단계) PDF를 이미지로 변환해서 세션 상태로 임시 저장
            with st.spinner("PDF 페이지를 이미지로 변환중"):
                images_dict = convert_pdfs_to_images(pdf_paths)
                st.session_state.images_dict = images_dict

        user_question = st.text_input("PDF 문서에 대해서 질문해 주세요",
                                      placeholder="무순위 청약 시에도 부부 중복신청이 가능한가요?")

        if user_question:
            response, context = process_question(user_question)
            st.write(response)
            i = 1
            for document in context:
                with st.expander("관련 문서"):
                    st.write(document.page_content)
                    file_path = document.metadata.get('source', '')
                    # Extract the file base name from the file path
                    file_base = os.path.splitext(os.path.basename(file_path))[0]
                    page_number = document.metadata.get('page', 0) + 1
                    button_key = f"link_{file_base}_{page_number}_{i}"
                    reference_button = st.button(f"🔎 {file_base} pg.{page_number}", key=button_key)
                    i = i + 1
                    if reference_button:
                        st.session_state.selected_file = file_base
                        st.session_state.page_number = page_number

    with right_column:
        selected_file = st.session_state.get('selected_file')
        page_number = st.session_state.get('page_number')
        if selected_file and page_number:
            images_dict = st.session_state.get('images_dict', {})
            image_paths = images_dict.get(selected_file, [])
            # Ensure the page number is within range
            if 1 <= page_number <= len(image_paths):
                image_path = image_paths[page_number - 1]
                display_pdf_page(image_path, page_number)
            else:
                st.error("페이지 번호가 범위를 벗어났습니다.")

if __name__ == "__main__":
    main()
