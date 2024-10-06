
## streamlit 관련 모듈 불러오기
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

## 환경변수 불러오기
from dotenv import load_dotenv
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
def convert_pdf_to_images(pdf_path: str, dpi: int = 400) -> List[bytes]:
    doc = fitz.open(pdf_path)  # open document
    images = []
    for page in doc:
        # Increase the resolution by setting a higher dpi
        zoom = dpi / 72  # 72 is the default dpi
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat) # type: ignore
        img_bytes = pix.tobytes("png")  # get image bytes in PNG format
        images.append(img_bytes)
    return images
def display_pdf_page(image_bytes: bytes, page_number: int, total_pages: int) -> None:
    st.image(image_bytes, caption=f"Page {page_number}", output_format="PNG", width=600)





def main():
    st.set_page_config("청약 FAQ 챗봇", layout="wide")

    # 세션상태 초기화
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = []
    if "page_number" not in st.session_state:
        st.session_state.page_number = 1
    if "images" not in st.session_state:
        st.session_state.images = []
    if "response" not in st.session_state:
        st.session_state.response = None
    if "context" not in st.session_state:
        st.session_state.context = []

    # 칼럼 2개 생성
    left_column, right_column = st.columns([1, 1])

    with left_column:
        st.header("청약 FAQ 챗봇")
        # 사용자가 파일을 업로드할 수 있는 컴포넌트
        pdf_doc = st.file_uploader("PDF Uploader", type="pdf")

        if pdf_doc and st.button("PDF 업로드하기"):
            with st.spinner("PDF문서 저장중"):
                # Convert the UploadedFile object to a file path
                pdf_path = save_uploadedfile(pdf_doc)
                pdf_document = pdf_to_documents(pdf_path)  # Process the single PDF
                smaller_documents = chunk_documents(pdf_document)
            with st.spinner("벡터DB 생성중"):
                save_to_vector_store(smaller_documents)
            st.session_state.uploaded_file = pdf_doc

            # (3단계) PDF를 이미지로 변환해서 세션 상태로 임시 저장
            with st.spinner("PDF 페이지를 이미지로 변환중"):
                images = convert_pdf_to_images(pdf_path)
                st.session_state.images.append(images)

        # 사용자 질문
        user_question = st.text_input("PDF 문서에 대해서 질문해 주세요",
                                    placeholder="무순위 청약 시에도 부부 중복신청이 가능한가요?")

        if user_question:
            response, context = process_question(user_question)
            st.session_state.response = response
            st.session_state.context = context

        if st.session_state.response:
            st.write(st.session_state.response)
            for idx, doc in enumerate(st.session_state.context):
                with st.expander("관련 문서"):
                    st.write(doc.page_content)
                    file_path = doc.metadata.get('source', '')
                    page_number = doc.metadata.get('page', 0) + 1
                    if file_path and page_number:
                        button_key = f"link_{file_path}_{page_number}_{idx}"  # Add idx to make the key unique
                        if st.button(f"🔎 {os.path.basename(file_path)} pg.{page_number}", key=button_key):
                            st.session_state.page_number = str(page_number)
                            print(st.session_state.page_number)

                            st.rerun()

    with right_column:
        # Safely get query parameters
        page_number = st.session_state.get('page_number')

        if page_number:
            try:
                page_number = int(page_number)
                images = st.session_state.images
                total_pages = len(images)
                display_pdf_page(images[page_number - 1], page_number, total_pages)


                # Add pagination buttons in a single row
                prev_col, _, next_col = st.columns([1, 5, 1])
                with prev_col:
                    if page_number > 1:
                        if st.button("이전"):
                            st.session_state.page_number = page_number - 1
                            st.rerun()
                with next_col:
                    if page_number < total_pages:
                        if st.button("다음"):
                            st.session_state.page_number = page_number + 1
                            st.rerun()      
            except (ValueError, IndexError, FileNotFoundError) as e:
                st.error(f"Error displaying PDF page: {str(e)}")

if __name__ == "__main__":
    main()