


import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    TextLoader,
    UnstructuredXMLLoader,
)
import mimetypes
from typing import List
import os
import fitz 
import re


from dotenv import load_dotenv,dotenv_values
load_dotenv()


def file_to_documents(file_path: str) -> List[Document]:
    ext = file_path.split('.')[-1].lower()

    if ext == 'pdf':
        loader = PyMuPDFLoader(file_path)
    elif ext in ['ppt', 'pptx']:
        loader = UnstructuredPowerPointLoader(file_path)
    elif ext in ['xls', 'xlsx']:
        loader = UnstructuredExcelLoader(file_path)
    elif ext == 'csv':
        loader = CSVLoader(file_path)
    elif ext == 'txt':
        loader = TextLoader(file_path, encoding='utf-8')
    elif ext == 'xml':
        loader = UnstructuredXMLLoader(file_path)
    else:
        raise ValueError(f"지원하지 않는 파일 타입: {ext}")

    documents = loader.load()
    for d in documents:
        d.metadata['file_path'] = file_path

    return documents



# 문서를 벡터DB에 저장
# 임시폴더에 파일 저장
def save_uploadedfile(uploadedfile: UploadedFile) -> str : 
    temp_dir = "pdf_temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.read()) 
    return file_path

# 파일을 Document로 변환
def pdf_to_documents(pdf_path:str) -> List[Document]:
    documents = []
    loader = PyMuPDFLoader(pdf_path)
    doc = loader.load()
    for d in doc:
        d.metadata['file_path'] = pdf_path
    documents.extend(doc)
    return documents

# chunking
def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100) # 800자 기준으로 쪼개면서 전후로 100자정도는 겹치도록함
    return text_splitter.split_documents(documents)


# 문서를 벡터DB(FAISS)로 저장
def save_to_vector_store(documents: List[Document]) -> None:
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")




# RAG 기능 구현
# RAG 처리
@st.cache_data
def process_question(user_question):

    #embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # DB 호출
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # 관련 문서 k개를 호출하는 Retriever 생성
    retriever = new_db.as_retriever(search_kwargs={"k": 3})
    # 사용자 질문을 기반으로 관련문서 k개 검색 
    retrieve_docs : List[Document] = retriever.invoke(user_question)
    # RAG 체인 선언
    chain = get_rag_chain()

    # 체인 결과 호출
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
    #model = ChatOpenAI(model="gpt-4o-mini")
    model = Ollama(model="mistral")

    return custom_rag_prompt | model | StrOutputParser()



# 응답결과와 문서를 함께 보도록하는 메서드
@st.cache_data(show_spinner=False)
def convert_pdf_to_images(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path)
    image_paths = []
    
    # 이미지 저장용 폴더 생성
    output_folder = "pdf_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_num in range(len(doc)):  #  각 페이지를 순회
        page = doc.load_page(page_num)  # 페이지 로드

        zoom = dpi / 72  # 72가 디폴트 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat) # type: ignore

        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")  # 페이지 이미지 저장 page_1.png, page_2.png, etc.
        pix.save(image_path)  # PNG 형태로 저장
        image_paths.append(image_path)  # 경로를 저장
        
    return image_paths

def display_pdf_page(image_path: str, page_number: int) -> None:
    image_bytes = open(image_path, "rb").read()  # 파일에서 이미지 인식
    st.image(image_bytes, caption=f"Page {page_number}", output_format="PNG", width=600)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def main():

    st.set_page_config("주택 FAQ 챗봇",layout="wide")
    left_column, right_column = st.columns([1,1])
    with left_column:
        st.header("주택 FAQ 챗봇")

        uploaded_doc = st.file_uploader("문서 업로드", type=['pdf', 'ppt', 'pptx', 'xls', 'xlsx', 'csv', 'txt', 'xml'])
        button = st.button("문서 업로드")

        if uploaded_doc and button:
            with st.spinner("문서 저장중"):
                file_path = save_uploadedfile(uploaded_doc)
                document = pdf_to_documents(file_path)
                smaller_documents = chunk_documents(document)
                save_to_vector_store(smaller_documents)

            if file_path.endswith(".pdf"):
                with st.spinner("PDF를 이미지로 변환중") :
                    images = convert_pdf_to_images(file_path)
                    st.session_state.images = images

        user_question = st.text_input("주택업무편람에 대해서 질문해주세요.",
                                    placeholder="신혼부부의 디딤돌 대출 조건은?")
        
        if user_question :
            response, context = process_question(user_question)
            st.text(response)
            # st.text(context)
            for document in context:
                with st.expander("관련 문서"):
                    st.text(document.page_content)
                    file_path = document.metadata.get('source','')
                    page_number = document.metadata.get('page',0)+1
                    button_key = f"link_{file_path}_{page_number}"
                    reference_button = st.button(f"🔎 {os.path.basename(file_path)} pg.{page_number}", key=button_key)
                    if reference_button :
                        st.session_state.page_number = str(page_number)

    with right_column :
        page_number = st.session_state.get('page_number')  
        # st.text(page_number)                  
        if page_number :
            page_number = int(page_number)
            image_folder = "pdf_images"
            images = sorted(os.listdir(image_folder), key=natural_sort_key)
            #print(images)
            image_paths = [os.path.join(image_folder, image) for image in images]
            #print(page_number)
            #print(image_paths[page_number -1])

            #display_pdf_page(image_paths[page_number - 1], page_number)
            if page_number and len(image_paths) >= page_number:
                display_pdf_page(image_paths[page_number - 1], page_number)
            else:
                st.warning("해당 페이지의 이미지를 찾을 수 없습니다.")


if __name__ == "__main__":
    main()
