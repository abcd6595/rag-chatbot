import streamlit as st
import os

# 🔑 API 키 (로컬 테스트용 / 배포 시 아래 줄 수정)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 📦 LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# 📌 제목
st.title("📄 논문 기반 RAG 챗봇")
st.write("논문 내용을 기반으로 질문에 답변합니다.")

# 📄 데이터 로딩 (캐싱)
@st.cache_resource
def load_data():
    loader = PyPDFLoader("paper.pdf")
    docs = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(texts, embeddings)
    return db

db = load_data()

# 🔍 Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# 🤖 LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# 🧠 Prompt
prompt = ChatPromptTemplate.from_template("""
다음 논문 내용을 기반으로 질문에 답하세요.
핵심 내용을 중심으로 간결하게 설명하세요.

{context}

질문: {question}
""")

# 🔗 RAG 체인
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 💬 사용자 입력
query = st.text_input("질문을 입력하세요:")

# 💡 예시 질문
st.write("### 💡 예시 질문")
st.write("- 이 논문의 연구 목적은 무엇인가?")
st.write("- 주요 결과는 무엇인가?")
st.write("- 시사점은 무엇인가?")

# 🧾 답변 출력
if query:
    with st.spinner("답변 생성 중..."):
        response = rag_chain.invoke(query)
        st.write(response.content)
