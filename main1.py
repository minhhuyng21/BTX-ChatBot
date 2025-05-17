import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI ,OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
import time
import speech_recognition as sr
from dotenv import load_dotenv
# Thay thế các imports cũ
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import create_openai_tools_agent  # Thêm import mới này
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain_core.vectorstores import InMemoryVectorStore
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
with open("docs.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

# Tạo lại list Document
docs = [Document(page_content=item["page_content"],
                 metadata=item["metadata"])
        for item in raw]
CUSTOM_SYSTEM_PROMPT = """[Phiên bản mới] Bạn là hệ thống trợ lý ảo Trường THPT Bùi Thị Xuân. Khi xử lý truy vấn SQL:
{agent_scratchpad}  # Thêm placeholder này vào prompt

Cấu trúc database:
- ThoiKhoaBieu (Thu, Tiet, Phong, MonHoc, TenGV, Lop)
- GiaoVien (MaGV, TenGV, MonDay, LopChuNhiem)
- Lop (MaLop, TenLop, SiSo, GVCN)
- HocSinh(Lop, MaSo, Hoten, NgaySinh)

Quy tắc:
1. Luôn kiểm tra tên bảng/cột trước khi truy vấn
2. Sử dụng toán tử LIKE cho các trường hợp không rõ ràng
3. Nếu kết quả trống, đề xuất các truy vấn thay thế
"""
load_dotenv()
SQL_KEYWORDS = [
    "thời khóa biểu","tkb", "giáo viên chủ nhiệm",
    "lớp", "tiết", "phòng", "tiết học", "môn","thứ","cô", "thầy","ai","tên", "số lượng"
]
OPENAI_API_KEY = os.getenv('OPENAI')
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
@st.cache_resource
def load_model():
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    llm = ChatOpenAI(model="gpt-4o")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = InMemoryVectorStore(embedding_model)
    prompt = hub.pull("rlm/rag-prompt")
    db = FAISS.load_local("faiss", embedding_model, allow_dangerous_deserialization=True)
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    _ = vector_store.add_documents(documents=docs)
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    # Define application steps
    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}


    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        return {"answer": response.content}


    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    # SQL Agent
    sql = SQLDatabase.from_uri("sqlite:///sql_new.db")
    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    sql_agent = create_sql_agent(
        llm=llm,
        db=sql,
        prompt=sql_prompt,
        agent_type="openai-tools",
        verbose=True,
        handle_parsing_errors=True,
    )

    return graph, sql_agent

# In your main function:
def response_generator(user_input):
    # First try RAG chain
    try:
        # response = st.session_state['rag_chain'].invoke(user_input + " và trả lời bằng tiếng việt")
        response = st.session_state['rag_chain'].invoke({"question": user_input + " và trả lời bằng tiếng việt"})
        return response
    except Exception as e:
        # Fallback to SQL agent
        return st.session_state['sql_agent'].invoke({"input": user_input})

def is_sql_query(user_input: str) -> bool:
    text = user_input.lower()
    return any(kw in text for kw in SQL_KEYWORDS)

def response_generator(user_input):
    # 1. Kiểm tra intent
    if is_sql_query(user_input):
        # 2a. Dùng SQL agent
        try:
            answer = st.session_state['sql'].invoke(user_input.upper())['output']
        except:
            answer = "Có lỗi truy vấn dữ liệu, vui lòng thử lại."
    else:
        # 2b. Dùng LLM
        try:
            response = st.session_state['model'].invoke({"question": user_input + " và trả lời bằng tiếng việt"})
            answer = response['answer']
        except:
            answer = "Có lỗi xử lý ngôn ngữ, vui lòng thử lại."

    # 3. Stream từng từ
    for w in answer.split():
        yield w + " "
        time.sleep(0.05)

def recognize_speech():
    # Tạo container chính để chứa toàn bộ nội dung
    main_container = st.empty()
    
    with main_container.container():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Đang thu âm... Hãy nói vào microphone.")
            audio = recognizer.listen(source)
        
        # Tạo container cho thông báo trạng thái
        status_container = st.empty()
        
        try:
            # Nhận dạng giọng nói
            text = recognizer.recognize_google(audio, language="vi-VN")
            status_container.success("Nhận dạng thành công!")
            time.sleep(5)  # Đợi 5 giây
            
            # Xóa toàn bộ nội dung
            main_container.empty()
            
            return text
            
        except sr.UnknownValueError:
            status_container.error("Không thể nhận dạng giọng nói.")
        except sr.RequestError:
            status_container.error("Lỗi kết nối với dịch vụ nhận dạng.")
        
        # Xóa toàn bộ nội dung sau 5 giây nếu có lỗi
        time.sleep(5)
        main_container.empty()
        return ""

def main():
    st.title("Trợ lí ảo Bùi Thị Xuân")
    voice_button = st.button("VOICE")
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Nhập promt")
    if voice_button:
        user_input = recognize_speech()
    if user_input:      
        # Lưu cuộc hội thoại người dùng
        st.session_state['messages'].append({"role": "user", "content": user_input})

        # Hiển thị cuộc hội thoại người dùng
        with st.chat_message("user"):
            st.markdown(user_input)

        # Hiển thị cuộc hội thoại AI
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(user_input))

        # Lưu cuộc hội thoại AI  
        st.session_state.messages.append({"role": "assistant", "content": response})    

if __name__ == '__main__':
    if 'model' not in st.session_state and 'sql' not in st.session_state:
        st.session_state['model'], st.session_state['sql'] = load_model()


    main()
