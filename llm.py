import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

## 환경변수 읽어오기 
load_dotenv()

# llm함수 정의===========================================================
def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm 

# database 함수 정의 ====================================================
def get_database():
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    Pinecone(api_key=PINECONE_API_KEY)
    index_name = 'retrieval02'

    ## 저장된 인덱스 가져오기
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )

    return database

# Statefully manage chat history =========================================
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


# RetrievalQA 함수 정의 =========================================================
def get_retrievalQA() -> RunnableWithMessageHistory:
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    #벡터스토어에서 index정보 가져옴
    database = get_database()


    ### Answer question ###   
    #프롬프트 설정
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 전세사기 피해 법률 전문가입니다. [context]를 참고하여 사용자의 질문에 답변하세요. 답변에 참조한 해당 조항을 '(xx법 제x조 제x호)'형식으로 문단 마지막 줄에 표시하세요. 전세사기피해 법률 이외 질문에는 '전세사기 피해와 관련된 질문만 해주세요.'로 답변하세요. [context]:{context}" ),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    ) 

    #llm 모델 지정
    llm = get_llm()
 
    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)

    from langchain_core.runnables import RunnableLambda

    input_str = RunnableLambda(lambda x: x['input'])

    qa_chain = ( 
                {
                'context': input_str | database.as_retriever() | format_docs, 
                'input': input_str,
                'chat_history': RunnableLambda(lambda x: x['chat_history'])
                } 
                | qa_prompt 
                | llm 
                | StrOutputParser())

    conversation_rag_chain = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return conversation_rag_chain


#[AI Message 함수 정의]===================================================
def get_ai_message(user_message, session_id='default') :
    conversation_rag_chain = get_retrievalQA()
    ai_message = conversation_rag_chain.invoke(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}},
        )
    
    # print(f'대화이력>>> {get_session_history(session_id)}\n\n')
    # print('='*50+'\n')
    
    return ai_message



