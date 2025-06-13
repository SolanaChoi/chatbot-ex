import os

from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


## 환경변수 읽어오기 =====================================================
load_dotenv()

## LLM 생성 ==============================================================
def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm

## Embedding 설정 + Vector Store Index 가져오기 ===========================
def get_database():
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

    ## 저장된 인덱스 가져오기
    Pinecone(api_key=PINECONE_API_KEY)
    database = PineconeVectorStore.from_existing_index(
        index_name='retrieval02',
        embedding=OpenAIEmbeddings(model='text-embedding-3-large'),
    )

    return database

## 세션별 히스토리 저장 =================================================
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


## 히스토리 기반 리트리버 ===============================================
def get_history_retriever(llm, retriever):
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    #문맥에 맞추어 질문 재구성할 수 있도록 하는 프롬프트. 답변은 안함
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever


def get_qa_prompt() :
    system_prompt = (
    '''[identity]
- 당신은 전세사기피해 법률 전문가입니다.
- [context]를 참고하여 사용자의 질문에 답변하세요.
- 답변에는 해당 조항을 '(XX법 제X조 제X항 제X호, XX법 제X조 제X항 제X호)' 형식으로 문단 마지막에 표시하세요.
- 항목별로 표시해서 답변해주세요.
- 전세사기피해 법률 이외의 질문에는 '답변할 수 없습니다.'로 답하세요.

[context]
{context} 
'''   
    )
    # few-shot 예시
    from langchain_core.prompts import PromptTemplate
    from langchain_core.prompts import FewShotPromptTemplate
    from config_fewshot import answer_examples

    example_prompt = PromptTemplate.from_template("질문: {input}\n답변: {answer}")


    few_shot_prompt = FewShotPromptTemplate(
        examples=answer_examples,  #질문/답변 예시들 (전체type: list, 질/답type: dict)
        example_prompt=example_prompt, #단일 예시가 들어갈 포맷. 질답 하나당 포맷에 들어가서 세 포맷 생김
        prefix='다음 질문에 답변하세요::', #위의 포맷들의 가장 위에 옴. 
        suffix="질문하세요: {input}",  #위의 포맷들의 가장 밑에 옴. 실제 사용자 질문이 들어갈 변수
        input_variables=["input"],
    )

    formatted_few_shot_prompt = few_shot_prompt.format(input='{input}')  #.format = 앞의것(여기선 f_s_prompt)를 str로 치환해주는 메서드, 그런데 이 때, fsp의 input도 str로 치환해버리기에 input을 다시 변수화 하는 작업  

    #답변까지 해주는 프롬프트. 퓨샷은 이 곳에 넣기 
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("assistant", formatted_few_shot_prompt),   #예시 넣어주는거임 휴먼의 인풋은 여기 인풋으로도 들어간다....
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return qa_prompt


## 전체 chain 구성 =================================================
def build_conversational_chain():
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    ## LLM 모델 지정
    llm = get_llm()

    ## vector store에서 index 정보
    database = get_database()
    retriever = database.as_retriever(search_kwargs={'k': 2})

    history_aware_retriever = get_history_retriever(llm, retriever)

    qa_prompt = get_qa_prompt()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key='answer',
    ).pick('answer')

    return conversational_rag_chain


## AI Message ===========================================================
def stream_ai_message(user_message, session_id='default'):
    qa_chain = build_conversational_chain()

    ai_message = qa_chain.stream(
        {'input': user_message},
        config={'configurable': {'session_id': session_id}},        
    )

    print(f'대화 이력 >> {get_session_history(session_id)} \n😎\n')
    print('=' * 50 + '\n')
    print(f'[stream_ai_message 함수 내 출력] session_id >> {session_id}')

    return ai_message

