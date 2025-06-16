import os

from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import (ChatPromptTemplate, MessagesPlaceholder, 
                                    FewShotPromptTemplate, PromptTemplate)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from config_fewshot import answer_examples  #퓨샷예시를 파일로 임포트하지 않고 벡터디비화하는게 더 토큰 적게 씀 



## 환경변수 읽어오기 =====================================================
load_dotenv()

## LLM 생성 ==============================================================
def load_llm(model='gpt-4o'):
    return ChatOpenAI(model=model)

## Embedding 설정 + Vector Store Index 가져오기 ===========================
def load_vectorstore():
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
def build_history_aware_retriever(llm, retriever):  #대화 내역 기억함으로 수준 높은 질문 생성 위함 
    contextualize_q_system_prompt = ('''이전 대화 기록과 최신 사용자 질문을 바탕으로, 대화 맥락 없이도 완전히 이해할 수 있는 독립형 질문을 다시 작성하세요. 질문에 대한 답변은 절대 작성하지 말고,
    필요할 경우에만 질문을 재구성하고, 그렇지 않으면 원문을 그대로 반환하세요.''')

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


def build_few_shot_examples() -> str:
    example_prompt = PromptTemplate.from_template("질문: {input}\n답변: {answer}")

    few_shot_prompt = FewShotPromptTemplate(
        examples=answer_examples,  #질문/답변 예시들 (전체type: list, 질/답type: dict)
        example_prompt=example_prompt, #단일 예시가 들어갈 포맷. 질답 하나당 포맷에 들어가서 세 포맷 생김
        prefix='다음 질문에 답변하세요::', #위의 포맷들의 가장 위에 옴. 
        suffix="질문하세요: {input}",  #위의 포맷들의 가장 밑에 옴. 실제 사용자 질문이 들어갈 변수
        input_variables=["input"],
    )

    formatted_few_shot_prompt = few_shot_prompt.format(input='{input}')  #.format = 앞의것(여기선 f_s_prompt)를 str로 치환해주는 메서드, 그런데 이 때, fsp의 input도 str로 치환해버리기에 input을 다시 변수화 하는 작업 

    return formatted_few_shot_prompt

def build_qa_prompt() :
    
    #[keyword dictionary]==================================================================
    # 1. 기본형태 (가장 일반적)
    # * 장점: 키 하나당 설명 하나, 단순 빠름
    # * 용도(실무활용예): FAQ 챗봇, 버튼식 응답
    # keyword_dictionary = {
    #     '임대인':'임대인은 주택을 임차인에게 제공하고 계약 종료시 보증금을 반환할 의무가 있는 자입니다.',
    #     '주택':'주택은 주택임대차보호법 제2조에 따른 주거용 건물을 의미합니다. 1. 주택 임대차계약에 사용되는 건물로, 공시상 주거용 건물이 아닐지라도 임대차계약체결 당시 구조와 용도가 주거용일 경우 이를 포함합니다. 이 정의는 법률적으로 중요한 기준이 됩니다. 🧐'
    # }
    # 2. 질문형 키워드 (질문 다양성 대응)
    # * 장점: 유사한 질문을 여러 키로 분기하여 모두 같은 대답으로 연결, fallback대응
    # * 용도(실무활용예): 단답 챗봇, 키워드 FAQ 챗봇
    # keyword_dictionary = {
    #     '임대인 알려줘':'임대인은 주택을 임차인에게 제공하고 계약 종료시 보증금을 반환할 의무가 있는 자입니다.',
    #     '임대인 설명 ':'임대인은 주택을 임차인에게 제공하고 계약 종료시 보증금을 반환할 의무가 있는 자입니다.',
    #     '주택 알려줘':'주택은 주택임대차보호법 제2조에 따른 주거용 건물을 의미합니다. 1. 주택 임대차계약에 사용되는 건물로, 공시상 주거용 건물이 아닐지라도 임대차계약체결 당시 구조와 용도가 주거용일 경우 이를 포함합니다. 이 정의는 법률적으로 중요한 기준이 됩니다. 🧐'
    # }
    # 3. 키워드 + 태그 기반 딕셔너리
    keyword_dictionary = {
        '임대인':{
            'definition':'전세사기피해자법 제 2조 제 2항에 따른 임대인 정의입니다.',
            'source':'전세사기피해자법 제 2조',
            'tags':['법률', '용어', '기초'],   #tag는 llm모델이 보기도 함 
        },
        '주택':{
            'definition':'「주택임대차보호법」 제2조에 따른 주택의 정의입니다.',
            'source':'주택임대차보호법 제 2조',
            'tags':['법률', '용어', '기초'],   

        },
    }

    dictionary_text = '\n'.join([
        f'{k}{v["tags"]}:{v["definition"]} [출처:{v["source"]}]' for k, v in keyword_dictionary.items()
    ])

    system_prompt = (
    '''[identity]
    - 당신은 전세사기피해 법률 전문가입니다.
    - [context]와 [keyword_dictionary]를 참고하여 사용자의 질문에 답변하세요.
    - 답변에는 해당 조항을 '(XX법 제X조 제X항 제X호, XX법 제X조 제X항 제X호)' 형식으로 문단 마지막에 표시하세요.
    - 항목별로 표시해서 답변해주세요.
    - 전세사기피해 법률 이외의 질문에는 '답변할 수 없습니다.'로 답하세요.

    [context]
    {context} 

    [keyword_dictionary]
    {dictionary_text}
    '''   
        ) 
    
    formatted_fsp = build_few_shot_examples()

    #답변까지 해주는 프롬프트. 퓨샷은 이 곳에 넣기 
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("assistant", formatted_fsp),   #예시 넣어주는거임 휴먼의 인풋은 여기 인풋으로도 들어간다....
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    ).partial(dictionary_text=dictionary_text)

    print('\nqa_prompt>>\n', qa_prompt.partial_variables)

    return qa_prompt


## 전체 chain 구성 =================================================
def build_conversational_chain():
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    ## LLM 모델 지정
    llm = load_llm()

    ## vector store에서 index 정보
    database = load_vectorstore()
    retriever = database.as_retriever(search_kwargs={'k': 2})

    history_aware_retriever = build_history_aware_retriever(llm, retriever)

    qa_prompt = build_qa_prompt()
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

#=====================================================================
#vector store에서 검색된 문서 확인
    retriever = load_vectorstore().as_retriever(search_kwargs={'k':1})
    search_results = retriever.invoke(user_message)

    print(f'\nPinecone 검새액결과>> \n{search_results[0].page_content[:100]}')
    return ai_message

