import streamlit as st
from llm import get_ai_message


st.set_page_config(page_title='전세사기피해 상담 챗봇', page_icon='📰')
st.title('👻전세사기피해 상담 챗봇')

print('\n=====start=====')
print('st.session_state>>', st.session_state)

###===================================================================
import uuid

# print('UUID는', uuid.uuid4())  # 완전 랜덤 방식(UUID 버전4)
# print('UUID의 타입', type(uuid.uuid4()))  # 완전 랜덤 방식(UUID 버전4)



# 세션id에 고유한 값 설정====================
#[방법1] 새로고침하면 새로발급
# if 'message_list' not in st.session_state:
#     #세션 id를 생성하여 저장 
#     st.session_state['session_id']= str(uuid.uuid4())
#     st.session_state.session_id = str(uuid.uuid4())
#     print('st.session_state.session_id>>',)


#[방법2] url의 parameter에 저장
query_params = st.query_params
st.query_params.update({'age':29})  #url에 파라미터 추가하는 것 

### 쿼리 파라미터에 session_id가 있으면 값을 가져오고 없으면 파라미터 설정


#Query파라미터
print('st.query_params>>', st.query_params)
print('session_id 값 추출 >>', st.query_params['session_id'])
#=====st.query_params.session_id 윗줄과 같은 문법

query_params = st.query_params  #딕셔너리 

if 'session_id' in query_params:
    session_id = query_params.session_id
    print('url에 session_id가 있다면.... UUID를 가져와서 변수 저장')

else:
    session_id = str(uuid.uuid4())
    st.query_params.update({'session_id': session_id})
    print('url에 session_id가 없다면.... UUID를 생성하여 추가')

print('after))session_state>>', st.session_state)
#####============================================================

# 스트림릿 내부 세션에도 저장
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = session_id
    print('[Streamlit 내부 세션] st.session_state.session_id>>', st.session_state.session_id)


#세션 초기화 - 상자에 기본값 넣는 작업 message_list라는 기억이 없으면, 빈리스트로 생성 
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

#이전 채팅 내용 화면 출력================================================
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


#채팅 메시지============================================================
placeholder='전세사기피해 관련 내용을 질문하세요.'  
if user_question := st.chat_input(placeholder=placeholder):  #prompt창
    with st.chat_message('user'):
        st.write(user_question)
    st.session_state.message_list.append({'role':'user','content':user_question})

# AI메시지 = get_ai_message(user_message=user_question, session_id='ywgw')
    with st.spinner('Generating the reply...😎'):
        # ai_message = get_ai_message(user_question)

        # session_id = 'user-session'
        session_id = st.session_state.session_id
        ai_message = get_ai_message(user_message=user_question, session_id='ywgw')

        with st.chat_message('ai'):
            st.write(ai_message)
        st.session_state.message_list.append({'role':'ai','content':ai_message})



