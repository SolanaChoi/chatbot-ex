import uuid
import streamlit as st
from llm import get_ai_message


st.set_page_config(page_title='전세사기피해 상담 챗봇', page_icon='📰')
st.title('🎈전세사기피해 상담 챗봇')

print('\n=====start=====')
print('BEFORE) st.session_state>>', st.session_state)

# url의 parameter에 session_id 저장 및 가져오기==============================================
query_params = st.query_params

if 'session_id' in query_params:
    session_id = query_params.session_id
else:
    session_id = str(uuid.uuid4())
    st.query_params.update({'session_id': session_id})

# Streamlit 내부 세션: session_id 저장
if 'session_id' not in st.session_state:
    st.session_state.session_id = session_id

# Streamlit 내부 세션: message_list 초기화 message_list라는 기억이 없으면, 빈리스트로 생성 
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

print('AFTER) st.session_state>>', st.session_state)

# 이전 채팅 내용 화면 출력: 과거 대화 내용이 화면에 뿌려짐 (질문을 하는것만으로 페이지화면 리셋되므로)
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 질문-> AI답변===================================================
placeholder='전세사기피해 관련 내용을 질문하세요.'  
if user_question := st.chat_input(placeholder=placeholder):  #prompt창
    with st.chat_message('user'):
        st.write(user_question)
    st.session_state.message_list.append({'role':'user','content':user_question})

# AI메시지 = get_ai_message(user_message=user_question, session_id='ywgw')
    with st.spinner('Generating the reply...😎'):
        session_id = st.session_state.session_id
        ai_message = get_ai_message(user_message=user_question, session_id='ywgw')

        with st.chat_message('ai'):
            ai_message = st.write_stream(ai_message)
        st.session_state.message_list.append({'role':'ai','content':ai_message})

