import streamlit as st
from llm import get_ai_message


st.set_page_config(page_title='전세사기피해 상담 챗봇', page_icon='📰')
st.title('🤖전세사기피해 상담 챗봇')

if 'message_list' not in st.session_state:
    st.session_state.message_list = []  #메시지리스트 키를 세션스테이트에 저장함

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

    with st.spinner('Generating the reply...😎'):
        ai_message = get_ai_message(user_question)  #if문 안에 있어야함 

        with st.chat_message('ai'):
            st.write(ai_message)
        st.session_state.message_list.append({'role':'ai','content':ai_message})



