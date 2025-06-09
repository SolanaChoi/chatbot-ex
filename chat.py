import streamlit as st

st.set_page_config(page_title='전세사기피해 상담 챗봇', page_icon='📰')
st.title('🤖전세사기피해 상담 챗봇')

if 'message_list' not in st.session_state:
    st.session_state.message_list = []  #메시지리스트 키를 세션스테이트에 저장함


print(f'before: {st.session_state.message_list}')

# #이전 채팅 내용 화면 출력
# for message in st.session_state.message_list:
#     print('message>>', message)
#     print('message["role"]>>', message['role'])
#     print('message["content"]>>', message['content'])

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        #사용자 메시지 화면 출력
        st.write(message["content"])


#prompt창(채팅창)===========================================================
placeholder='전세사기피해 관련 궁금한 내용을 질문하세요.>>' #코드 옆으로길어짐 방지
if user_question := st.chat_input(placeholder=placeholder):
    with st.chat_message('user'):
        #사용자 메시지 화면출력
        st.write(user_question)
    st.session_state.message_list.append({'role':'user','content':user_question})

with st.chat_message('ai'):
    st.write('여기는 AI메시지')
st.session_state.message_list.append({'role':'ai','content':'여기는 AI메시징'})


print(f'after: {st.session_state.message_list}')

#write : 화면에 찍힘
#print : 서버에 찍힘. 개발자가 보기 위함 





