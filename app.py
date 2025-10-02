import streamlit as st
from langchain_core.messages import HumanMessage
from services.pipeline import get_weather_qa_chain

# Streamlit Initialization
st.set_page_config(page_title="Weather RAG App", layout="centered")
st.title("Weather RAG App")
st.markdown("Ask about weather data")


# Initialize the weather RAG chain and chat history
if "qa_chain" not in st.session_state:
    try:
        st.session_state.qa_chain = get_weather_qa_chain()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
        
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display the chat messages in user and assistant modes
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)


if user_query := st.chat_input("Ask a question about the weather data..."):
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Thinking... Please wait"):
        try:
            response = st.session_state.qa_chain.invoke(
                {"question": user_query, "chat_history": st.session_state.chat_history}
            )
            ai_response = response["answer"]
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(response["chat_history"][-1])

        except Exception as e:
            ai_response = f"An error occurred: {e}"
            st.error(ai_response)

    with st.chat_message("assistant"):
        st.markdown(ai_response)