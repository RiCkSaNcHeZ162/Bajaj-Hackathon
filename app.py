from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
import cassio
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# setup huggingface embeddings
ASTRA_ID = os.getenv("ASTRA_ID")
ASTRA_TOKEN = os.getenv("ASTRA_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# setting up the vectorstore and llm
cassio.init(token=ASTRA_TOKEN, database_id=ASTRA_ID)
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="bajaaj_factsheet_oct",
    session=None,
    keyspace=None,
)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o")

session_id = st.text_input("Session ID", value="default_session")

## statefully manage chat history

if "store" not in st.session_state:
    st.session_state.store = {}

retriever = astra_vector_store.as_retriever()
history_aware_retrieval_sytem_prompt = (
    "Given chat history and latest asked question"
    "which might reference to context in chat history"
    "formulate a standalone question which can be understood,"
    "without the chat history. DO NOT answer the question,"
    "just refrmulate it otherwise return as it is"
)

history_aware_retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", history_aware_retrieval_sytem_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

## q n a flow prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "You are given a factsheet to study answer as if you are expert of finance"
    "Use the following pieces of retrieved context to answer "
    "the question. IF YOU DON'T KNOW THE ANSWER THEN SAY THAT YOU DON'T KNOW"
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# creating history aware retriever and q n' a retrieval chain
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, history_aware_retrieval_prompt
)
stuff_chain = create_stuff_documents_chain(llm, qa_prompt)
retrieval_chain = create_retrieval_chain(history_aware_retriever, stuff_chain)


def get_session_history(session_id) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    retrieval_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

user_input = st.text_input("Your question:")
if user_input:
    session_history = get_session_history(session_id)
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id": session_id}
        },  # constructs a key "abc123" in `store`.
    )
    st.write(st.session_state.store)
    st.write("Assistant:", response["answer"])
    st.write("Chat History:", session_history.messages)
