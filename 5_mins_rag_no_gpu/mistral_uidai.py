# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a simple standalone implementation showing rag pipeline using Nvidia AI Foundational models.
# It uses a simple Streamlit UI and one file implementation of a minimalistic RAG pipeline.

############################################
# Component #1 - Document Loader
############################################

import streamlit as st
from langchain_core.messages import HumanMessage
import os
from streamlit import session_state as ss
st.set_page_config(layout = "wide")

os.environ['NVIDIA_API_KEY']='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'
with st.sidebar:
    DOCS_DIR = os.path.abspath("./uploaded_docs")
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    st.subheader("Add to the Knowledge Base")
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:", accept_multiple_files = True)
        submitted = st.form_submit_button("Upload!")

    if uploaded_files and submitted:
        for uploaded_file in uploaded_files:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            with open(os.path.join(DOCS_DIR, uploaded_file.name),"wb") as f:
                f.write(uploaded_file.read())

############################################
# Component #2 - Embedding Model and LLM
############################################

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# make sure to export your NVIDIA AI Playground key as NVIDIA_API_KEY!
llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
document_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="passage")
query_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="query")

############################################
# Component #3 - Vector Database Store
############################################

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
import pickle

with st.sidebar:
    # Option for using an existing vector store
    use_existing_vector_store = st.radio("Use existing vector store if available", ["Yes", "No"], horizontal=True)

# Path to the vector store file
vector_store_path = "vectorstore_mistral.pkl"




# Check for existing vector store file
vector_store_exists = os.path.exists(vector_store_path)
vectorstore = None
if use_existing_vector_store == "Yes" and vector_store_exists:
    with open(vector_store_path, "rb") as f:
        vectorstore = pickle.load(f)
    with st.sidebar:
        st.success("Existing vector store loaded successfully.")
else:
    # Load raw documents from the directory
    raw_documents = DirectoryLoader(DOCS_DIR).load()
    with st.sidebar:
        if raw_documents:
            with st.spinner("Splitting documents into chunks..."):
                text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                documents = text_splitter.split_documents(raw_documents)

            with st.spinner("Adding document chunks to vector database..."):
                vectorstore = FAISS.from_documents(documents, document_embedder)

            with st.spinner("Saving vector store"):
                with open(vector_store_path, "wb") as f:
                    pickle.dump(vectorstore, f)
            st.success("Vector store created and saved.")
        else:
            st.warning("No documents available to process!", icon="??")

############################################
# Component #4 - LLM Response Generation and Chat
############################################

st.subheader("Chat with your AI Assistant, Aadhar Sarathi !")

if "messages" not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

from langchain_core.output_parsers import StrOutputParser
user_input = st.chat_input("Can you tell me how to enroll into aadhar ?")
llm = ChatNVIDIA(model="mixtral_8x7b")

#chain = prompt_template | llm | StrOutputParser()

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might refer context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
qa_system_prompt = """You are a helpful AI assistant named Aadhar Sarathi. You will reply to questions only based on documents given to you \
and the history of the current session chat else decline to answer. \
You will not  mention the phrases similar to "assuming the context" or "assist". \
Never include "[Assuming the context is]" in your response. \
Never include "As a helpful AI assistant, I will answer your question based on the provided document." \
Do not give extra information. Do not output assumptions\
If something is out of context, you will refrain from replying and decline to respond to the user. The answers should be precise and should include website links \
and e-mail address whereever applicable. \
Do not include the source document name or any reference to it in your response.

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
if user_input and vectorstore!=None:
    st.session_state.messages.append({"role": "user", "content": user_input})
    retriever = vectorstore.as_retriever()
    history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    docs = retriever.get_relevant_documents(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"

    augmented_user_input = "Context: " + context + "\n\nQuestion: " + user_input + "\n"

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        output=rag_chain.invoke({"input": user_input+' answer specifically with respect to question context', "chat_history": ss.chat_history})
        for response in output['answer'].split():
            #print(response)
            full_response += response+" "
            message_placeholder.markdown(full_response + "�")
        message_placeholder.markdown(full_response)
        ss.chat_history.extend([HumanMessage(content=user_input),full_response])
    st.session_state.messages.append({"role": "assistant", "content": full_response})
