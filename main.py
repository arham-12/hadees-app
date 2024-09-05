import os
import faiss
import numpy as np
import pandas as pd
import streamlit as st 
from  streamlit_chat import message
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

st.title("Ai Hadees App")


st.markdown("generate your GROQ API key from : [API_KEY](https://console.groq.com/keys)")

api_key = st.text_input("Enter your GROQ API key", type="password")

csv_file = 'document_metadata (1).csv'
emb = 'faiss_index (1).bin'
if api_key:
# Initialize the language model with specified parameters
    llm = ChatGroq(
        api_key=api_key,
        model_name="llama-3.1-70b-versatile",
        temperature=0.3,
    )


if "chats" not in st.session_state:
    st.session_state.chats = []


def load_vectorstore():
    """
    Load the FAISS index and document metadata from saved files.
    """
    index = faiss.read_index(emb)
    document_data = pd.read_csv(csv_file)
    return index, document_data

def generate_embeddings(texts):
    """
    Generate embeddings for a list of texts using Hugging Face Inference API.

    Args:   
        texts (list): A list of texts for which embeddings are to be generated.

    Returns:
        list: A list of embeddings.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings.embed_documents(texts)


def retrieve_context(question, index, document_data, top_k=3):
    """
    Retrieves relevant documents from the vector store based on the user's question.

    Args:
        question (str): The user's question for which relevant context documents are to be retrieved.
        index (faiss.Index): The FAISS index used for similarity search.
        document_data (pd.DataFrame): DataFrame containing document metadata and embeddings.
        top_k (int): The number of top similar documents to retrieve.

    Returns:
        str: Combined content of the relevant context documents.
    """
    # Generate the embedding for the query
    query_vector = generate_embeddings([question])[0]  # Ensure generate_embeddings returns a single vector

    # Convert query_vector to numpy array and reshape if needed
    query_vector = np.array(query_vector).astype(np.float32).reshape(1, -1)

    # Perform similarity search
    distances, indices = index.search(query_vector, top_k)

    # Extract document metadata and content
    retrieved_content = []
    for idx in indices[0]:
        if idx >= 0:  # Ensure index is valid
            doc_uuid = document_data.iloc[idx]['uuid']
            content = document_data[document_data['uuid'] == doc_uuid]['content'].values[0]
            pdf_name = document_data[document_data['uuid'] == doc_uuid]['pdf_name'].values[0]
            page = document_data[document_data['uuid'] == doc_uuid]['page'].values[0]
            retrieved_content.append(f"\n reffernce: {pdf_name}, Page: {page}\n content:\n{content}")

    # Combine the content of all retrieved documents into a single string
    context = "\n".join(retrieved_content)
    
    return context

def generate_chat_response(question, index, document_data, api_key):
    """
    Generates a chat response based on the user's question, context retrieved from documents, and model parameters.

    Args:
        question (str): The user's question or query.
        index (faiss.Index): The FAISS index used for similarity search.
        document_data (pd.DataFrame): DataFrame containing document metadata.
        api_key (str): API key for accessing the language model.

    Returns:
        str: The response content from the AI model.
    """
    # Initialize memory to keep track of the conversation history
    history = StreamlitChatMessageHistory(key="chat_messages")

    # Retieve relevant context from documents based on the user's question
    content = retrieve_context(question, index, document_data)
    # st.write(content)
    
    # Define the prompt template for the chatbot conversation
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert Islamic scholar that have full knowladge of sahi bukhari and proficient in Urdu and Arabic language. You are provided with a collection of Hadiths from Sahih Bukhari in English. When a user asks a question or provides a query, your task is to:1. Generate a response in Urdu language that addresses the user's query with the relevant Hadiths along with reference of that hadees which is provided you in the relivent content.2.translate the Hadith into Arabic and urdu  language.3. Explain the Hadith or the relevant content in Urdu to provide a comprehensive answer. Make sure the response is accurate, respectful, and clear. The translation into Arabic should be precise, and the explanation in Urdu should be accessible and detailed."),  # System message introducing the assistant's role
            MessagesPlaceholder(variable_name="history"),  # Placeholder for chat history
            ("human", """explain the topic of user : {question} with reference of hadith in arabic and urdu.
            this is Relevant material for you : {context}.If the user question is not relivent about the islam then say please ask relivent question about islam and dont give any other explanation or hadees reference."""),
            # Message template for user input, with placeholders for user's question and context
        ]
    )
    
    formatted_prompt = prompt.format_prompt(
        question=question,
        history=history.messages,
        context=content
    )
    # print(prompt.invoke({"question": question, "content": content}))
    chain = prompt | llm


    # Create a chain with the prompt and language model
    runable_chain = RunnableWithMessageHistory(
        chain,
        get_session_history=lambda: history,
        input_messages_key="question",
        history_messages_key="history"
    )
    # st.write(runable_chain)
    # Add the user's message to the history
    history.add_user_message(question)
   
  
    # Get the AI response from the language model using the chain with history
    config = {"configurable": {"session_id": "any"}}
    response = runable_chain.invoke({
        "question": question,
        "history": history.messages,
        "context": content
    }, config)

    # Add the AI response to the history
    history.add_ai_message(response.content)
    st.session_state.chats.append({"user": question, "ai": response.content})
    # Return the response content from the AI model
    return response.content


def main():
    path = 'image/Screenshot from 2024-08-31 16-50-58.png'

    st.sidebar.image(path, use_column_width=True)


    user_input = st.chat_input("Enter your question regarding to nukhari ")
   
    if user_input:
      
        if api_key:
            index, document_data = load_vectorstore()
            response = generate_chat_response(question=user_input, api_key=api_key,index=index, document_data=document_data)
        else:
            st.info("Please provide an API key to use the chatbot.")
        for i, chat in enumerate(st.session_state.chats):
        # Create two columns: one for the AI message (left) and one for the user message (right)
        
            with st.container():
                message(chat["user"], is_user=True, key=f"user_{i}")
            
            with st.container():
                message(chat["ai"], key=f"ai_{i}")

            # st.chat_message(f"ai : {chat["ai"]}")
            
   


    






if __name__ == "__main__":
    main()