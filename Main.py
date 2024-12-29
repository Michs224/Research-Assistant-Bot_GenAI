import os
import streamlit as st
import pickle
import time
import langchain
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
import faiss
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import io
import tempfile

load_dotenv()

st.title('📈 Research Assistant Bot')

st.sidebar.title('Article URL/File')

input_type = st.sidebar.radio('Choose Input Type', ['URL', 'Upload PDF'])

if input_type == 'URL':
    urls=[]
    for i in range(3):
        url = st.sidebar.text_input(f'Article URL {i+1}')
        urls.append(url)
elif input_type == 'Upload PDF':
    uploaded_files = st.sidebar.file_uploader("Upload File (PDF)", type=['pdf'], accept_multiple_files=True)

process_clicked = st.sidebar.button('Process')
file_path = 'faiss_store_gistv0.pkl'

main_placefolder = st.empty()

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens=500,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

if process_clicked:
    # Load data
    main_placefolder.text('Data loading...▶️')

    if input_type == "URL":
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

    elif input_type == "Upload PDF":
        # Handle file uploads for PDFs
        data = []
        for uploaded_file in uploaded_files:
            # file_bytes = uploaded_file.read()
            # file = io.BytesIO(file_bytes)
            # print(file)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_file.read())  # Write file to temp
                temp_file_path = temp_file.name  # Get the file path
            loader = PyPDFLoader(temp_file_path, extract_images=True)
            pdf_data = loader.load()
            for doc in pdf_data:
                doc.metadata['source'] = uploaded_file.name
            data.extend(pdf_data)

            os.remove(temp_file_path)

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        # separators=['\n\n', '\n', '. ', ', '],
        chunk_size=1000
        # chunk_overlap=100
    )
    main_placefolder.text('Data split...⏩')
    docs = text_splitter.split_documents(data)
    print(len(docs))
    # Create embeddings and save it to FAISS index
    embeddings = SentenceTransformerEmbeddings(model_name="avsolatorio/GIST-small-Embedding-v0")
    main_placefolder.text('Embedding Vector...✅')
    vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)
    # time.sleep(3)
    main_placefolder.text('Vector Store...✅')
    with open(file=file_path, mode='wb') as f:
        pickle.dump(vector_store, f)

query = main_placefolder.text_input("Question: ")
ask = st.button('Ask')
if query or ask:
    if os.path.exists(path=file_path):
        with open(file=file_path, mode='rb') as f:
            vector_store = pickle.load(f)

        # Load QA chain
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(provider=res, device=0, index=vector_store.index)
        vector_store.index = gpu_index
        retriever = vector_store.as_retriever(search_kwargs={'k': 10})
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        result = chain({'question': query}, return_only_outputs=True)
        # {"answer": "", "sources": []}
        st.header('Answer:')
        st.text(result['answer'])

        # Display sources, if any
        sources = result.get('sources', "")
        # print(result)
        print(sources)
        if sources:
            st.subheader('Sources:')
            sources_list = sources.split('\n')
            for source in sources_list:
                st.write(source)

