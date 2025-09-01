from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


def create_qa_chain(documents):
    print(f"Creating QA chain with {len(documents)} documents")

    # Split documents into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Split all documents
    split_docs = []
    for doc in documents:
        if isinstance(doc, Document):
            chunks = text_splitter.split_documents([doc])
            split_docs.extend(chunks)
        else:
            # If it's just text, convert to Document first
            doc_obj = Document(page_content=str(doc))
            chunks = text_splitter.split_documents([doc_obj])
            split_docs.extend(chunks)

    print(f"Split into {len(split_docs)} chunks")

    # embedding using OpenAi
    # embeddings = OpenAI(model="text-embedding-3-small")

    # Create embeddings with Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create vector store
    vectorstore = Chroma.from_documents(split_docs, embeddings)

    # for OpenAI models with larger than 10 webiste to handle the large contexts
    # llm = ChatOpenAI(
    #     model="gpt-3.5-turbo",
    #     temperature=0.1,  # Lower temperature for more focused answers
    #     num_predict=512
    # )
    # Mistral model used by Ollama - 7billion parameter
    llm = ChatOllama(
        model="mistral",
        temperature=0.1,  # Lower temperature for more focused answers
        num_predict=512,  # Limit response length
    )

    # Create retriever with more documents
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve more relevant chunks
    )

    # Create QA chain with custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,  # Set to True if you want to see sources
        verbose=True  # Enable verbose mode for debugging
    )

    print("QA chain created successfully")
    return qa_chain
