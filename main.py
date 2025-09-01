import os
from typing import List
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub
from langchain_core.tools import render_text_description
from langchain.schema import Document


class WebsiteQAAgent:
    def __init__(self):
        # Initialize Ollama Mistral
        self.llm = ChatOllama(
            model="mistral",
            temperature=0.1
        )

        # Initialize Nomic embeddings with local inference
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
        )

        # Initialize vector store
        self.vector_store = None
        self.qa_chain = None

        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def scrape_website(self, url: str) -> List[Document]:
        """Scrape content from a website and return as documents"""
        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text content
            text = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            # Create document
            doc = Document(
                page_content=text,
                metadata={"source": url, "title": soup.title.string if soup.title else ""}
            )

            return [doc]

        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return []

    def load_websites(self, urls: List[str]):
        """Load multiple websites and create vector store"""
        all_documents = []

        for url in urls:
            print(f"Loading content from: {url}")
            docs = self.scrape_website(url)
            all_documents.extend(docs)

        if not all_documents:
            raise ValueError("No documents were loaded successfully")

        # Split documents into chunks
        splits = self.text_splitter.split_documents(all_documents)
        print(f"Created {len(splits)} text chunks")

        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )

        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        print("Vector store created and QA chain initialized!")

    def create_tools(self):
        """Create tools for the agent"""

        def website_qa_tool(query: str) -> str:
            """Answer questions based on loaded website content"""
            if not self.qa_chain:
                return "No websites have been loaded yet. Please load websites first."

            try:
                result = self.qa_chain.invoke({"query": query})
                answer = result["result"]
                sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]

                response = f"Answer: {answer}\n\nSources: {', '.join(set(sources))}"
                return response
            except Exception as e:
                return f"Error answering question: {str(e)}"

        def load_website_tool(urls_string: str) -> str:
            """Load websites for analysis. Provide URLs separated by commas."""
            try:
                urls = [url.strip() for url in urls_string.split(",")]
                self.load_websites(urls)
                return f"Successfully loaded {len(urls)} websites into the knowledge base."
            except Exception as e:
                return f"Error loading websites: {str(e)}"

        return [
            Tool(
                name="load_websites",
                description="Load websites for analysis. Input should be URLs separated by commas.",
                func=load_website_tool
            ),
            Tool(
                name="answer_from_websites",
                description="Answer questions based on loaded website content. Input should be a question.",
                func=website_qa_tool
            )
        ]

    def create_agent(self):
        """Create the LangChain agent"""
        tools = self.create_tools()

        # Get the ReAct prompt from LangChain hub
        prompt = hub.pull("hwchase17/react")

        # Customize the prompt for website QA
        prompt = prompt.partial(
            tools=render_text_description(tools),
            tool_names=", ".join([t.name for t in tools]),
        )

        # Create ReAct agent
        agent = create_react_agent(self.llm, tools, prompt)

        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

        return agent_executor


# Usage Example
def main():
    # Initialize the agent
    qa_agent = WebsiteQAAgent()
    agent_executor = qa_agent.create_agent()

    print("Website QA Agent initialized!")
    print("Available commands:")
    print("1. Load websites: 'Load these websites: url1, url2, url3'")
    print("2. Ask questions: 'What does the website say about X?'")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'quit':
            break

        if not user_input:
            continue

        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"\nAgent: {response['output']}\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")


if __name__ == "__main__":
    main()
