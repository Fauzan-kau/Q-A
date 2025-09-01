import streamlit as st
from loaders import MultiUrlLoader
from qa_chain import create_qa_chain
from voice_io import speech_to_text

st.title("Voice-Driven Q&A Application")

# Initialize session state
if "documents" not in st.session_state:
    st.session_state.documents = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "num_urls" not in st.session_state:
    st.session_state.num_urls = 1

# URL input section
st.header("Configure URL Loading")

# Ask for number of websites
num_urls = st.number_input(
    "How many websites do you want to load?",
    min_value=1,
    max_value=10,
    value=st.session_state.num_urls,
    step=1
)

# Update session state
st.session_state.num_urls = num_urls

st.subheader("Enter URLs to Load Content")

# Generate dynamic URL input fields based on the number specified
urls = []
for i in range(num_urls):
    url = st.text_input(f"Enter URL {i + 1}:", key=f"url_{i}")
    if url.strip():
        urls.append(url.strip())

if st.button("Load Content and Initialize"):
    if not urls:
        st.error("Please enter at least one URL.")
    else:
        with st.spinner("Loading and processing URLs..."):
            try:
                loader = MultiUrlLoader(urls)
                documents = loader.load()

                if not documents:
                    st.error("No content loaded from URLs. Please check the URLs and ensure they are accessible.")
                else:
                    st.success(f"Loaded {len(documents)} documents.")

                    # Show some document info for debugging
                    st.write("Document preview:")
                    for i, doc in enumerate(documents[:2]):  # Show first 2 docs
                        st.write(f"Doc {i + 1}: {doc.page_content[:200]}...")

                    # Create QA chain
                    qa_chain = create_qa_chain(documents)
                    st.session_state.documents = documents
                    st.session_state.qa_chain = qa_chain
                    st.success("QA system initialized successfully!")

            except Exception as e:
                st.error(f"Error loading content: {str(e)}")

# Text input for questions (for testing)
st.header("Ask Questions")
if st.session_state.qa_chain is not None:
    text_question = st.text_input("Or type your question here:")
    if st.button("Ask Text Question") and text_question:
        with st.spinner("Getting answer..."):
            try:
                answer = st.session_state.qa_chain.run(text_question)
                st.write(f"**Question:** {text_question}")
                st.write(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"Error getting answer: {str(e)}")

# Voice input section
if st.button("Ask Question by Voice"):
    if st.session_state.qa_chain is None:
        st.error("Please load content first.")
    else:
        with st.spinner("Listening..."):
            try:
                question = speech_to_text()
                if question.startswith("Sorry"):
                    st.error(question)
                else:
                    st.write(f"**Question:** {question}")
                    with st.spinner("Getting answer..."):
                        answer = st.session_state.qa_chain.run(question)
                        st.write(f"**Answer:** {answer}")
                        # Uncomment if you want text-to-speech
                        # text_to_speech(answer)
            except Exception as e:
                st.error(f"Error with voice input: {str(e)}")

# Debug section
if st.session_state.documents:
    with st.expander("Debug Information"):
        st.write(f"Total documents loaded: {len(st.session_state.documents)}")
        st.write("Document sources:")
        for i, doc in enumerate(st.session_state.documents):
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                st.write(f"- {doc.metadata['source']}")
            st.write(f"  Content length: {len(doc.page_content)} characters")
            if i >= 5:  # Limit display
                st.write("... and more")
                break
