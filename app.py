import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import PIL.Image
from streamlit_option_menu import option_menu
from streamlit_extras.let_it_rain import rain

# Load secrets - OpenAI API key
openai_api_key=st.secrets["OPENAI_API_KEY"] # Opción para Streamlit share

# Page features
st.set_page_config(
    page_title="Herramientas AI - Qüid Lab",
    page_icon="random",
    layout="centered",
    initial_sidebar_state="expanded",
)


# Style
hide_streamlit_style = """
				<style>
				#MainMenu {visibility: hidden;}

				footer {visibility: hidden;}
				</style>
				"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# Function success
def success():
	rain(
		emoji="🎈",
		font_size=54,
		falling_speed=5,
		animation_length=1, #'infinite'
	)



# Logo sidebar
image = PIL.Image.open('logo_blanco.png')
st.sidebar.image(image, width=None, use_column_width=None)


# Title and header
st.set_page_config(page_title="Ask Your PDF")
st.set_page_config(page_title="Ask Your PDF")
st.header("Ask your PDF 💬")

# extract the text
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("Ask a question about your PDF")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        success()

        st.write(response)
