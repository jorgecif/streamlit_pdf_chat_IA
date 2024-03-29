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
import base64




# Load secrets deploy - OpenAI API key
openai_api_key=st.secrets["OPENAI_API_KEY"] # Opción para Streamlit share

# Load secrets local
#from dotenv import load_dotenv
#load_dotenv()


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

# Function to visualize PDF
def displayPDF(upl_file):
    # Read file as bytes:
    bytes_data = upl_file.getvalue()

    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8")

    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width=100% height=300px type="application/pdf"></iframe>'

    # Display file
    st.sidebar.markdown(pdf_display, unsafe_allow_html=True)

# Logo sidebar
image = PIL.Image.open('logo_blanco.png')
st.sidebar.image(image, width=None, use_column_width=None)


st.header("Pregúntale a tu PDF 💬")




# Upload  file
pdf = st.sidebar.file_uploader("Sube un documento en formato .pdf", type=['pdf'] )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if pdf is not None:
    #ui_width = st_javascript("window.innerWidth")
    displayPDF(pdf)

    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Simple numbers about pdf
    n_pages= len(pdf_reader.pages)
    n_char=len(text)
    n_words=len(text.split())
    st.sidebar.write("Número de páginas ", n_pages)
    st.sidebar.write("Número de caracteres ", n_char)
    st.sidebar.write("Número de palabras ", n_words)

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

    user_question = st.chat_input("Qué quieres saber de tu documento?")
   
    if user_question:
        st.chat_message("user").markdown(user_question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        docs = knowledge_base.similarity_search(user_question)

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        success()
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})



