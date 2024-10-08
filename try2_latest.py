from llama_index.llms.ollama import Ollama
from llama_index.readers.file import CSVReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from pathlib import Path
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.objects import ObjectIndex
from llama_index.core.query_engine import ToolRetrieverRouterQueryEngine
import streamlit as st

system_prompt = """
You are a multi-lingual expert system who has knowledge, based on 
real-time data. You will always try to be helpful and try to help them 
answering their question. If you don't know the answer, say that you DON'T
KNOW.

Jawablah semua dalam Bahasa Indonesia.
Anda adalah asesor yang memiliki tugas untuk memberikan tes asesmen kepada pengguna untuk menentukan 3 aspek teratas holland personality yang dimiliki oleh pengguna. Anda harus memberikan pertanyaan tes sesuai dengan dokumen "holland_questions.csv" yang ada sejumlah 6 pertanyaan. Setiap pertanyaan dalam dokumen harus Anda tanyakan satu per satu. Pertanyaan yang Anda tanyakan harus mencakup semua aspek RIASEC yang ada, jangan ada yang tertinggal, dan buat jumlah pertanyaan untuk setiap aspek sama. Tunggu pengguna untuk menjawab terlebih dahulu baru Anda boleh menanyakan pertanyaan selanjutnya. Pertanyaan harus sesuai dengan dokumen yang ada, namun pembahasaan dapat disesuaikan agar terdengar FRIENDLY DAN PADAT ke pengguna. Akan ada saatnya ketika pengguna bingung menjawab atau belum bisa memberikan jawaban yang konkrit. Saat hal itu terjadi, Anda TIDAK DIPERBOLEHKAN untuk melanjutkan ke pertanyaan selanjutnya, Anda wajib menjelaskan lebih dalam pertanyaan yang Anda sedang tanyakan, KEMUDIAN MINTA PENGGUNA UNTUK MENJAWAB KEMBALI.

Percakapan sejauh ini:
"""

Settings.llm = Ollama(model="llama3.1:latest", base_url="http://127.0.0.1:11434", system_prompt=system_prompt) 
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest") #buat apa?


#load/read documents using SimpleDirectoryReader
holland_infos = SimpleDirectoryReader("docs/holland/infos").load_data()
holland_questions = CSVReader(concat_rows=False).load_data(file = Path("docs/holland/questions/holland-questions.csv"))


infos_index = VectorStoreIndex.from_documents(holland_infos)

for docs in holland_questions:
    infos_index.insert(docs)

retriever = infos_index.as_retriever()

condense_question_prompt = """
Diberikan suatu percapakan (antara manusia dan asisten) dan sebuah pesan lanjutan dari manusia. Ubah pesan lanjutan menjadi pertanyaan independen yang mencakup semua konteks relevan
dari percakapan sebelumnya.

<Chat History>
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""


context_question_prompt = "Anda adalah petugas tes asesmen holland personality test. Tugas Anda adalah memberikan pertanyaan asesmen berdasarkan pertanyaan yang ada di dokumen sejumlah 6 pertanyaan namun dengan pembahasaan yang FRIENDLY DAN SINGKAT ke pengguna. Jika pengguna kesulitan untuk menjawab, atau jawaban pengguna belum cukup jelas, JANGAN LANJUT ke pertanyaan selanjutnya, berikan penjelasan lebih lanjut mengenai pertanyaan YANG SEDANG DITANYAKAN, kemudian MINTA PENGGUNA UNTUK MENJAWAB KEMBALI. Tanyakan pertanyaan satu per satu, tunggu pengguna menjawab baru tanyakan pertanyaan lainnya. Bantu pengguna untuk menemukan 3 holland personality teratas mereka yang paling sesuai berdasarkan jawaban-jawaban yang mereka berikan dari pertanyaan Anda.\n Ini adalah dokumen yang mungkin relevan terhadap konteks:\n\n {context_str} \n\nInstruksi: Gunakan riwayat obrolan sebelumnya, atau konteks di atas, untuk berinteraksi dan membantu pengguna."
            
st.title("Asesmen RIASEC")
st.write("Saya akan melakukan asesmen kepada anda untuk menentukan Anda masuk ke aspek RIASEC mana saja")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Halo, apakah Anda siap untuk melakukan tes RIASEC?"}
    ]

if "chat_engine" not in st.session_state:

    memory = ChatMemoryBuffer.from_defaults(token_limit=50384)

    st.session_state.chat_engine = CondensePlusContextChatEngine(
    retriever=retriever,
    condense_prompt=condense_question_prompt,
    context_prompt = context_question_prompt,
    memory=memory,
    llm=Settings.llm,
    verbose=True
)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Berpikir..."):
            response_stream = st.session_state.chat_engine.chat(prompt)
            st.markdown(response_stream)

    # Add user message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_stream})