import os
import requests
import streamlit as st

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from sentence_transformers import SentenceTransformer

load_dotenv()

st.set_page_config(page_title="PSU Harrisburg Assistant")
st.title("PSU Harrisburg RAG Assistant")


def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


PSU_LINKS = [
    "https://harrisburg.psu.edu/",
    "https://harrisburg.psu.edu/academics",
    "https://harrisburg.psu.edu/admissions",
    "https://harrisburg.psu.edu/tuition-and-financial-aid",
    "https://harrisburg.psu.edu/student-life",
    "https://harrisburg.psu.edu/campus-life",
    "https://liveon.psu.edu/harrisburg",
    "https://liveon.psu.edu/meal-plans",
    "https://tuition.psu.edu/",
    "https://harrisburg.psu.edu/registrar"
]


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


def scrape_page(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        return soup.get_text(" ", strip=True)

    except Exception as e:
        return f"ERROR scraping {url}: {e}"


def chunk_text(text, chunk_size=900, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


@st.cache_resource
def build_vector_db():
    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="psu_harrisburg_pages"
    )

    existing = collection.count()

    if existing > 0:
        return collection

    doc_id = 0

    for url in PSU_LINKS:
        text = scrape_page(url)
        chunks = chunk_text(text)

        for chunk in chunks:
            embedding = embedding_model.encode(chunk).tolist()

            collection.add(
                ids=[str(doc_id)],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": url}]
            )

            doc_id += 1

    return collection


def vector_search(question, top_k=5):
    collection = build_vector_db()

    question_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    matched_text = ""

    for i in range(len(results["documents"][0])):
        source = results["metadatas"][0][i]["source"]
        document = results["documents"][0][i]
        distance = results["distances"][0][i]

        similarity = 1 - distance

        matched_text += f"""
SOURCE:
{source}

SIMILARITY SCORE:
{similarity}

TEXT:
{document}

---
"""

    return matched_text, results


question = st.chat_input("Ask something about PSU Harrisburg")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching PSU pages using vector search..."):

            api_key = get_secret("OPENROUTER_API_KEY")
            base_url = get_secret("OPENROUTER_BASE_URL")

            if not api_key:
                st.error("Missing OPENROUTER_API_KEY.")

            else:
                website_info, results = vector_search(question)

                st.subheader("Vector Search Scores")

                for i in range(len(results["documents"][0])):
                    source = results["metadatas"][0][i]["source"]
                    distance = results["distances"][0][i]
                    similarity = 1 - distance

                    st.write(f"Similarity: {similarity:.4f} | Source: {source}")

                with st.expander("View retrieved chunks"):
                    st.write(website_info[:12000])

                llm = ChatOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    model="google/gemini-2.0-flash-001",
                    temperature=0
                )

                prompt = f"""
You are a helpful PSU Harrisburg assistant.

Use ONLY the retrieved website text below.

Rules:
- Answer in simple bullet points.
- Include the source website URL.
- If the retrieved text does not answer the question, say that clearly.
- Do not make up facts.

RETRIEVED WEBSITE TEXT:
{website_info}

QUESTION:
{question}
"""

                try:
                    response = llm.invoke([
                        HumanMessage(content=prompt)
                    ])

                    st.write(response.content)

                except Exception as e:
                    st.error("OpenRouter error.")
                    st.write(e)