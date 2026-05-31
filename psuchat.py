import os
import re
import streamlit as st

from difflib import SequenceMatcher
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

st.set_page_config(page_title="PSU Harrisburg Assistant")
st.title("PSU Harrisburg Website Assistant")


def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


def extract_firecrawl_text(result):
    if isinstance(result, dict):
        if "markdown" in result:
            return result["markdown"]

    return str(result)


def clean_words(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)

    stop_words = {
        "the", "is", "are", "a", "an", "to", "of", "and", "in", "for",
        "what", "how", "does", "do", "me", "about", "at", "on"
    }

    words = text.split()
    return [word for word in words if word not in stop_words]


def keyword_overlap_score(question, text):
    question_words = set(clean_words(question))
    text_words = set(clean_words(text))

    if not question_words:
        return 0

    overlap = question_words.intersection(text_words)
    return round((len(overlap) / len(question_words)) * 100, 2)


def similarity_score(text1, text2):
    return round(SequenceMatcher(None, text1.lower(), text2.lower()).ratio() * 100, 2)


@st.cache_data(ttl=3600)
def scrape_psu_pages():
    firecrawl_key = get_secret("FIRECRAWL_API_KEY")
    app = FirecrawlApp(api_key=firecrawl_key)

    urls = [
        "https://harrisburg.psu.edu/",
        "https://harrisburg.psu.edu/academics",
        "https://harrisburg.psu.edu/admissions",
        "https://harrisburg.psu.edu/student-life",
        "https://harrisburg.psu.edu/registrar",
        "https://liveon.psu.edu/harrisburg"
    ]

    all_text = ""

    for url in urls:
        try:
            result = app.scrape_url(
                url,
                formats=["markdown"]
            )

            page_text = extract_firecrawl_text(result)

            all_text += f"\n\nSOURCE: {url}\n{page_text}"

        except Exception as e:
            all_text += f"\n\nSOURCE: {url}\nERROR: {e}"

    return all_text[:60000]


question = st.chat_input("Ask something about PSU Harrisburg")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching PSU Harrisburg websites..."):

            openrouter_key = get_secret("OPENROUTER_API_KEY")
            base_url = get_secret("OPENROUTER_BASE_URL")
            firecrawl_key = get_secret("FIRECRAWL_API_KEY")

            if not openrouter_key:
                st.error("Missing OPENROUTER_API_KEY.")

            elif not firecrawl_key:
                st.error("Missing FIRECRAWL_API_KEY.")

            else:
                website_info = scrape_psu_pages()

                with st.expander("View website text"):
                    st.write(website_info[:10000])

                llm = ChatOpenAI(
                    api_key=openrouter_key,
                    base_url=base_url,
                    model="google/gemini-2.0-flash-001",
                    temperature=0
                )

                prompt = f"""
You are a helpful assistant for Penn State Harrisburg.

Only answer questions related to Penn State Harrisburg.

Use only the website information below.

Answer in simple bullet points.

Include the source URL when possible.

If the question is clearly unrelated, say:
"I can only answer questions related to Penn State Harrisburg."

Website information:
{website_info}

User question:
{question}
"""

                try:
                    response = llm.invoke([
                        HumanMessage(content=prompt)
                    ])

                    answer = response.content
                    st.write(answer)

                    retrieval_score = keyword_overlap_score(question, website_info)
                    answer_relevance_score = keyword_overlap_score(question, answer)
                    similarity = similarity_score(question, answer)

                    st.subheader("Evaluation Scores")

                    st.write(f"Retrieval relevance: {retrieval_score}%")
                    st.write(f"Answer relevance: {answer_relevance_score}%")
                    st.write(f"Similarity score: {similarity}%")

                except Exception as e:
                    st.error("OpenRouter error.")
                    st.write(e)