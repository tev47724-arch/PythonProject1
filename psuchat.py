import os
import streamlit as st

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
    text = ""

    if isinstance(result, dict):
        if "markdown" in result and result["markdown"]:
            text += result["markdown"]

        if "data" in result:
            data = result["data"]

            if isinstance(data, dict):
                text += data.get("markdown", "")

            elif isinstance(data, list):
                for page in data:
                    if isinstance(page, dict):
                        source = page.get("metadata", {}).get("sourceURL", "Unknown source")
                        markdown = page.get("markdown", "")
                        text += f"\n\nSOURCE: {source}\n{markdown}"

    return text


@st.cache_data(ttl=3600)
def scrape_psu_pages():
    firecrawl_key = get_secret("FIRECRAWL_API_KEY")
    app = FirecrawlApp(api_key=firecrawl_key)

    urls = [
        "https://harrisburg.psu.edu/",
        "https://harrisburg.psu.edu/academics",
        "https://harrisburg.psu.edu/admissions",
        "https://harrisburg.psu.edu/student-life",
        "https://harrisburg.psu.edu/campus-life",
        "https://harrisburg.psu.edu/tuition-and-financial-aid",
        "https://harrisburg.psu.edu/registrar",
        "https://liveon.psu.edu/harrisburg",
        "https://liveon.psu.edu/meal-plans",
    ]

    all_text = ""

    for url in urls:
        try:
            result = app.scrape_url(
                url,
                formats=["markdown"]
            )
            st.write(result)

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

                    st.write(response.content)

                except Exception as e:
                    st.error("OpenRouter error.")
                    st.write(e)