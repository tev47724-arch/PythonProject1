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
        if "markdown" in result:
            text += result["markdown"]
        elif "content" in result:
            text += result["content"]
        elif "data" in result:
            data = result["data"]

            if isinstance(data, list):
                for page in data:
                    url = page.get("metadata", {}).get("sourceURL", "Unknown source")
                    markdown = page.get("markdown", "")
                    text += f"\n\nSOURCE: {url}\n{markdown}"

            elif isinstance(data, dict):
                text += data.get("markdown", "")

    elif isinstance(result, list):
        for page in result:
            if isinstance(page, dict):
                url = page.get("metadata", {}).get("sourceURL", "Unknown source")
                markdown = page.get("markdown", "")
                text += f"\n\nSOURCE: {url}\n{markdown}"

    return text


@st.cache_data(ttl=3600)
def crawl_psu_site():
    firecrawl_key = get_secret("FIRECRAWL_API_KEY")

    app = FirecrawlApp(api_key=firecrawl_key)

    result = app.crawl_url(
        "https://harrisburg.psu.edu/",
        params={
            "limit": 20,
            "scrapeOptions": {
                "formats": ["markdown"]
            }
        }
    )

    website_text = extract_firecrawl_text(result)

    return website_text[:60000]


question = st.chat_input("Ask something about PSU Harrisburg")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching PSU Harrisburg website..."):

            openrouter_key = get_secret("OPENROUTER_API_KEY")
            base_url = get_secret("OPENROUTER_BASE_URL")
            firecrawl_key = get_secret("FIRECRAWL_API_KEY")

            if not openrouter_key:
                st.error("Missing OPENROUTER_API_KEY.")

            elif not firecrawl_key:
                st.error("Missing FIRECRAWL_API_KEY.")

            else:
                website_info = crawl_psu_site()

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

Only answer questions related to:
- Penn State Harrisburg
- academics
- admissions
- financial aid
- housing
- dining
- student life
- campus resources
- majors and programs

If the question is clearly unrelated, say:
"I can only answer questions related to Penn State Harrisburg."

Use only the website information below.

Answer in simple bullet points.

Include the source URL if available.

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
                    st.error("OpenRouter error. Check your API key, credits, or model name.")
                    st.write(e)