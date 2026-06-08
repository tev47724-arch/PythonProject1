import os
import streamlit as st

from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

st.set_page_config(page_title="PSU Harrisburg Assistant")
st.title("PSU Harrisburg RAD Assistant")


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


def decide_topic(question):
    question = question.lower()

    if "housing" in question or "dorm" in question or "live" in question:
        return "Housing"

    elif "dining" in question or "food" in question or "meal" in question:
        return "Dining"

    elif "major" in question or "program" in question or "academic" in question or "degree" in question:
        return "Academics"

    elif "admission" in question or "apply" in question or "application" in question:
        return "Admissions"

    elif "registrar" in question or "calendar" in question or "class" in question or "schedule" in question:
        return "Registrar"

    elif "student life" in question or "club" in question or "organization" in question or "activity" in question:
        return "Student Life"

    elif "financial" in question or "aid" in question or "tuition" in question or "scholarship" in question:
        return "Financial Aid"

    elif "canvas" in question:
        return "Canvas"

    elif "lionpath" in question:
        return "LionPATH"

    else:
        return "General"


def get_urls_for_topic(topic):
    if topic == "Housing":
        return [
            "https://liveon.psu.edu/harrisburg",
            "https://harrisburg.psu.edu/student-life"
        ]

    elif topic == "Dining":
        return [
            "https://liveon.psu.edu/harrisburg",
            "https://liveon.psu.edu/meal-plans"
        ]

    elif topic == "Academics":
        return [
            "https://harrisburg.psu.edu/academics",
            "https://harrisburg.psu.edu/academics/undergraduate-programs",
            "https://harrisburg.psu.edu/academics/graduate-programs"
        ]

    elif topic == "Admissions":
        return [
            "https://harrisburg.psu.edu/admissions",
            "https://harrisburg.psu.edu/admissions/first-year-students-faq",
            "https://harrisburg.psu.edu/admissions/transfer-students"
        ]

    elif topic == "Registrar":
        return [
            "https://harrisburg.psu.edu/registrar",
            "https://harrisburg.psu.edu/academic-calendar"
        ]

    elif topic == "Student Life":
        return [
            "https://harrisburg.psu.edu/student-life",
            "https://harrisburg.psu.edu/student-engagement",
            "https://harrisburg.psu.edu/recreation-and-fitness"
        ]

    elif topic == "Financial Aid":
        return [
            "https://harrisburg.psu.edu/financial-aid",
            "https://tuition.psu.edu/",
            "https://studentaid.psu.edu/"
        ]

    elif topic == "Canvas":
        return [
            "https://canvas.psu.edu/"
        ]

    elif topic == "LionPATH":
        return [
            "https://lionpath.psu.edu/"
        ]

    else:
        return [
            "https://harrisburg.psu.edu/",
            "https://harrisburg.psu.edu/academics",
            "https://harrisburg.psu.edu/admissions",
            "https://harrisburg.psu.edu/student-life",
            "https://harrisburg.psu.edu/registrar",
            "https://liveon.psu.edu/harrisburg"
        ]


@st.cache_data(ttl=3600)
def scrape_urls(urls):
    firecrawl_key = get_secret("FIRECRAWL_API_KEY")
    app = FirecrawlApp(api_key=firecrawl_key)

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
        with st.spinner("Using RAD to choose the right PSU information..."):

            openrouter_key = get_secret("OPENROUTER_API_KEY")
            base_url = get_secret("OPENROUTER_BASE_URL")
            firecrawl_key = get_secret("FIRECRAWL_API_KEY")

            if not openrouter_key:
                st.error("Missing OPENROUTER_API_KEY.")

            elif not firecrawl_key:
                st.error("Missing FIRECRAWL_API_KEY.")

            else:
                topic = decide_topic(question)
                urls = get_urls_for_topic(topic)
                website_info = scrape_urls(urls)

                st.subheader("RAD Decision")
                st.write(f"Detected topic: {topic}")
                st.write("Selected sources:")
                for url in urls:
                    st.write(f"- {url}")

                with st.expander("View website text"):
                    st.write(website_info[:10000])

                llm = ChatOpenAI(
                    api_key=openrouter_key,
                    base_url=base_url,
                    model="google/gemini-2.5-flash",
                    temperature=0
                )

                prompt = f"""
You are a helpful assistant for Penn State Harrisburg.

The system used RAD, Retrieval-Augmented Decisioning, to classify the user's question.

Detected topic:
{topic}

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