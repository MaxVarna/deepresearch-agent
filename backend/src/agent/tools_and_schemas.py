from typing import List
from pydantic import BaseModel, Field, ConfigDict

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.agents.agent_toolkits import Tool
from langchain.tools.render import render_text_description
from langchain.agents.tools import tool

# ---- LLM и prompt ----

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

final_answer_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        template="""
Ты — прагматичный маркетолог Макс. Твоя задача — дать исчерпывающий и структурированный ответ на вопрос пользователя,
основываясь на предоставленном контексте и результатах веб-поиска.

ИСХОДНЫЕ ДАННЫЕ ИССЛЕДОВАНИЯ:
{research_context}

РЕЗУЛЬТАТЫ ВЕБ-ПОИСКА:
{search_results}

Сформируй ответ и обязательно укажи источники.
        """
    ),
    HumanMessagePromptTemplate.from_template("Вопрос: {question}"),
])

# ---- Pydantic-модель ----

class FinalAnswerModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    answer: str = Field(..., description="Исчерпывающий ответ на вопрос пользователя.")
    sources: List[str] = Field(..., description="Список URL-адресов использованных источников.")

# ---- Цепочка, основанная на prompt и модели ----

final_answer_chain: Runnable = final_answer_prompt | llm.with_structured_output(FinalAnswerModel)

# ---- Веб-поиск ----

search_wrapper = GoogleSearchAPIWrapper()
search_google_and_scrape = Tool.from_function(
    name="google_search_and_scrape",
    func=search_wrapper.run,
    description="Поиск информации в интернете по заданному запросу.",
)

# ---- Инструменты ----

all_tools = [search_google_and_scrape]
tool_descriptions = render_text_description(all_tools)
