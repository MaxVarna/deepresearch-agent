from typing import List, Dict
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate

# Временная заглушка для функции поиска (здесь должна быть ваша реализация)
class SearchWrapper:
    def run(self, query: str) -> str:
        return f"Результат поиска по запросу: {query}"

search_wrapper = SearchWrapper()

# Превращаем функцию в инструмент
@tool
def search_google_and_scrape(query: str) -> str:
    """Поиск информации в интернете по заданному запросу."""
    return search_wrapper.run(query)

# Инструменты
all_tools = [search_google_and_scrape]

# Модель финального ответа
class FinalAnswerModel(BaseModel):
    summary: str = Field(description="Краткий ответ на вопрос пользователя.")
    details: str = Field(description="Развернутое объяснение или аналитика.")
    citations: List[str] = Field(description="Список использованных источников.")

# Основное состояние агента
class AgentState(BaseModel):
    research_context: str = Field(
        description="Полный контекст из всех предоставленных .md файлов для глубокого анализа."
    )
    question: str = Field(
        description="Вопрос или задача от пользователя."
    )
    search_queries: List[str] = Field(
        description="Список поисковых запросов для Google."
    )
    search_results: List[Dict] = Field(
        description="Результаты веб-поиска."
    )
    reflection: str = Field(
        description="Размышления агента о полноте найденной информации."
    )
    final_answer: FinalAnswerModel = Field(
        description="Финальный, структурированный ответ для пользователя."
    )
    revision_number: int = Field(
        description="Номер текущей итерации поиска."
    )
    max_revisions: int = Field(
        description="Максимальное количество итераций."
    )

    class Config:
        arbitrary_types_allowed = True
