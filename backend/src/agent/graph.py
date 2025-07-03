import os
from pathlib import Path
from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END

# --- ИЗМЕНЕНИЕ: Заменяем относительный импорт на абсолютный ---
from agent.tools_and_schemas import (
    FinalAnswerModel,
    final_answer_chain,
    all_tools,
    search_google_and_scrape,
)

# --- 1. Определяем состояние нашего графа (State) ---
# Это структура данных, которая будет передаваться между узлами графа.

class AgentState(BaseModel):
    """Определяет состояние нашего агента."""
    # Контекст из наших файлов
    research_context: str = Field(
        description="Полный контекст из всех предоставленных .md файлов для глубокого анализа."
    )
    # Вопрос пользователя
    question: str = Field(
        description="Вопрос или задача от пользователя."
    )
    # Сгенерированные поисковые запросы
    search_queries: List[str] = Field(
        description="Список поисковых запросов для Google."
    )
    # Результаты поиска
    search_results: List[dict] = Field(
        description="Результаты веб-поиска."
    )
    # Мысли агента после анализа результатов
    reflection: str = Field(
        description="Размышления агента о полноте найденной информации."
    )
    # Финальный ответ
    final_answer: FinalAnswerModel = Field(
        description="Финальный, структурированный ответ для пользователя."
    )
    # Счетчик итераций, чтобы не уйти в бесконечный цикл
    revision_number: int = Field(
        description="Номер текущей итерации поиска."
    )
    # Максимальное количество итераций
    max_revisions: int = Field(
        description="Максимальное количество итераций."
    )

    class Config:
        arbitrary_types_allowed = True


# --- 2. Загрузка нашего контекста ---
# Эта функция будет выполняться один раз при старте.

def load_research_context():
    """Загружает и объединяет все .md файлы из папки context_data."""
    context_dir = Path(__file__).parent.parent.parent / "context_data"
    all_texts = []
    if not context_dir.is_dir():
        return "Контекстные файлы не найдены."

    for md_file in context_dir.glob("*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                all_texts.append(f"--- НАЧАЛО ДОКУМЕНТА: {md_file.name} ---\n\n{f.read()}\n\n--- КОНЕЦ ДОКУМЕНТА: {md_file.name} ---")
        except Exception as e:
            print(f"Ошибка при чтении файла {md_file}: {e}")

    return "\n\n".join(all_texts)

# Загружаем контекст при старте приложения
RESEARCH_CONTEXT = load_research_context()

# --- 3. Определяем узлы нашего графа (Nodes) ---
# Каждый узел - это функция, выполняющая определенное действие.

def generate_queries_node(state: AgentState):
    """Узел для генерации поисковых запросов."""
    print("--- ГЕНЕРАЦИЯ ПОИСКОВЫХ ЗАПРОСОВ ---")
    # Формируем промпт для модели, включая наш контекст
    prompt = f"""Ты — прагматичный маркетолог Макс. Твоя задача — помочь мне доработать исследование.

    ИСХОДНЫЕ ДАННЫЕ ДЛЯ АНАЛИЗА:
    {state['research_context']}

    ЗАДАЧА:
    Проанализируй следующий вопрос/задачу и сформулируй 3-5 точных поисковых запросов для Google, чтобы найти внешние доказательства (статистику, исследования, мнения экспертов).

    Вопрос/задача: "{state['question']}"
    """
    queries = generate_search_queries.invoke({"prompt": prompt})
    return {"search_queries": queries.queries}

def research_node(state: AgentState):
    """Узел для выполнения веб-поиска."""
    print("--- ПОИСК В ИНТЕРНЕТЕ ---")
    results = search_google_and_scrape.invoke(
        {"queries": state["search_queries"]}
    )
    return {"search_results": results}

def reflection_node(state: AgentState):
    """Узел для анализа результатов и принятия решения."""
    print("--- АНАЛИЗ РЕЗУЛЬТАТОВ ---")
    reflection = reflect_on_results.invoke(
        {
            "question": state["question"],
            "search_results": state["search_results"],
            "research_context": state["research_context"]
        }
    )
    return {"reflection": reflection.reflection}

def final_answer_node(state: AgentState):
    """Узел для генерации финального ответа."""
    print("--- ГЕНЕРАЦИЯ ФИНАЛЬНОГО ОТВЕТА ---")
    final_answer = final_answer_chain.invoke(
        {
            "question": state.question,
            "search_results": state.search_results,
            "research_context": state.research_context,
        }
    )
    return {"final_answer": final_answer}


# --- 4. Определяем логику переходов между узлами (Edges) ---

def should_continue(state: AgentState):
    """Функция, определяющая, нужно ли продолжать поиск или можно генерировать ответ."""
    if state["revision_number"] > state["max_revisions"]:
        return "end"
    if "нет" in state["reflection"].lower() or "достаточно" in state["reflection"].lower():
        return "end"
    else:
        return "continue"

# --- 5. Собираем граф ---

# Создаем экземпляр графа
builder = StateGraph(AgentState)

# Добавляем узлы
builder.add_node("generate_queries", generate_queries_node)
builder.add_node("research", research_node)
builder.add_node("reflect", reflection_node)
builder.add_node("final_answer", final_answer_node)

# Определяем точку входа
builder.set_entry_point("generate_queries")

# Добавляем связи между узлами
builder.add_edge("generate_queries", "research")
builder.add_edge("research", "reflect")
builder.add_conditional_edges(
    "reflect",
    should_continue,
    {
        "continue": "generate_queries",  # Если нужно, возвращаемся к генерации запросов
        "end": "final_answer",      # Если информации достаточно, генерируем ответ
    },
)
builder.add_edge("final_answer", END)

# Компилируем граф в исполняемый объект
graph = builder.compile()


# --- 6. Функция для запуска графа с начальными данными ---

def run_agent(question: str):
    """Запускает агент с вопросом пользователя."""
    return graph.invoke({
        "question": question,
        "research_context": RESEARCH_CONTEXT,
        "revision_number": 0,
        "max_revisions": 3,
    })

# Добавляем наш граф в приложение LangServe
from langserve import add_routes
from fastapi import FastAPI

app = FastAPI(
  title="DeepResearch Agent Server",
  version="1.0",
  description="Сервер для аналитического агента Макса",
)

# Добавляем эндпоинт для нашего агента
add_routes(
    app,
    run_agent,
    path="/deepresearch",
    input_type=str,
    output_type=dict,
)