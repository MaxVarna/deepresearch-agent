from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="models/chat-bison-001",  # поддерживается API v1beta
    google_api_key="AIzaSyCfmcZ-pRwmtsof3LXJT_SCeTJ6wwZgDvU"
)

response = llm.invoke("Привет! Ты работаешь?")
print(response.content)
