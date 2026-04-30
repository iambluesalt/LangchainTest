import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from app.backend.semantic_functions import search

DEFAULT_MODEL = "gemini-2.5-flash-lite"


def chat_with_docs(query: str, k: int = 4, model: str = DEFAULT_MODEL) -> dict:
    results = search(query, k=k)

    if results:
        context_parts = []
        sources = []
        for doc, score in results:
            page = doc.metadata.get("page_label", doc.metadata.get("page", "?"))
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            context_parts.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
            sources.append({
                "score": round(score, 4),
                "page": page,
                "source": source,
                "content": doc.page_content,
            })
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = "No relevant documents found."
        sources = []

    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0.2,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )

    messages = [
        SystemMessage(content=(
            "You are a helpful assistant that answers questions strictly based on the "
            "provided document excerpts. Ground every answer in the context below. "
            "If the context does not contain enough information to answer, say so clearly.\n\n"
            f"Context:\n{context}"
        )),
        HumanMessage(content=query),
    ]

    response = llm.invoke(messages)
    return {"answer": response.content, "sources": sources}
