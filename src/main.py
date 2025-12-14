from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from src.tools import search, get_weather

model = ChatOllama(
    model="qwen3:8b",
    temperature=0.1,
    reasoning=True,
    base_url="http://127.0.0.1:11434/",
)
agent = create_agent(model, tools=[search, get_weather])


# for step in agent.stream(
#     {"messages": [{"role": "user", "content": "Search the weather in San Francisco?"}]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

url = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"

graph = Neo4jGraph(url=url, username=username, password=password)

chain = GraphCypherQAChain.from_llm(
    cypher_llm=model,
    qa_llm=model,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
)

for step in chain.stream(
    {"query": "Name all the US fighter aircraft which has parent 'Air' domain"},
    stream_mode="values",
):
    step["query"][-1].pretty_print()
# chain.invoke(
#     {"query": "Name all the US fighter aircraft which has parent 'Air' domain"}
# )
