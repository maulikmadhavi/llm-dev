import os
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

tools = [search_tool, wiki_tool, save_tool]

# Structured response model
class ResearchResponseModel(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

parser = PydanticOutputParser(pydantic_object=ResearchResponseModel)

# Initialize chat history
chat_history = []

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Your task is to research topics and provide structured information. Wrap your answer in JSON following the format instructions.\n{format_instructions}"),
    ("human", "{in}"),
    ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions())

# Instantiate the LLM
llm = ChatNVIDIA(
    model="meta/llama-3.3-70b-instruct",
    nvidia_api_key=os.environ["NVIDIA_API_KEY"],
    temperature=0.2,
    top_p=0.7,
    max_tokens=4096,
)

# Create agent and executor
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run with proper input handling and chat history
def run_research(query: str):
    try:
        response = agent_executor.invoke({
            "in": query,
            "chat_history": chat_history
        })
        # Update chat history
        chat_history.extend([
            ("human", query),
            ("assistant", str(response))
        ])
        return response
    except Exception as e:
        print(f"Error during research: {str(e)}")
        return None

# Execute research
result = run_research("GPU")
if result:
    print("Research Results:")
    print(result)
