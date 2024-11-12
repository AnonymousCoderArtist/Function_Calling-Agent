# Agents: The Final Frontier of Tools
# The most common use case for tools and function calling is for creating agentic systems.
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

# Get the prompt
oaif_prompt = hub.pull("hwchase17/openai-functions-agent")

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.7)
tools = [web_search, get_exchange_rate]

# Create the agent
oaif_agent = create_tool_calling_agent(llm, tools, oaif_prompt)

# Create the Agent Executor
# This is the runtime for an agent. This is what actually calls the agent, executes the actions it chooses, passes the action outputs back to the agent, and repeats. I
oaif_agent_executor = AgentExecutor(agent=oaif_agent, tools=tools, verbose=True)

query = "When was webscout released? and whats the feature it has?"

response = oaif_agent_executor.invoke({"input": query})

print("\n", response['output'][0]['text'])
