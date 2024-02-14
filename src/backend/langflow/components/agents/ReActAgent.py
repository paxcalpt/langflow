from typing import Callable, List, Union

from langflow import CustomComponent
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from typing import Union, Callable
from langflow.field_typing import BaseLanguageModel


class ReActAgentComponent(CustomComponent):
    display_name = "ReActAgent"
    description = "Create an agent that uses ReAct prompting."
    documentation = "https://python.langchain.com/docs/modules/agents/agent_types/react"

    def build_config(self):
        return {
            "llm": {"display_name": "LLM"},
            "tools": {"display_name": "Tools"},
        }

    def build(
        self,
        llm: BaseLanguageModel,
        tools: List[Tool],
    ) -> Union[AgentExecutor, Callable]:
        
        template = \
        '''Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}'''

        # remove trailing whitespace from each line
        template = "\n".join([line.strip() for line in template.split("\n")])
        prompt = PromptTemplate.from_template(template)
        return create_react_agent(llm=llm, tools=tools, prompt=prompt)
