import os
from enum import Enum
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
import openai

class Role(Enum):
    CALLER = "caller"
    CALLEE = "callee"

class Get_Question(BaseModel):
    asked_by: Role
    question: str

class Get_Action(BaseModel):
    done_by: Role
    action: str

question_tool = openai.pydantic_function_tool(Get_Question)
action_tool = openai.pydantic_function_tool(Get_Action)

question_tool["function"]["description"] = """To get a question asked by either the caller or the callee."""
question_tool["function"]['parameters']['properties']['question']['description'] = "The question asked by either the caller or the callee."

action_tool["function"]["description"] = """To get an action taken by either the caller or the callee."""
action_tool["function"]['parameters']['properties']['action']['description'] = "The action taken by either the caller or the callee."

def get_openai_completion(prompt: str) -> str:
    """
    Get a completion from OpenAI API with error handling
    
    Args:
        prompt (str): The prompt to send to OpenAI
        
    Returns:
        str: The completion text from OpenAI
        
    Raises:
        Exception: If there is an error connecting to OpenAI API
    """

    system_prompt = """
    Given a conversation between a business AI agent (callee) and a tester AI agent (caller), 
    you should output a list of questions asked by actions taken by the callee and caller.
    First identify who is the callee and who is the caller, then output the questions and actions.
    """
    try:
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="gpt-4o-mini",
            tools=[question_tool, action_tool],
            tool_choice="auto",
            max_tokens=4096
        )

        return chat_completion.choices[0].message.tool_calls
        
    except Exception as e:
        raise Exception(f"Error getting OpenAI completion: {str(e)}")

text = open('examples/transcription_2.txt', 'r').read()
prompt = """The conversation is"""+text
nodes = get_openai_completion(prompt)
for node in nodes:
    print(node.function.arguments)
