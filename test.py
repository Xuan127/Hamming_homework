import os
from groq import Groq
from pydantic import BaseModel
from typing import Optional
import openai

class Get_BusinessAgent_Question(BaseModel):
    question: str

class Get_BusinessAgent_Action(BaseModel):
    action: str

question_tool = openai.pydantic_function_tool(Get_BusinessAgent_Question)
action_tool = openai.pydantic_function_tool(Get_BusinessAgent_Action)

question_tool["function"]["description"] = """To get a question asked by the AI business agent."""
question_tool["function"]['parameters']['properties']['question']['description'] = "The question asked by the AI business agent."
action_tool["function"]["description"] = """To get an action taken by the AI business agent."""
action_tool["function"]['parameters']['properties']['action']['description'] = "The action taken by the AI business agent."

def get_groq_completion(prompt: str) -> str:
    """
    Get a completion from Groq API with error handling
    
    Args:
        prompt (str): The prompt to send to Groq
        
    Returns:
        str: The completion text from Groq
        
    Raises:
        Exception: If there is an error connecting to Groq API
    """

    system_prompt = """
    Given a conversation between a business AI agent (callee) and a tester AI agent (caller), 
    you should output a list of questions asked by the callee AI and actions taken by the callee AI.
    """
    try:
        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
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
            model="llama3-groq-8b-8192-tool-use-preview",
            tools=[action_tool],
            tool_choice="auto",
            max_tokens=4096
        )

        return chat_completion.choices[0].message.tool_calls
        
    except Exception as e:
        raise Exception(f"Error getting Groq completion: {str(e)}")

text = open('examples/transcription_1.txt', 'r').read()
prompt = """The conversation is"""+text
nodes = get_groq_completion(prompt)
for node in nodes:
    print(node.function.arguments)
