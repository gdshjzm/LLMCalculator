from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage,ToolMessage

llm = OllamaFunctions(
    model="llama3.1",
    format="json",
)

@tool
def calculator(a:float,b:float,operation:str):
    """Simple Calculator of 4 basic operations(+,-,*,/)"""
    if operation == '+':
        return a + b
    elif operation == '-':
        return a - b
    elif operation == '*':
        return a * b
    elif operation == '/':
        if b == 0:
            raise ValueError("Division by zero is not allowed")
        return a / b
    else:
        raise ValueError("Invalid operation")

llm_with_tools = llm.bind_tools([calculator])

message = [
    SystemMessage(   
        content = '''
        You are a calculator expert. When human ask about **ONLY** calculations, you need to cal functions,
        '''
     ),
]

while True:
    prompt = input('---You:')
    if prompt == 'exit':
        break
    else:
        message.append(HumanMessage(content =  prompt))
        ai_msg = llm_with_tools.invoke(message)

        if len(ai_msg.tool_calls) == 0:
            resp = ai_msg.content
            print('---AI:',resp)
            message.append(('ai',resp))
        else:
            print('---Tool use: Using calculator...')
            cal_result = calculator.invoke(ai_msg.tool_calls[0]['args'])
            message.append(SystemMessage(content = f'Calculation result is {cal_result},then you need to reply to Human about calculation result.'))
            print('---System: Calculation completed',end = ' ')
            print('Calculation result:',cal_result)
            ai_msg_math = llm_with_tools.invoke(message)
            ai_msg_content = ai_msg_math.content
            print('---AI:',ai_msg_content)
            message.append(ai_msg_math)



# prompt = input('---You:')
# input_msg = ('human',prompt)
# message.append(input_msg)
# ai_msg = llm_with_tools.invoke(message)
# print(ai_msg)
# print('---Tool use: Using calculator...')
# cal_result = calculator.invoke(ai_msg.tool_calls[0]['args'])
# message.append(('tool',f'Tool calcukation result is {cal_result}'))
# print('---System: Calculation completed',end = ' ')
# print('Calculation result:',cal_result)
# ai_msg_math = llm_with_tools.invoke(message)
# ai_msg_content = ai_msg_math.content
# print('---AI:',ai_msg_content)
# message.append('ai',ai_msg_content)
            


# prompt = input('Please input your query:')
# ai_msg = llm_with_tools.invoke(
#     prompt,
# )
# tool_call= ai_msg.tool_calls[0]['args']
# final_result = calculator.invoke(tool_call)
# print('Calculation final result:',final_result)
