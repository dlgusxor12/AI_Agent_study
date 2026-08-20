import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()


class BMIInput(BaseModel):
    height_cm: float = Field(
        ..., gt=50, lt=250,
        description="키 (cm 단위, 50~250)"
    )
    weight_kg: float = Field(
        ..., gt=10, lt=300,
        description="몸무게 (kg 단위, 10~300)"
    )


@tool("calculate_bmi", args_schema=BMIInput)
def calculate_bmi(height_cm: float, weight_kg: float) -> str:
    """체질량지수(BMI)를 계산해 수치와 해석을 반환합니다.
    
    Args:
        height_cm: 키 (cm 단위, 50~250)
        weight_kg: 몸무게 (kg 단위, 10~300)
    """
    bmi = weight_kg / (height_cm / 100) ** 2
    if bmi < 18.5:
        label = "저체중"
    elif bmi < 23:
        label = "정상"
    elif bmi < 25:
        label = "과체중"
    else:
        label = "비만"
    return f"BMI {bmi:.1f} ({label})"


agent = create_agent(model="openai:gpt-5.4-mini", tools=[calculate_bmi])

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "키 175cm, 몸무게 68kg 내 BMI는?",
    }]
})
print(result["messages"][-1].content)