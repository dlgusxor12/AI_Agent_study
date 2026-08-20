from langchain_core.tools import tool


@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """도시 이름을 받아 현재 날씨(섭씨)를 반환합니다.

    Args:
        city: 도시 영문 이름 (예: "Seoul", "Tokyo")
    """
    return "dummy"


print("name:", get_weather.name)
print("description:", get_weather.description)
print("args_schema:", get_weather.args_schema.model_json_schema())