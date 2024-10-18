import os
from swarm import Swarm, Agent
from dotenv import load_dotenv
load_dotenv(override=True)

client = Swarm()

print(os.environ["OPENAI_API_BASE"])


def get_weather(location) -> str:
    """
    Get weather information for a specified location.

    Parameters:
    location (str): The location to query weather for

    Returns:
    str: A JSON formatted string containing location, temperature and unit
    """

    print("get_weather called with location: ", location)
    return f"{{'location': '{location}', 'temperature': 67, 'unit': 'F'}}"


agent = Agent(
    name="Agent",
    instructions="You are a helpful agent.",
    functions=[get_weather],
)

messages = [{"role": "user", "content": "What's the weather in New York?"}]

response = client.run(agent=agent, messages=messages)
print(response.messages[-1]["content"])
