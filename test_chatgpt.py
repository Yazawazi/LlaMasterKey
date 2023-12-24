import os
os.environ['OPENAI_BASE_URL'] = "http://localhost:8000/openai"
os.environ['OPENAI_API_KEY'] = "WHATEVER_PLACEHOLDER"


from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is FastAPI?"}
  ]
)

print(completion.choices[0].message.content)
