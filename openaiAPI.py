
import os
import openai
openai.api_key = "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"

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "꽃이란 무엇일까까"}
  ]
)
print(completion)
print(completion.choices[0].message.content)


import os
import openai

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="서울근교 주말 데이트코스 추천해줘줘",
  max_tokens=2048,
  temperature=0
)

print(response)
print(response.choices[0].text)





