
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

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


openai.api_key = os.getenv("OPENAI_API_KEY")

from urllib.request import urlopen
from PIL import Image

response = openai.Image.create(
  prompt="juggling players in  miyazaki style",
  n=1,
  size="1024x1024"
)
image_url = response.data[0].url


img = Image.open(urlopen(image_url))
img.show()


