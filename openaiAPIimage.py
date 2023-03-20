import openai
openai.api_key = "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"

from urllib.request import urlopen
from PIL import Image

response = openai.Image.create(
  prompt="2 soccer players in  miyazaki style",
  n=1,
  size="1024x1024"
)
image_url = response.data[0].url


img = Image.open(urlopen(image_url))
img.show()