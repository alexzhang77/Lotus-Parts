import requests
import os

# The URL of your FastAPI server (adjust to your local or deployed server)
URL = "http://127.0.0.1:8000/get_embeddings"

# The image file you want to upload
print(os.listdir())
FILE_PATH = "truck.jpg"

# Open the image in binary mode and send it in the request
with open(FILE_PATH, "rb") as image_file:
    files = {"img": image_file}
    data = {"prompt": "Tiger."}
    response = requests.post(URL, files=files, data=data)

# Print the response from the server
# print(response.json())
