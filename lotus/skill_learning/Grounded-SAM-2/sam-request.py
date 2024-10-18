import requests

# The URL of your FastAPI server (adjust to your local or deployed server)
url = "http://127.0.0.1:8000/get_embeddings"

# The image file you want to upload
file_path = "/home/davin123/SAM2-Request/truck.jpg"

# Open the image in binary mode and send it in the request
with open(file_path, "rb") as image_file:
    files = {"img": image_file}
    data = {"prompt": "Truck. Tire."}
    response = requests.post(url, files=files, data=data)

# Print the response from the server
# print(response.json())
