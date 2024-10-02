import base64
import json
import requests

url = "https://e6834e4d-ce99-4122-9c90-4fdec4901058-00-3rnvmws4ksvni.riker.replit.dev/generate-tile"
data = {
"prompt": "Skatepark",
"promptStrength": 0.7
}

response = requests.post(url, json=data)
result = response.json()
print(result)

# Save generated image
with open("generated_image.png", "wb") as f:
    f.write(base64.b64decode(result["generatedImage"]))

# Save background-removed image
with open("bg_removed_image.png", "wb") as f:
    f.write(base64.b64decode(result["bgRemovedImage"]))

print("Images saved successfully!")