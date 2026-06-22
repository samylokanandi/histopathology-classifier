import requests, base64

image_path = "/Users/samylokanandi/Desktop/some_photo.png"

r = requests.post("http://localhost:8000/predict", files={"file": open(image_path, "rb")})

print("Status code:", r.status_code)
print("Response text:", r.text[:500])

data = r.json()
print("Prediction:", data["prediction"])
print("Confidence:", data["confidence"])

with open("heatmap_test.png", "wb") as f:
    f.write(base64.b64decode(data["heatmap_base64"]))