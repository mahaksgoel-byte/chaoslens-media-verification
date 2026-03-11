import requests

files = {
    "train.jsonl": "https://fever.ai/download/fever/train.jsonl",
    "dev.jsonl": "https://fever.ai/download/fever/paper_dev.jsonl"
}

for filename, url in files.items():
    print(f"Downloading {filename}...")
    r = requests.get(url)
    
    with open(filename, "wb") as f:
        f.write(r.content)

print("Download complete.")