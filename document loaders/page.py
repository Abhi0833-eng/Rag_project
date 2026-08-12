import requests
from bs4 import BeautifulSoup


def fetch_page_text(url: str) -> str:
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return text


def main():
    url = "https://example.com"
    text = fetch_page_text(url)

    print(f"URL: {url}")
    print(f"Total characters: {len(text)}")
    print(f"Total words: {len(text.split())}")
    print("\n--- Page Content ---\n")
    print(text[:4000])


if __name__ == "__main__":
    main()
