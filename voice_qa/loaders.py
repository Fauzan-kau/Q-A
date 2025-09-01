from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
import requests


class MultiUrlLoader:
    def __init__(self, urls):
        self.urls = urls
        self.loaders = []

        # Validate URLs first
        for url in urls:
            try:
                # Basic URL validation
                response = requests.head(url, timeout=10, allow_redirects=True)
                if response.status_code == 200:
                    self.loaders.append(RecursiveUrlLoader(
                        url,
                        max_depth=1,
                        continue_on_failure=True,
                        timeout=30
                    ))
                else:
                    print(f"URL {url} returned status code {response.status_code}")
            except Exception as e:
                print(f"Error validating URL {url}: {e}")

    def load(self):
        documents = []

        if not self.loaders:
            print("No valid URLs to load from")
            return documents

        for loader in self.loaders:
            try:
                print(f"Loading from {loader.url}...")
                docs = loader.load()
                if docs:
                    print(f"Loaded {len(docs)} documents from {loader.url}")
                    documents.extend(docs)
                else:
                    print(f"No documents loaded from {loader.url}")
            except Exception as e:
                print(f"Error loading from {loader.url}: {e}")

        print(f"Total documents loaded: {len(documents)}")
        return documents
