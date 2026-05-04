from dotenv import load_dotenv
load_dotenv()

import os

from pinecone import Pinecone


def main():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not api_key:
        raise ValueError("PINECONE_API_KEY is missing from .env")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME is missing from .env")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    stats = index.describe_index_stats()
    namespaces = list(stats.get("namespaces", {}).keys())

    if namespaces:
        for namespace in namespaces:
            index.delete(delete_all=True, namespace=namespace)
            print(f"Deleted namespace: {namespace}")
    else:
        index.delete(delete_all=True)
        print("Deleted default namespace")

    print(f"Cleanup complete for index: {index_name}")


if __name__ == "__main__":
    confirm = input("This will delete all vectors from the Pinecone index. Type DELETE to continue: ")
    if confirm == "DELETE":
        main()
    else:
        print("Cancelled.")
