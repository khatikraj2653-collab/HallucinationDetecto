from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def build_vectorstore(context):
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split_text(context)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def check_claim_grounding(vectorstore, claim, threshold=0.75):
    results = vectorstore.similarity_search_with_relevance_scores(claim, k=1)
    if not results:
        return False, 0.0, ""
    best_chunk, score = results[0]
    is_grounded = score >= threshold
    return is_grounded, round(score, 3), best_chunk.page_content