from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import tempfile
import os

def qa_agent(openai_api_key, file, memory, question):
    # 1) LLM / Embeddings は新しい引数名・モデル名で
    llm = ChatOpenAI(
        model="gpt-4o-mini",   # 旧 gpt-3.5-turbo は非推奨
        temperature=0,
        api_key=openai_api_key,
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key,
    )

    # 2) 一時ファイルは安全に
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    try:
        # 3) 読み込み→分割（空文字のセパレータは削除）
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n", "。", "!", "?", "、", ","]
        )
        chunks = splitter.split_documents(docs)

        # 4) ベクタストア & Retriever（k を明示）
        db = FAISS.from_documents(chunks, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 4})

        # 5) 会話チェーン（memory はここで渡す）
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,  # 必要なら
        )

        # 6) invoke には question“だけ”渡す
        result = qa.invoke({"question": question})
        return result

    finally:
        # 7) 後始末
        try:
            os.remove(tmp_path)
        except Exception:
            pass
