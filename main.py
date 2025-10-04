import streamlit as st
from langchain.memory import ConversationBufferMemory
from utils import qa_agent

# OpenAI 例外は環境差があるため、まずは広めに握る
try:
    from openai import AuthenticationError
except Exception:
    AuthenticationError = Exception  # フォールバック

st.title("PDF Chatbot with LangChain and OpenAI")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("[OpenAI API key](https://platform.openai.com/account)")

if openai_api_key:  # 入力があれば保存（空で上書きしない）
    st.session_state["api_key"] = openai_api_key

# メモリは“チェーンが使える形”で保持
if "memory" not in st.session_state:
    # 返り値に合わせて output_key を "answer" に統一（または省略）
    st.session_state["memory"] = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

file = st.file_uploader("Upload a PDF file", type=["pdf"])
question = st.text_input("Ask a question about the PDF", disabled=not file)

# APIキー未入力の早期案内
if file and question and not openai_api_key:
    st.info("Please enter your OpenAI API key to continue.")

# 実行
if file and question and openai_api_key:
    try:
        with st.spinner("Generating response..."):
            # ★ ここが最重要修正：Memory“本体”を渡す
            response = qa_agent(
                openai_api_key,
                file,
                st.session_state["memory"],  # ← messages ではなく memory
                question,
            )
    except AuthenticationError:
        st.info("Invalid OpenAI API key. Please check and try again.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()

    # 返り値が dict/str のどちらでも耐える
    answer = response.get("answer") if isinstance(response, dict) else str(response)
    st.write("Answer:")
    st.write(answer)

    # 履歴は memory から読み出す or response 側にあればそれを使う
    chat_history = (
        response.get("chat_history")
        if isinstance(response, dict) and "chat_history" in response
        else st.session_state["memory"].chat_memory.messages
    )
    st.session_state["chat_history"] = chat_history

# 履歴表示（奇数長でも安全に）
if "chat_history" in st.session_state and st.session_state["chat_history"]:
    with st.expander("Chat History"):
        msgs = st.session_state["chat_history"]
        # 1件ずつロール毎に描画（偶数ペア前提をやめる）
        for idx, msg in enumerate(msgs):
            role = getattr(msg, "type", getattr(msg, "role", "message"))
            prefix = "🧑‍💻 You" if role in ("human", "user") else "🤖 Assistant"
            st.markdown(f"**{prefix}:**")
            # langchain の Message は .content を持つ
            st.write(getattr(msg, "content", str(msg)))
            if idx < len(msgs) - 1:
                st.divider()
