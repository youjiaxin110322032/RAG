"""RAG 線上問答系統 + LINE Bot 推播 (Streamlit)

執行：streamlit run app.py
流程：上傳文件 -> 切塊並向量化 -> 提問檢索 -> LLM 回答 (500 字內) -> 推播到 LINE
"""
import hashlib
import io

import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

from linebot.v3.messaging import (
    ApiClient,
    BroadcastRequest,
    Configuration,
    MessagingApi,
    PushMessageRequest,
    TextMessage,
)

MAX_CHARS = 500          # 回答與推播的字數上限
CHUNK_SIZE = 500         # 每個文件片段字數
CHUNK_OVERLAP = 80       # 片段重疊字數
EMBED_MODEL = "text-embedding-3-small"

st.set_page_config(page_title="RAG 問答 + LINE 推播", page_icon="💬", layout="wide")


# ---------- 工具函式 ----------
def clip(text: str, limit: int = MAX_CHARS) -> str:
    """硬性截斷到指定字數內。"""
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def read_file(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    for enc in ("utf-8", "utf-8-sig", "big5", "cp950"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def embed(client: OpenAI, texts):
    vecs = []
    for i in range(0, len(texts), 100):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + 100])
        vecs.extend(d.embedding for d in resp.data)
    arr = np.array(vecs, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def retrieve(client: OpenAI, question: str, top_k: int):
    q = embed(client, [question])[0]
    scores = st.session_state.vectors @ q
    idx = np.argsort(scores)[::-1][:top_k]
    return [(st.session_state.chunks[i], float(scores[i])) for i in idx]


def answer(client: OpenAI, model: str, question: str, hits) -> str:
    context = "\n\n".join(f"[片段{i + 1}] {c}" for i, (c, _) in enumerate(hits))
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是文件問答助理。只根據提供的資料回答，資料中沒有就說「資料中找不到相關內容」。"
                    f"使用繁體中文，回答必須控制在 {MAX_CHARS - 50} 字以內，直接給重點。"
                ),
            },
            {"role": "user", "content": f"資料：\n{context}\n\n問題：{question}"},
        ],
    )
    return clip(resp.choices[0].message.content or "")


def send_line(token: str, text: str, user_id: str = ""):
    """user_id 有填 -> 推播給單一使用者；沒填 -> 廣播給所有好友。"""
    config = Configuration(access_token=token)
    with ApiClient(config) as api_client:
        api = MessagingApi(api_client)
        msg = [TextMessage(text=clip(text))]
        if user_id:
            api.push_message(PushMessageRequest(to=user_id, messages=msg))
        else:
            api.broadcast(BroadcastRequest(messages=msg))


# ---------- 側邊欄：金鑰與設定 ----------
with st.sidebar:
    st.header("🔑 金鑰設定")
    st.caption("金鑰只存在目前瀏覽器工作階段，不會寫入檔案。")
    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    line_token = st.text_input("LINE Channel Access Token", type="password")
    line_user = st.text_input(
        "LINE User ID（選填）",
        placeholder="Uxxxxxxxx…；留空＝廣播給所有好友",
    )
    st.header("⚙️ 參數")
    model = st.selectbox("回答模型", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
    top_k = st.slider("檢索片段數", 1, 8, 4)
    auto_send = st.checkbox("回答後自動推播到 LINE", value=False)

# ---------- 主畫面 ----------
st.title("💬 RAG 線上問答系統 × LINE 推播")

for key, default in (("chunks", []), ("vectors", None), ("sig", ""), ("answer", ""), ("hits", [])):
    st.session_state.setdefault(key, default)

st.subheader("1️⃣ 上傳知識文件")
files = st.file_uploader("支援 PDF / TXT / MD（可多選）", type=["pdf", "txt", "md"], accept_multiple_files=True)

if files:
    sig = hashlib.md5(b"".join(f.getvalue() for f in files)).hexdigest()
    if sig != st.session_state.sig:
        if not openai_key:
            st.warning("請先在左側輸入 OpenAI API Key，才能建立向量索引。")
        else:
            with st.spinner("建立索引中…"):
                try:
                    chunks = []
                    for f in files:
                        chunks += split_text(read_file(f))
                    client = OpenAI(api_key=openai_key)
                    st.session_state.vectors = embed(client, chunks)
                    st.session_state.chunks = chunks
                    st.session_state.sig = sig
                    st.success(f"完成：共 {len(chunks)} 個片段")
                except Exception as e:
                    st.error(f"建立索引失敗：{e}")
    else:
        st.info(f"索引已就緒（{len(st.session_state.chunks)} 個片段）")

st.subheader("2️⃣ 提問")
question = st.text_input("輸入你的問題")

if st.button("送出問題", type="primary"):
    if not openai_key:
        st.error("請輸入 OpenAI API Key")
    elif st.session_state.vectors is None:
        st.error("請先上傳文件並建立索引")
    elif not question.strip():
        st.error("請輸入問題")
    else:
        with st.spinner("思考中…"):
            try:
                client = OpenAI(api_key=openai_key)
                hits = retrieve(client, question, top_k)
                st.session_state.hits = hits
                st.session_state.answer = answer(client, model, question, hits)
            except Exception as e:
                st.error(f"問答失敗：{e}")
                st.session_state.answer = ""
        if auto_send and st.session_state.answer:
            if not line_token:
                st.warning("已勾選自動推播，但尚未輸入 LINE Channel Access Token")
            else:
                try:
                    send_line(line_token, st.session_state.answer, line_user.strip())
                    st.success("已自動推播到 LINE ✅")
                except Exception as e:
                    st.error(f"LINE 推播失敗：{e}")

if st.session_state.answer:
    st.subheader("3️⃣ 回答")
    st.write(st.session_state.answer)
    st.caption(f"{len(st.session_state.answer)} / {MAX_CHARS} 字")
    with st.expander("參考片段"):
        for c, s in st.session_state.hits:
            st.markdown(f"**相似度 {s:.3f}**\n\n{c}")
            st.divider()

    if st.button("📤 推播到 LINE"):
        if not line_token:
            st.error("請在左側輸入 LINE Channel Access Token")
        else:
            try:
                send_line(line_token, st.session_state.answer, line_user.strip())
                st.success("已推播到 LINE ✅")
            except Exception as e:
                st.error(f"LINE 推播失敗：{e}")
