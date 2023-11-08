import logging
import os

import functions_framework
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from slack_bolt import App, BoltResponse
from slack_bolt.adapter.flask import SlackRequestHandler

app = App(process_before_response=True)
handler = SlackRequestHandler(app)
logging.basicConfig(level=logging.DEBUG)


def create_connection_string():
    """db接続文字列を生成"""
    instance_connection_name = os.environ["INSTANCE_CONNECTION_NAME"]
    db_name = os.environ["DB_NAME"]
    db_user = os.environ["DB_USER"]
    db_password = os.environ["DB_PASSWORD"]
    socket_path = f"/cloudsql/{instance_connection_name}"
    connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@/{db_name}?host={socket_path}"
    return connection_string


@app.middleware
def handle_retry(req, logger, next):
    """
    slack-botは3s以内にレスポンスを返す必要があり、それができない場合には再度呼び出される（retry）
    理想の対処は、即時レスポンスを返したうえで、非同期にメイン処理のレスポンスを返すこと。
    ここでは簡易対策として、retry requestは200を返してスルーする
    """
    logger.info(req.headers)
    if "x-slack-retry-num" in req.headers and req.headers["x-slack-retry-reason"][0] == "http_timeout":
        return BoltResponse(status=200, body="success")

    next()


@app.event("app_mention")
def respond_thread(body, say, logger):
    """
    BotがメンションされたらQAの結果をスレッドに投稿する
    """
    embeddings = OpenAIEmbeddings()
    event = body["event"]
    question = event["text"]
    thread_ts = event.get("thread_ts", None) or event["ts"]

    vector_store = PGVector(
        connection_string=create_connection_string(),
        embedding_function=embeddings,
        collection_name="pages",
    )
    qna = RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(temperature=0.0, model="gpt-4"),
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )

    res = qna(
        {
            "question": question + "\n\nanswer in Japanese.",
        },
        return_only_outputs=True,
    )

    say(text=res["answer"], thread_ts=thread_ts)


@functions_framework.http
def main(request):
    return handler.handle(request)
