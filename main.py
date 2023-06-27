import os
import openai
import pandas as pd

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# データの読み込み
df = pd.read_csv('./data/titanic.csv')

if __name__ == '__main__':
    # モデルの作成
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    # エージェントの作成
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        # return_intermediate_steps=True,
        max_iterations=2,
    )

    # エージェントの実行
    agent.run('男女別の生存率を計算し、棒グラフで描いてください。変数sexの値ですが、0は女性で1が男性です。変数survivedの値ですが、0は死亡で1が生存です。')