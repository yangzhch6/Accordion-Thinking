import pandas as pd

path = "/mnt/weka/home/yongxin.wang/workspace/lark/FlameR/data/old/MATH-AIME-Evaluation.parquet"
data = pd.read_parquet(path)
# convert to dict lst
data = data.to_dict(orient='records')
print(data[0])
print(data[400])