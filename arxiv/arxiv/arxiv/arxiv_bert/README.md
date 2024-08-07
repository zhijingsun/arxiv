# arxiv_bert

model training
 ```
python bert.py
```

test set testing
```
python test.py
```

## 每日model test流程

1.仓库[arxiv](https://github.com/zhijingsun/arxiv), 进crawl文件夹，跑`python crawl_title_abstract.py`爬取当天最新论文，输出json文件latest_papers.json

2.latest_papers.json拉到仓库[arxiv_bert](https://github.com/zhijingsun/arxiv_bert)中，跑`python autolabel.py`输出label预测的excel文件

3.人工筛选检测label准确率

4.将筛好的excel输入到training_data文件夹中的excel_json.ipynb，获得json文件new_training_data.json

5.new_training_data.json输入到bert_augmentation.py中，`python bert_augmentation.py`增强训练

6.`python test_augmentation.py`跑最新test
