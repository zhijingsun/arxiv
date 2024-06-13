# arxiv_analysis

虚拟机

```
python3 -m venv ~/.venvs/aienv
source ~/.venvs/aienv/bin/activate
```

arxiv 直接解析线上pdf数据库链接

`python link_extract.py`

arxiv 解析本地pdf数据库链接(需先下载pdf到本地)

`python link_extract_local.py`

前端网页

```
streamlit run web_table.py
```
