# MetaQA source

The raw files in this directory come from the official MetaQA release:

- Repository: https://github.com/yuyuz/MetaQA
- Official data folder: https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE
- License: Creative Commons Attribution 3.0, included as `LICENSE.txt`

Only the movie knowledge base and Vanilla text data are needed for this project:

```text
kb/kb.txt
1-hop/vanilla/qa_train.txt
1-hop/vanilla/qa_dev.txt
1-hop/vanilla/qa_test.txt
2-hop/vanilla/qa_train.txt
2-hop/vanilla/qa_dev.txt
2-hop/vanilla/qa_test.txt
3-hop/vanilla/qa_train.txt
3-hop/vanilla/qa_dev.txt
3-hop/vanilla/qa_test.txt
```

The audio and NTM paraphrased variants are intentionally excluded.

From the repository root, prepare the data with:

```powershell
backend\venv311\Scripts\python.exe scripts\data\prepare_metaqa.py
backend\venv311\Scripts\python.exe scripts\data\create_midterm_benchmark.py
```
