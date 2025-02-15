# rag-on-documents

Question Answering system on PDF, Text files using RAGs.

## Requirement
To run this project, you must have installed version 3.10 of Python. It is also highly recommended that you create an environment via conda environment
* Firstly, clone the repository's main branch into your desired directory using your git command prompt.
```git clone https://github.com/king17pvp/rag-on-documents.git```

* Secondly, you can access the directory by this command.
```cd rag-on-documents```

* Thirdly, install required libraries via requirement.txt

```conda create -n rag_on_doc_env python=3.10 -y```

```conda activate rag_on_doc_env```

```pip install -r requirements.txt```

* Finally, run the project by 

```chainlit run main.py --host 0.0.0.0 --port 8000```

You can access the demo web through: ```localhost:8000```
