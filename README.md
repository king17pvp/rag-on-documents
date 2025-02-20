# rag-on-documents

Question Answering system on PDF, Text files using RAGs.

https://github.com/user-attachments/assets/23d90d19-4888-488c-b026-103cd3d36593



## Requirement
To run this project, you must have installed version 3.10 of Python. It is also highly recommended that you create an environment via conda environment
* Firstly, clone the repository's main branch into your desired directory using your git command prompt.

```git clone https://github.com/king17pvp/rag-on-documents.git```

* Secondly, you can access the directory using this command.
```cd rag-on-documents```

* Thirdly, install required libraries via requirement.txt

```conda create -n rag_on_doc_env python=3.10 -y```

```conda activate rag_on_doc_env```

```pip install -r requirements.txt```

* Then, export your Hugging Face API Key so we can use Inference Service by HuggingFace 

REMINDER: Inference APIs provided by HF are limited per account

```export API_KEY=<YOUR_HF_API_KEY>```

* Finally, run the project by 

```chainlit run main.py --host 0.0.0.0 --port 8000```

You can access the demo web through: ```localhost:8000```


