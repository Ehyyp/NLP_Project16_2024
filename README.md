# NLP_Project16_2024
Natural Language Processing and Text Mining 2024, group project 16
Eetu Hyypiö, Sami Karhumaa & Rainer Laaksonen

The program is provided as a jupyter notebook.
The last task is also provided as python source file, since the notebook kernel may crash when executing the task. Should this happen, run task 8 with
    python task8.py

Steps to using the notebook:

1. Fork the repository
2. Install dependencies from requirements.txt
3. Install desired spacy language models. Default is the English medium model, "en_core_web_md". This can be installed with:
    python -m spacy download en_core_web_md
   Other options are the small "en_core_web_sm" and large "en_core_web_lg" models. All three are used in the report.

Make sure that you have included the "wnaffect.py" and "emotion.py" files. These are used in the emotion and sentiment calculations.
The dataset is saved in the "ijcnlp_dailydialog" directory.
All other files are data files that the notebook blocks create from the Dailydialog dataset.