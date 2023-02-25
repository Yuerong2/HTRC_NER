import nltk
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import pandas as pd
from pathlib import Path
import os
import glob
import pandas as pd

st = StanfordNERTagger('/Download/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
                       '/Download/stanford-ner-2018-10-16/stanford-ner.jar',
                       encoding='utf-8')

def page_ner(filepath:str,vol_id):
   page_text= open(filepath,errors='replace').read()
   tokenized_text = word_tokenize(page_text)
   classified_text = st.tag(tokenized_text)
   word_lst=[]
   class_lst=[]
   for item in classified_text:
      #print(type(item))#<class 'tuple'>
      aList = list(item)
      word=aList[0]
      cls=aList[-1]
      word_lst.append(word)
      class_lst.append(cls)
   ner_df = pd.DataFrame({'word': word_lst,'ner': class_lst})
   select_ner_df = ner_df.loc[ner_df['ner'] != 'O']
   select_ner_df = select_ner_df.drop_duplicates() #deduplicate
   page_id= filepath.split('/')[-1].rstrip('.txt')
   select_ner_df['vol_id'] = vol_id
   select_ner_df['page_id'] = page_id
   return select_ner_df

def process_workset(workset_path:str):
  df_lst=[]
# iterate over folders in workset directory
  vol_paths = Path(workset_path).glob('*')
  for vol_path in vol_paths:
      print(vol_path)
      # print(type(vol_path)) # <class 'pathlib.PosixPath'>
      vol_path_str=vol_path.as_posix()#convert to str
      vol_id=vol_path_str.split('/')[-1]
      print(vol_id)
      os.chdir(vol_path)
      extension = 'txt'
      all_pages = [i for i in glob.glob('*.{}'.format(extension))]
      for page in all_pages:
        filepath=vol_path_str+'/'+page #fullpath
        #print(filepath)
        sub_df=page_ner(filepath,vol_id)
        df_lst.append(sub_df)
        #print(len(df_lst))
  df_combined=pd.concat(df_lst) 
  return df_combined

  test_df=process_workset('/secure_volume/HTRC_testing_volume/')

  test_df.to_csv('/content/drive/MyDrive/HTRC_DC_NER_stanfordNLP/testing_output2.csv')