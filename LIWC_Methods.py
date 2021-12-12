import os, sys, re, time, gc, csv, math
read_loc = 'dataset/Raw/Read/'
store_loc = 'dataset/Raw/'

import pandas as pd
import numpy as np

import requests
from urllib.request import urlsplit

from newspaper import Article
from bs4 import BeautifulSoup
from bs4.element import Comment

# # natural language tool kit
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle


class Custom_Methods(object):

    def __init__(self):
        self.x = 0
        
    @staticmethod
    # Get URL base:
    def get_url_base(df,col,result):
        for idx, url in enumerate(df[col]):
            try:
                split_url = urlsplit(url)
                # Check for this being an archived site
                if 'archive.org' in split_url.netloc:
                    split_url = urlsplit(split_url.path[split_url.path.find('http'):])
                # Now check for trailing ":"
                if ':' in split_url.netloc:
                    df[result][idx] = split_url.netloc[:split_url.netloc.find(':')]
                elif len(split_url.netloc) == 0:
                    df[result][idx] = split_url.path[:split_url.path.find('/')]
                else:
                    # df[result][idx] = split_url.netloc[:split_url.netloc.find('.')+4]
                    df[result][idx] = split_url.netloc
                
                # Now check for www.
                url = df[result][idx]
                if 'ww' in url[:url.find('.')]:
                    df[result][idx] = url[url.find('.')+1:]               
                
            
            except:
                df[result][idx] = 'Unknown'
        return df[result]
    
    @staticmethod
    # Stem words
    def stemming(text):
        port_stem = PorterStemmer()
        text = text.lower()
        text = re.sub("'",'',text) 
        text = re.sub('[^a-z]',' ',text) # ^ is except
        text = text.split()
        text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]
        text = ' '.join(text)
        return text

    @staticmethod
    # Non-Stem words
    def non_stem(text):
        text = text.lower()
        text = re.sub('[^a-z ]',' ',text) # ^ is except
        text = text.split()
        text = ' '.join(text)
        return text
    
    
    @staticmethod
    def clean_text(row):
        body_text = row.body.lower()
        body_text = re.sub(r'\s{2,}',' ',body_text)
        if ') -' in body_text[:50]:
            body_text = body_text[body_text.find(') -')+3:]
        title_text = row.title.lower()
        title_text = re.sub(r'\s{2,}',' ',title_text)
        row.body_url_min = body_text.lower()
        row.body_url = Custom_Methods.non_stem(body_text)
        row.body_stem = Custom_Methods.stemming(body_text)
        row.title_url_min = title_text.lower()
        row.title_url = Custom_Methods.non_stem(title_text)
        row.title_stem = Custom_Methods.stemming(title_text)
        return row
   
    @staticmethod
    def get_df(loc,max_n,file,isPass=True):
        df = pd.read_csv(loc)
        if isPass:
            df = df[df.process_state == 'Pass']
        if 'class' not in df.columns:
            df['class'] = 1    
            if 'True' in file:
                df['class'] = 0  
        df.drop(columns=['link_cnt'],inplace=True)
        df.dropna(subset=['title_url','body_url'],inplace=True)
        min_max_n = min(len(df),max_n)
        df = df.sample(n = min_max_n,random_state=42)
        df.reset_index(drop=True,inplace=True)
        id_list = list(df.id)
        return df,id_list

 
    @staticmethod
    def cnt_links(soup):
        links = []
        for link in soup.find_all('a'):
            try:
                if len(link.get('href')) > 1:
                    links.append(link.get('href'))
            except:
                pass
        return len(links)

    
    @staticmethod
    def scrap_page(row):
        url =  re.sub(r'\r\n', '', row['news_url'])
        if url[:4] != 'http':
            url = f'https://{url}'
        webpage_req = requests.get(url)
        soup = BeautifulSoup(webpage_req.text, 'html.parser')
        row.link_cnt = Custom_Methods.cnt_links(soup)
        # try:
        #For different language newspaper refer above table
        webpage = Article(url, language="en") # en for English
        #To download the article
        webpage.download()
        #To parse the article
        try:
            webpage.parse()
            #To perform natural language processing ie..nlp
            webpage.nlp()
            #To extract title
            title_text = webpage.title
            #To extract body text
            body_text = webpage.text
            if ') -' in body_text[:50]:
                body_text = body_text[body_text.find(') -')+3:]
            # Remove extra spaces
            # body_text = re.sub(r'[^a-z]',' ',body_text)
            body_text = re.sub(r'\s{2,}',' ',body_text)
            if len(body_text) < 200:
                body_text, title_text = Custom_Methods.alt_scrap_page(soup)
        except:
        # Or try altern
            body_text, title_text = Custom_Methods.alt_scrap_page(soup)
        if 'ad-blocker' in body_text[:50]:
            title_text = 'Scrapper blocked by website'
            body_text = 'Scrapper blocked by website'
        if ') -' in body_text[:50]:
            body_text = body_text[body_text.find(') -')+3:]
        row.body_url_min = body_text.lower()
        row.body_url = Custom_Methods.non_stem(body_text)
        row.body_stem = Custom_Methods.stemming(body_text)
        row.title_url_min = title_text.lower()
        row.title_url = Custom_Methods.non_stem(title_text)
        row.title_stem = Custom_Methods.stemming(title_text)

        return row


    @staticmethod
    def alt_scrap_page(soup):
        # print('Alt scrap',end=',')
        tags = [tag.name for tag in soup.find_all()]
        tag_list = list(set(tags))
        tag_word_cnt = []
        for t in tag_list:
            # print(t,end=',')
            body_text = [body.get_text(" ") for body in soup.find_all(t,limit=1)][0].replace('\n',' ').strip().lower()
            tag_word_cnt.append(len(body_text))

        try:
            article_idx = tag_list.index('article')
        except:
            article_idx = 0

        if (article_idx > 0) and (tag_word_cnt[article_idx] > 200):
            body_text = [body.get_text(" ") for body in soup.find_all('article' ,limit=1)][0].replace('\n',' ').replace("\'","'").strip().lower()
            # body_text = re.sub(r'[^a-z]',' ',body_text)
            body_text = re.sub(r'\s{2,}',' ',body_text)
        else: 
            threshold = 0.5
            best_tag = ''
            best_word_cnt = max(tag_word_cnt)
            max_word_cnt = max(tag_word_cnt)
            idx = np.argsort(tag_word_cnt)[::-1]
            for i in idx:
                if (tag_word_cnt[i] >= max_word_cnt * threshold) and tag_word_cnt[i] <= best_word_cnt:
                    # print('New best tag',end=',')
                    best_word_cnt = tag_word_cnt[i]
                    best_tag = tag_list[i]
                    # print(f'{tag_list[i]},{tag_word_cnt[i]}')
                    body_text = [body.get_text(" ") for body in soup.find_all(best_tag,limit=1)][0].replace('\n',' ').replace("\'","'").strip().lower()
                    body_text = re.sub(r'\s{2,}',' ',body_text)

        title_text = [title.get_text() for title in soup.find_all('title',limit=1)][0].strip().lower()
        try: 
            t_end_1 = title_text.index('-')
        except:
            t_end_1 = 0
        try:
            t_end_2 = title_text.index('|')
        except:
            t_end_2 = 0
        t_end = max(t_end_1,t_end_2)
        title_text = title_text[:t_end].strip()

        return body_text,title_text
    
    
    @staticmethod
    def mySortFunc(val):
        return int("".join([(i) for i in re.sub("[^0-9]","",val)]))
    
    @staticmethod
    def normalize(df):
        result = df.copy()
        for col in df.columns:
            max_ = df[col].max()
            min_ = df[col].min()
            result[col] = (df[col] - min_) / (max_ - min_)
        return result

    @staticmethod
    def save_LIWC_csv(save_loc,idx,df,part,sufix,id_):
        file_name_save = f'{save_loc}{sufix}_{part}/{id_}.txt'
        # print(file_name_save)
        # print('---------------')
        text_file = open(file_name_save, "w")
        temp_str = df[f'{part}_url_min'][idx].strip()
        try:
            temp_str = re.sub(r'[^A-Za-z0-9 " , : ! # $ % & ? : ; . - + ^ & * ( \\) \\[ \\] \' \\ _ = > / | { \} ]+', ' ', temp_str)
            temp_str = re.sub(r' {2,}',' ',temp_str)
            text_file.write(temp_str)
            text_file.close()
        except:
            temp_str = re.sub(r'[^A-Za-z0-9 " , : ! # $ % & ? : ; . - + ^ & * ( \[ \] \'  _ = > / | { \} ]+', ' ', temp_str)
            temp_str = re.sub(r' {2,}',' ',temp_str)
            text_file.write(temp_str)
            text_file.close()


    @staticmethod
    def save_dict(loc,dict_):
        csv_columns = ['Key','Value']
        with open(loc, 'w') as f:
            for key in dict_.keys():
                try:
                    f.write("%s,%s\n"%(key,dict_[key]))
                except:
                    pass
                
    @staticmethod
    def save_file(idx, df,read_file):
        # print(f'Save file loc as: {store_loc}{read_file}.csv')
        if idx == 0:
            df.to_csv(f'{store_loc}{read_file}.csv',mode='w', header=True, index=False)
        else:
            df.to_csv(f'{store_loc}{read_file}.csv',mode='a', header=False, index=False)    

            
    
    @staticmethod 
    def build_IDF(save_loc,file_loc_list, main_dict_name):
        cnt = len(file_loc_list)
        print(f'Total records: {cnt}')
        cycle_len = max(500, int(round(cnt/10000,0)))
        main_dict = pd.read_csv(f'{main_dict_name}.csv', header=None, dtype= {0:str}).set_index(0).squeeze().to_dict()
        idfDict = dict.fromkeys(main_dict.keys(), 0)

        # Doc Freq
        for num, file in enumerate(file_loc_list):
            loc = f'{save_loc}TF_dicts/{file}'  
            if num%cycle_len == 0:
                print(f'{num}: {loc}')
            try:
                try:
                    dict_ = pd.read_csv(loc, header=None, dtype= {0:str}).set_index(0).squeeze().to_dict()
                except:
                    print('First exception: ', loc,end=',')
                    temp = pd.read_csv(loc, header=None, dtype= {0:str})
                    dict_ = {temp[0].values[0]:temp[1].values[0]} 
                for word, val in dict_.items():
                    if val > 0:
                        idfDict[word] += 1

                # Inverse Doc Freq 
                for word, val in idfDict.items():
                    try:
                        idfDict[word] = round(math.log(cnt / float(val)),10)
                    except:
                        idfDict[word] = 0
            except:
                print('Full exception: ', loc)
        return idfDict
            

    @staticmethod 
    def build_TF(dict_,Base_dict):
        tf  = dict.fromkeys(Base_dict.keys(), 0)
        for key, val in dict_.items():
            if key in list(Base_dict.keys()):
                tf[key] = val
        tf = list(tf.values())
        return tf
    
    @staticmethod 
    def build_TFIDF(dict_, idfs):
        tfidf  = dict.fromkeys(idfs.keys(), 0)
        for key, val in dict_.items():
            if key in list(idfs.keys()):
                tfidf[key] = val * idfs[key]
        tfidf = list(tfidf.values())
        return tfidf
     
    @staticmethod
    def buildRow(text, Dict):
        one_ex_vector = dict.fromkeys(Dict.keys(), 0)
        # one_ex_vector = np.zeros(dictSize)
        for word in text.split():
            if word.strip() in Dict.keys():
                one_ex_vector[word] = Dict[word]
        one_ex_vector = list(one_ex_vector.values())
        return one_ex_vector
     
        
        
    @staticmethod
    def print_progress(idx,row,start,last_start):
        now = time.time()
        cycle = round((now - last_start)/60.0,1)
        total_cycle = round((now - start)/60.0,)
        print(f'Total time:{total_cycle}  , Cycle time: {cycle}, RowID:{idx} Attempting URL number:{row.id}, URL: {row.news_url}')
        return now
    
        
    @staticmethod    
    def f_importances(coef, names,top=20):
        imp = coef
        names = [str(l) for l in names]
        imp,names = zip(*sorted(zip(imp,names)))
        zipped = list(zip(imp,names))
        # Using sorted by abs and desc with lambda
        res = sorted(zipped, key = lambda x: abs(x[0]),reverse=True)
        # Unzip results
        imp, names = zip(*res)
        plt.barh(range(len(names[:top])), imp[:top], align='center')
        plt.yticks(range(len(names[:top])), names[:top])
        plt.show()
        return imp,names
    
    @staticmethod
    def degree_range(n): 
        start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
        end = np.linspace(0,180,n+1, endpoint=True)[1::]
        mid_points = start + ((end-start)/2.)
        return np.c_[start, end], mid_points
    
    @staticmethod
    def rot_text(ang): 
        rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
        return rotation
    
    @staticmethod
    def gauge(labels=['LOW','MEDIUM','HIGH','VERY HIGH','EXTREME'], \
          colors='jet_r', arrow=1, ang_ = -75, title='', fname=False): 
       
        N = len(labels)

        if arrow > N: 
            raise Exception("\n\nThe category ({}) is greated than \
            the length\nof the labels ({})".format(arrow, N))


        if isinstance(colors, str):
            cmap = cm.get_cmap(colors, N)
            cmap = cmap(np.arange(N))
            colors = cmap[::-1,:].tolist()
        if isinstance(colors, list): 
            if len(colors) == N:
                colors = colors[::-1]
            else: 
                raise Exception("\n\nnumber of colors {} not equal \
                to number of categories{}\n".format(len(colors), N))

        fig, ax = plt.subplots()
        ang_range, mid_points = Custom_Methods.degree_range(N)
        labels = labels[::-1]

        patches = []
        for ang, c in zip(ang_range, colors): 
            # sectors
            patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
            # arcs
            patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))

        [ax.add_patch(p) for p in patches]

        for mid, lab in zip(mid_points, labels): 

            ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab, \
                horizontalalignment='center', verticalalignment='center', fontsize=14, \
                fontweight='bold', rotation = Custom_Methods.rot_text(mid))

        r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
        ax.add_patch(r)

        ax.text(0, -0.05, title, horizontalalignment='center', \
             verticalalignment='center', fontsize=22, fontweight='bold')

        pos = mid_points[abs(arrow - N)]

        if ang_ < 0:
            ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                         width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
        else:    
            ax.arrow(0, 0, 0.225 * np.cos(np.radians(ang_)), 0.225 * np.sin(np.radians(ang_)), \
                         width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

    #     ax.arrow(0, 0,  0.225 * .5 ,  0.225 * .5 , \
    #                  width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

        ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
        ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

        ax.set_frame_on(False)
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axis('equal')
        plt.tight_layout()
        if fname:
            fig.savefig(fname, dpi=200)
            
    ## ########################################################################################


    ## ########################################################################################

    ## ########################################################################################

    ## ########################################################################################

    ## ########################################################################################

