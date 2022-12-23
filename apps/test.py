import requests
from bs4 import BeautifulSoup


def fetch_image_urls(query:str, max_links_to_fetch:int, wd:str, sleep_between_interactions:int=1):
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={query}&oq={query}&gs_l=img"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Safari/537.36'
    }
    response = requests.get(search_url.format(q=query), headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    img_tags = soup.find_all('img')
    urls = []
    for img in img_tags:
        img_url = img.attrs.get('src')
        if not img_url:
            continue
        else:
            urls.append(img_url)
    return urls

fetch_image_urls(query="cat",wd="cat",max_links_to_fetch=50)
# import requests
# import pandas as pd
# # URL of the CSV file on GitHub 
# url = "https://raw.githubusercontent.com/pearl-jinju/Korea_stockprice_predict/main/data_loader/thema_data.csv" 

# # Send an HTTP request to the URL 
# response = requests.get(url)
# if response.status_code == 200:
#     data = response.text 
#     df = pd.read_csv(data, encoding='utf-8')
# print(df.head(5))


# import joblib

# final_lgb_model = joblib.load("..\\model\\lgbm_model_0.20_0.20_iter_50001_day_5.pkl") 

# print(final_lgb_model.predict([1.2,1.5,1.5,1.2,2.2])[0] )


    
# df = get_stock_basic_info()
# print(df[df['종목명'].str.contains("YG")])
    
"""
    
arr = np.array(thema_name_list)
df =pd.DataFrame(arr)
df.to_csv('thema_name_list')
print(thema_name_list)
"""



    # font_size = [cmap(norm(i)) for i in df_result['시가총액']]

    #보여줄 주식 수 결정
#     df_result = df_result #.head()

#     sizes = df_result['시가총액']
#     # print(sizes)
#     label = df_result['종목명']

#     #시총 몇%까지 보여줄거야?
#     sizes_rate=0
#     h=0
#     while sizes_rate<0.75:
#         sizes_rate += sizes[h]/sizes.sum()
#         h+=1


#     #레이블 이름 정하기
#     label_pc = df_result['등락률']
#     label_new= [f'{list(label)[q]}\n{list(label_pc)[w]:.2f}%' for q, w in zip(range(len(sizes)), range(len(sizes)))]
#     #컬러맵 결정
#     cmap = matplotlib.cm.bwr  # select cmap
#     norm_cmap = matplotlib.colors.Normalize(vmin=-5, vmax=5)
#     # font_size = [st.norm(i) for i in np.array(df_result['시가총액'].tolist())]

#     colors = [cmap(norm_cmap(i)) for i in df_result['등락률']]

#     plt.axis('off')
#     squarify.plot(sizes, label=label_new[:h], color=colors, text_kwargs={'color': 'black', 'size': 17 , 'weight': 'bold'}, alpha=0.6)


#     print(plt.show())
#     plt.savefig(f'thema_data_to_insert.png')
#     # squarify.plot.savefig(f'{thema_name} {df_result["등락률"].mean():.2f}%.png')

# print(thema_rank)
# thema_rank  상위 10개
# thema_rank  하위 10개
#상위 테마별 사유 등락률 기준

# import time
# # # 반복 호출시 # requests.exceptions.JSONDecodeError: Expecting value: line 1 column 1 (char 0) 에러
# # start = time.process_time()
# # print(get_stock_basic_info(0,market="ALL",detail="BASIC"))
# # end = time.process_time()
# # print( end - start)  # seconds

# start = time.process_time()
# print("aaa")
# end = time.process_time()
# print( end - start)  # seconds


    
    