#0-----
# http://blog.romanofoti.com/download_from_kaggle/ for when working from remote server


#1----
#https://stackoverflow.com/questions/49386920/download-kaggle-dataset-by-using-python
#not using kaggle package
import requests
data_url = 'https://www.kaggle.com/mohansacharya/graduate-admissions' # The direct link to the Kaggle data set
local_filename = "C:/Users/Mahum/Documents/uni/BigData&Programmin/PYTHON/kc_house_data.csv" # The local path where the data set is saved.

kaggle_info = {'UserName': "mtofiq", 'Password': "m@humtofiq21580"} # Kaggle Username and Password
#{"username":"mtofiq","key":"4221a8c7a14e80b9cc67f2b72e828e78"} m@humtofiq21580

r = requests.get(data_url) # Attempts to download the CSV file. Gets rejected because we are not logged in.
r = requests.post(r.url, data = kaggle_info)# ,prefetch = False) # Login to Kaggle and retrieve the data.

f = open(local_filename, 'w') # Writes the data to a local file one chunk at a time
for chunk in r.iter_content(chunk_size = 512 * 1024): # Reads 512KB at a time into memory
    if chunk: # filter out keep-alive new chunks
        f.write(chunk)
f.close()

#2-----
#https://github.com/EKami/kaggle-data-downloader #pip install -U kaggle_data
from kaggle_data.downloader import KaggleDataDownloader
destination_path = "C:/Users/Mahum/Documents/uni/BigData&Programmin/PYTHON/"
downloader = KaggleDataDownloader("yourusername", "yourpassword", "housesalesprediction")
output_path = downloader.download_dataset("housesalesprediction.zip", destination_path)
downloader.decompress(output_path, destination_path)
downloader.decompress(destination_path + "housesalesprediction.zip", destination_path)

#3-----
#https://stackoverflow.com/questions/49386920/download-kaggle-dataset-by-using-python
#https://stackoverflow.com/questions/50863516/issue-in-extracting-titanic-training-data-from-kaggle-using-jupyter-notebook/50876207#50876207
import requests
payload = {
    '__RequestVerificationToken': '',
    'username': 'OMITTED',
    'password': 'OMITTED',
    'rememberme': 'false'}
loginURL = 'https://www.kaggle.com/account/login'
dataURL = "https://www.kaggle.com/c/3136/download/train.csv"
with requests.Session() as c:
    response = c.get(loginURL).text
    AFToken = response[response.index('antiForgeryToken')+19:response.index('isAnonymous: ')-12]
    print("AntiForgeryToken={}".format(AFToken))
    payload['__RequestVerificationToken']=AFToken
    c.post(loginURL + "?isModal=true&returnUrl=/", data=payload)
    response = c.get(dataURL)
    print(response.text)
    download = c.get(dataURL)
    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',') 
    my_list = list(cr)
    for row in my_list:
        print(row)
#df = pd.DataFrame(my_list)
#header = df.iloc[0]
#df = df[1:]
#diab = df.set_axis(header, axis='columns', inplace=False) # diab # to make sure it worked, uncomment this next line:

#4----
from kaggle.api.kaggle_api_extended import KaggleApi #import kaggle
api = KaggleApi()
api.authenticate() #kaggle.api.authenticate()
KaggleApi.dataset_download_file( self, dataset, creditcardfraud, path='https://www.kaggle.com/mlg-ulb/creditcardfraud', force=True, quiet=True)
#kaggle.api.dataset_download_files('housesalesprediction', path='C:/Users/Mahum/Documents/uni/BigData&Programmin/PYSPARK/', unzip=True)

#5----
# download and submit competiton files
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi('copy and paste kaggle.json content here')
api.authenticate()
files = api.competition_download_files("creditcardfraud")

api = KaggleApi()
files = api.competition_download_files("creditcardfraud")
api.competitions_submit("submission.csv", "my submission message", "twosigmanews")

