import requests
import json
import os
from global_parameters import download_parameters
from concurrent.futures import ThreadPoolExecutor

headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}

def download_extract(language):
    download_path = os.path.join(os.getcwd(), download_parameters['download_path'])
    if not os.path.isdir(download_path):
        os.mkdir(download_path)
    print(download_path)
    language=language.strip()
    json_response = requests.get(download_parameters['info_url']+language,headers=headers).json()
    #print(json_response)
    if len(json_response) == 0:
        print("Invalid language code >> {} <<. Skipping.".format(language))
        return

    for details in json_response:
        if  details['name'] == download_parameters['version']:
            print("Faunded {} for language >> {} <<".format(download_parameters['version'], language))
            print("Valid clips duration: {:.02f} hours.".format(details['valid_clips_duration']/(3600 * 1000)))
            print("Size In GB: ", details['size']/1024/1024)
            print("Voices: ", details['total_users'])

            file_path = details['download_path'].replace("{locale}", language)
            filename=file_path.split("/")[1]
            file_path=file_path.replace("/","%2F")
            generatedurl = requests.get(download_parameters['download_url']+file_path,headers=headers).json()
            doc = requests.get(generatedurl['url'],headers=headers)
            with open(os.path.join(download_path, filename), 'wb') as f1:
                f1.write(doc.content)
    print("Completed for language >> {} <<".format(language))

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=len(download_parameters['languages'])) as exe:
        futures = exe.map(download_extract, download_parameters['languages'])
