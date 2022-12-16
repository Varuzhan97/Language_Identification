import requests
import json
import os

from global_parameters import download_parameters


headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}


if __name__ == "__main__":
    main_dir = os.getcwd()

    for language in download_parameters['languages']:
        language=language.strip()
        json_response = requests.get(download_parameters['info_url']+language,headers=headers).json()
        #print(json_response)
        if len(json_response) == 0:
            print("Invalid language code >> {} <<. Skipping.".format(language))
            continue

        for details in json_response:
                print(len(json_response), details)
                if  details['name'] == download_parameters['version']:
                    print("Faunded {} for language >> {} <<".format(download_parameters['version'], language))
                    print("Valid clips duration: {} {:.02f}".format(details['valid_clips_duration'], details['valid_clips_duration']/(3600 * 1000)))
                    print("Size: ", details['size'], details['size']/1024/1024)
                    print("Voices: ", details['total_users'])

                    '''
                    download_path = details['download_path'].replace("{locale}", language)
                    filename=download_path.split("/")[1]
                    download_path=download_path.replace("/","%2F")
                    generatedurl = requests.get(download_parameters['download_url']+download_path,headers=headers).json()
                    doc = requests.get(generatedurl['url'],headers=headers)
                    with open('/home/varuzhan/Desktop/Language_Identification/Data'+filename, 'wb') as f1:
                         f1.write(doc.content)

                    '''
        #if 'str' in language:
        #    break
