import os
import base64

import pandas as pd


def encode_url(url):
    base64_url = base64.b64encode(url.encode('utf-8')).decode('utf-8')
    encoded_url = 'u!' + base64_url[:-1].replace('/', '_').replace('+', '_')
    return f'https://api.onedrive.com/v1.0/shares/{encoded_url}/root/content'


if __name__ == '__main__':
    df = pd.read_csv('onedrive_datasets_url.csv')
    print(df)
    
    for _, row in df.iterrows():
        out_folder_path = os.path.join('..', '0_practice_coding', row.out_folder)

        if os.path.exists(os.path.join(out_folder_path, row.dataset_name)):
            continue

        print(f'Downloading ... {row.dataset_name}')
        os.system(f'wget -q {encode_url(row.onedrive_url)}')

        print(f'Unzipping ...')
        os.makedirs('data', exist_ok=True)
        os.system('unzip content -d data')
        os.system(f'rm content')

        print(f'Symlinking ...')
        os.makedirs(out_folder_path, exist_ok=True)
        from_link = os.path.abspath(os.path.join("data", row.dataset_name))
        to_link = os.path.abspath(os.path.join(out_folder_path, row.dataset_name))
        os.system(f'ln -s {from_link} {to_link}')
