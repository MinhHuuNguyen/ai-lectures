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
        out_folder_path = os.path.join('..', '0_practice_coding', row.out_folder, 'data')
        out_file_path = os.path.join(out_folder_path, row.dataset_name)

        if os.path.exists(out_file_path):
            continue

        print(f'Downloading ... {row.dataset_name}')
        os.system(f'wget -q {encode_url(row.onedrive_url)}')

        if not os.path.exists(out_folder_path):
            os.makedirs(out_folder_path, exist_ok=True)
        os.system(f'cp content {out_folder_path}')

        print(f'Unzipping ...')
        unzip_cmd_str = f'unzip {os.path.join(out_folder_path, "content")} -d {out_folder_path}'
        os.system(unzip_cmd_str)
        os.system(f'rm content')
        os.system(f'rm {os.path.join(out_folder_path, "content")}')
