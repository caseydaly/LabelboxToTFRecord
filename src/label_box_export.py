from labelbox import Client
import time
import requests
import json
import shutil
from graphqlclient import GraphQLClient
from requests import Session
import urllib


# TODO read from gitignorable config file
API_KEY = ''
PROJECT_ID = 'ckcgqorltvxoi08974xshx1wi'

POLL_SEC = 5
LABEL_BOX_URL = 'https://api.labelbox.com/graphql'

# Make a single request for a label export, may result in "shouldPoll==true"
# until the file is ready.
def request_data_gql():
    client = GraphQLClient(LABEL_BOX_URL)
    client.inject_token('Bearer ' + API_KEY)
    res_str = client.execute(f"""
        mutation{{ 
            exportLabels(data:{{ 
                projectId:"{PROJECT_ID}"
            }}){{ 
                downloadUrl 
                createdAt 
                shouldPoll 
            }}
        }}
    """)
    # TODO detect failure here?
    res = json.loads(res_str)['data']
    return res['exportLabels']

# Initiate label export and block until its completion
def download_gql():
    # Sample object structure:
    """
    {
      "data": {
        "exportLabels": {
          "downloadUrl": "https://storage.googleapis.com/labelbox-exports/blahblah",
          "createdAt": "2020-07-23T19:31:24.000Z",
          "shouldPoll": false
        }
      }
    }
    """
    while (export := request_data_gql()) and export['shouldPoll']:
        print(export)
        time.sleep(POLL_SEC)

    filename = 'export-' + export['createdAt'].replace(':', ' ') + '.json'
    download_file( export['downloadUrl'], filename)

def download_file(url, filename):
    # Taken from https://stackoverflow.com/a/39217788/356887
    # and https://github.com/psf/requests/issues/2155
    with requests.get(url, stream=True) as r:
        r.raw.decode_content = True
        with open(filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)


#TODO pass in a destination directory
def main():
    download_gql()


if __name__ == '__main__':
    main()