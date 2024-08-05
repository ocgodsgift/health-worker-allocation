import requests
import pandas as pd

# Define DHIS2 API endpoint and credentials
dhis2_url = 'https://edodigital.org.ng/api'
username = 'ahmad'
password = 'Ahmad@2024'

# Authenticate with DHIS2 API
auth = (username, password)

def dataset():
    # Function to fetch data from DHIS2 API
    def fetch_data(endpoint, params=None):
        try:
            response = requests.get(f'{dhis2_url}/{endpoint}', params=params, auth=auth)
            response.raise_for_status()  # Raise exception for non-200 status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f'Error fetching data from {endpoint}: {e}')
            return None
        except ValueError as e:
            print(f'Error decoding JSON response from {endpoint}: {e}')
            return None

    # Function to fetch all paginated data
    def fetch_all_data(program_id, start_date):
        endpoint = 'events'
        params = {
            'program': program_id,
            'startDate': start_date,
            'pageSize': 50,  # Set the page size to a reasonable number
            'page': 1        # Start with the first page
        }
        all_events = []
        
        while True:
            data_sets = fetch_data(endpoint, params=params)
            if data_sets and 'events' in data_sets:
                all_events.extend(data_sets['events'])
                # Check if there are more pages to fetch
                if data_sets.get('pager', {}).get('isLastPage', True):
                    break
                else:
                    params['page'] += 1
            else:
                break
        return all_events

    # Fetching all datasets
    program_id = 'F8tpL8M7fq8'
    start_date = '2024-02-01T00:00:00.000'
    all_events = fetch_all_data(program_id, start_date)

    data_values_list = []
    for event in all_events:
        data_values_list.extend(event['dataValues'])

    # Normalize nested dictionaries
    df = pd.json_normalize(data_values_list)
    df = df[['lastUpdated', 'dataElement', 'value']]
    # Pivot the table to reformat it
    df1 = df.pivot_table(index='lastUpdated', columns='dataElement', values='value', aggfunc='first').reset_index()
    # Dropping the unnecessary columns before renaming
    df1 = df1[['lastUpdated', 'hxCXXKUENNy', 'FTIj7ddCqeu', 'wGqFLK9GUU3']]
    # Rename columns
    df1.columns = ['date', 'comment', 'headline', 'subheadline']
    # Focusing on Health and Public health
    df2 = df1[df1['headline'].isin(['Health', 'Public Health'])].reset_index(drop=True)

    # Save the output
    df2.to_csv('Health Related Citizen Reports.csv')
    df2['date'] = pd.to_datetime(df2['date']).dt.date

    return df2