#Name: Trey Gardner
#Class: CS3080
#Worksheet: 12


import requests
import pandas as pd
import time 
def get_btc_price():
    url = 'https://api.coinbase.com/v2/prices/spot?currency=USD'
    try:
        response = requests.get(url)
        response.raise_for_status() # Raises an HTTPError for bad responses
        data = response.json()
        price = float(data['data']['amount'])
        return price
    except (requests.RequestException, ValueError, KeyError) as e:
        raise Exception(f"Error fetching BTC price: {e}")

def depth_chart():
    #Coinbase API URL for order book
    url = "https://api.exchange.coinbase.com/products/BTC-USD/book"

    #Parameters: level=2 for the top 50 bids and asks
    params = {"level": 2}

    #Send GET request
    response = requests.get(url, params=params)

    if response.status_code == 200:
        order_book = response.json()

        #Print bids and ask
        df_bids = pd.DataFrame(order_book['bids'], columns=['price', 'size', 'num-orders'])
        df_asks = pd.DataFrame(order_book['asks'], columns=['price', 'size', 'num-orders'])

        #conver to float
        df_bids['price'] = df_bids['price'].astype(float)
        df_bids['size'] = df_bids['size'].astype(float)
        df_asks['price'] = df_asks['price'].astype(float)
        df_asks['size'] = df_asks['size'].astype(float)

        #wait 1 minute and fetch price using
        print("Fetching BTC price after 1 minute...")
        time.sleep(60)
        price = get_btc_price()
        print(f'BTC price after 1 minute: {price}')

        df_save = pd.DataFrame()
        #Create a list from the means
        data = {
            'Mean Bids Price': df_bids['price'].mean(),
            'Mean Bids Size': df_bids['size'].mean(),
            'Mean Asks Price': df_asks['price'].mean(),
            'Mean Asks Size': df_asks['size'].mean(),
            'Price Aftger 1min': price
        }
        df_save = pd.DataFrame(data, index=[0])
        #append if exists
        try:
            df_existing = pd.read_csv('btc_order_book_summary.csv')
            df_save = pd.concat([df_existing, df_save],
            ignore_index=True)
        except FileNotFoundError:
            pass
        df_save.to_csv('btc_order_book_summary.csv',index=False)
if __name__ == "__main__":
    while True:
        depth_chart()
