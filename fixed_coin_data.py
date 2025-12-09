#Name: Trey Gardner
#Worksheet: 13
#Class CS3080

"""Author: Zanyar Zohourian Shahzadi
Date: 2025-12-01
Description: session 13 worksheet code...
"""

import requests
import pandas as pd
import time

def get_coin_price(coin="btc"):
      url = f'https://api.coinbase.com/v2/prices/{coin.upper()}-USD/spot'
      try:
        response = requests.get(url)
        response.raise_for_status() #Raises an HTTPErron for bad response
        data = response.json()
        price = float(data['data']['amount'])
        return price
      except (requests.RequestException, ValueError, KeyError) as e:
        raise Exception(f'Error fetching {coin.upper()} price: {e}')

def depth_chart(coin='btc'):
    #Coinbase API URL for order book
    url = f'https://api.exchange.coinbase.com/products/{coin.upper()}-USD/book'

    #Parameters: level=2 for top 50 bids and asks
    params = {'level': 2}

    #Send GET request
    response = requests.get(url, params=params)

    if response.status_code == 200:
        order_book = response.json()

        #Print bids and askss':
        df_bids = pd.DataFrame(order_book['bids'], columns=['price', 'size', 'num-orders'])
        df_asks = pd.DataFrame(order_book['asks'], columns=['price', 'size', 'num-orders'])
        #convert to float
        df_bids['price'] = df_bids['price'].astype(float)
        df_bids['size'] = df_bids['size'].astype(float)
        df_asks['price'] = df_asks['price'].astype(float)
        df_asks['size'] = df_asks['size'].astype(float)

        #wait 1 min and fetch price...
        print(f'Fetching {coin.upper()} price after 1 minute...')
        time.sleep(60)
        price = get_coin_price(coin)
        print(f'{coin.upper()} Price after 1 minute: {price}')

        #create a list(dictionary?) from the means
        data = {
              'Mean Bids Price': df_bids['price'].mean(),
              'Mean Bids Size': df_bids['size'].mean(),
              'Mean Asks Price': df_asks['price'].mean(),
              'Mean Asks Size': df_asks['size'].mean(),
              'Price After 1min': price
        }#data

        df_save = pd.DataFrame(data, index=[0])
        #append if exists
        try:
              df_existing = pd.read_csv(f'{coin}_order_book_summary.csv')
              df_save = pd.concat([df_existing, df_save], ignore_index=True)
        except FileNotFoundError:
              pass
        df_save.to_csv(f'{coin}_order_book_summary.csv', index=False)

def depth_chart_live(coin='btc'):
      #Coinbase API URL for order book
      url = f'https://api.exchange.coinbase.com/products/{coin.upper()}-USD/book'

      #Parameter: level=2 for top 50 bids and asks
      params = {'level': 2}

      #Send GET request
      response = requests.get(url, params=params)

      if response.status_code == 200:
            order_book = response.json()

            #print binds and askss'
            df_bids = pd.DataFrame(order_book['bids'], columns=['price', 'size', 'num_orders'])
            df_asks = pd.DataFrame(order_book['asks'], columns=['price', 'size', 'num_orders'])

            #convert to float
            df_bids['price'] = df_bids['price'].astype(float)
            df_bids['size'] = df_bids['size'].astype(float)
            df_asks['price'] = df_asks['price'].astype(float)
            df_asks['size'] = df_asks['size'].astype(float)

            #create a list from the means - FIXED: Only return 2 values to match model
            data = [df_bids['price'].mean(), df_asks['price'].mean()]

            return data

if __name__ == '__main__':
      coin_choice = input("Enter the coin you want to fetch data for (Default: btc, Other options: eth, shib, etc.): ").strip().lower() or 'btc'
      while True:
            depth_chart(coin_choice)
