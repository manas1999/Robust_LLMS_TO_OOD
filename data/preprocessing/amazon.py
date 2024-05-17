def download_amazon_dataset():

    urls = r"""
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/AMAZON_FASHION_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/All_Beauty_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Appliances_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Arts_Crafts_and_Sewing_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Automotive_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Books_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/CDs_and_Vinyl_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Cell_Phones_and_Accessories_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Clothing_Shoes_and_Jewelry_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Digital_Music_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Electronics_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Gift_Cards_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Grocery_and_Gourmet_Food_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Home_and_Kitchen_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Industrial_and_Scientific_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Kindle_Store_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Luxury_Beauty_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Magazine_Subscriptions_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Movies_and_TV_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Office_Products_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Patio_Lawn_and_Garden_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Pet_Supplies_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Prime_Pantry_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Software_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Sports_and_Outdoors_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Tools_and_Home_Improvement_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Toys_and_Games_5.json.gz
    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz
    """.split("\n")

    urls = [url for url in urls if url != ""]
    import os

    os.system("cd /Users/manasmadine/Desktop/OneDrive/NLP/Project_Experements/EXP_1/data/raw/SentimentAnalysis/amazon")

    for url in urls:
        os.system(f"wget {url} --no-check-certificate")
        dataset = url.split("/")[-1]
        os.system(f"gzip -d {dataset}")
        dataset_name = dataset.strip(".json.gz")
        new_name = dataset_name.strip("_5").lower()
        os.system(f"mv {dataset_name}.json {new_name}.json")

def download_and_process(urls, max_records_per_file=1000, total_max_records=20000):
    total_records = 0
    for url in urls:
        if total_records >= total_max_records:
            break
        print(f"Downloading and processing from {url}...")
        with requests.get(url, stream=True) as response:
            if response.status_code == 200:
                decompressor = gzip.GzipFile(fileobj=response.raw)
                
                reader = io.BufferedReader(decompressor)
                
                count = 0
                for line in reader:
                    if count >= max_records_per_file or total_records >= total_max_records:
                        break
                    data = json.loads(line.decode('utf-8'))
                    count += 1
                    total_records += 1
                    print(data.get('reviewText', 'No review text found'))

                print(f"Processed {count} records from {url.split('/')[-1]}")





