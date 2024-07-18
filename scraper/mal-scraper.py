import pandas as pd
import aiohttp
import asyncio
import random
import time
from bs4 import BeautifulSoup

# Load the dataset
animes_df = pd.read_csv("./data/animes.csv")
animes_df = animes_df.drop_duplicates(subset=['uid'], keep='first')

# Define the columns we want to scrape
columns = ['Type', 'Producers', 'Studios', 'Source', 'Themes', 'Demographic', 'Duration', 'Favorites', 'Streaming Platforms']

# Asynchronous function to fetch page content
async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

# Asynchronous function to scrape data from a single page
async def scrape(uid, url, session):
    try:
        html = await fetch(session, url)
        soup = BeautifulSoup(html, 'html.parser')

        info = {
            'uid': uid,
            'Type': None,
            'Producers': [],
            'Studios': [],
            'Source': None,
            'Themes': [],
            'Demographic': None,
            'Duration': None,
            'Favorites': None,
            'Streaming Platforms': []
        }

        # Extract the 'Information' section
        info_section = soup.find('h2', string='Information')
        if info_section:
            for div in info_section.find_next_siblings('div', class_='spaceit_pad'):
                label = div.find('span', class_='dark_text').text.strip(':')
                if label in info:
                    if label == 'Producers' or label == 'Studios' or label == 'Themes':
                        info[label] = [a.text for a in div.find_all('a')]
                    elif label == 'Source':
                        info[label] = div.find('a').text.strip() if div.find('a') else div.contents[-1].strip()
                    else:
                        info[label] = div.find('a').text if div.find('a') else div.contents[-1].strip()

        # Extract the 'Statistics' section
        stats_section = soup.find('h2', string='Statistics')
        if stats_section:
            for div in stats_section.find_next_siblings('div', class_='spaceit_pad'):
                label = div.find('span', class_='dark_text').text.strip(':')
                if label in info:
                    info[label] = div.contents[-1].strip()

        # Extract the 'Streaming Platforms' section
        streaming_section = soup.find('h2', string='Streaming Platforms')
        if streaming_section:
            for div in streaming_section.find_next_siblings('div', class_='pb16'):
                platforms = [a.find('div', class_='caption').text for a in div.find_all('a', class_='broadcast-item')]
                if platforms:
                    info['Streaming Platforms'].extend(platforms)

        return info

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {'uid': uid, **{col: None for col in columns}}

# Asynchronous main function to handle multiple scraping tasks
async def main(entries):
    async with aiohttp.ClientSession() as session:
        tasks = [scrape(uid, url, session) for uid, url in entries]
        results = await asyncio.gather(*tasks)
        return results

# Function to handle the entire scraping process in batches
def run_scraping(animes_df, batch_size=300):
    total_entries = len(animes_df)
    all_results = []
    
    start_index = 0
    while start_index < total_entries:
        entries = [(row['uid'], row['link']) for _, row in animes_df.iloc[start_index:start_index+batch_size].iterrows()]
        results = asyncio.run(main(entries))
        all_results.extend(results)
        
        # Save progress to a CSV file after each batch
        scraped_data = pd.DataFrame(all_results)
        scraped_data.to_csv('./data/scraped_anime_data_progress.csv', index=False)
        
        print(f"Scraping progress: {min(start_index + batch_size, total_entries)} of {total_entries} entries completed.")
        
        # Check for consecutive empty rows
        current_batch_empty_rows = sum(1 for result in results if all(v is None or (isinstance(v, list) and not v) for k, v in result.items() if k != 'uid'))
        
        if current_batch_empty_rows >= 50:
            print("Ten consecutive empty rows detected. Pausing for 10 minutes...")
            time.sleep(10 * 60)  # Sleep for 10 minutes
            all_results = all_results[:-batch_size]  # Remove the last batch results
            continue  # Retry the last batch without advancing the start_index
        
        # Introduce a delay between batches to avoid detection
        time.sleep(random.uniform(80, 150))  # Sleep for 80-150 seconds
        
        # Move to the next batch
        start_index += batch_size

    # Save the final results to a CSV file
    scraped_data = pd.DataFrame(all_results)
    scraped_data.to_csv('./data/scraped_anime_data_final.csv', index=False)

    print("Scraping completed and data saved to scraped_anime_data_final.csv")

# Run the scraping process
run_scraping(animes_df)