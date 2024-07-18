import pandas as pd
import aiohttp
import asyncio
from bs4 import BeautifulSoup

# Manually provide a list of numbers (uids)
uids = [
    35654,
    27829,
    6880,
    24625,
    2649,
    13599,
    25755,
    1117,
    34318,
    8676,
    36043,
    10075,
    35828,
    10378,
    33082,
    16934
]

# Define the columns we want to scrape
columns = ['Type', 'Producers', 'Studios', 'Source', 'Themes', 'Demographic', 'Duration', 'Favorites', 'Streaming Platforms']

# Create a DataFrame to store the scraped data
scraped_data = pd.DataFrame(columns=['uid'] + columns)

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
                label = div.find('span', class_='dark_text')
                if not label:
                    continue
                label_text = label.text.strip().strip(':')
                if label_text == 'Type':
                    info['Type'] = div.find('a').text if div.find('a') else div.contents[-1].strip()
                elif label_text == 'Producers':
                    info['Producers'] = [a.text for a in div.find_all('a')]
                elif label_text == 'Studios':
                    info['Studios'] = [a.text for a in div.find_all('a')]
                elif label_text == 'Source':
                    info['Source'] = div.find('a').text.strip() if div.find('a') else div.contents[-1].strip()
                elif label_text == 'Themes' or label_text == 'Theme':
                    info['Themes'] = [a.text for a in div.find_all('a')] or [div.contents[-1].strip()]
                elif label_text == 'Demographic':
                    info['Demographic'] = div.find('a').text if div.find('a') else div.contents[-1].strip()
                elif label_text == 'Duration':
                    info['Duration'] = div.contents[-1].strip()

        # Extract the 'Statistics' section
        stats_section = soup.find('h2', string='Statistics')
        if stats_section:
            for div in stats_section.find_next_siblings('div', class_='spaceit_pad'):
                label = div.find('span', class_='dark_text')
                if label and label.text.strip().strip(':') == 'Favorites':
                    info['Favorites'] = div.contents[-1].strip()
                    break

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
async def main(uids):
    async with aiohttp.ClientSession() as session:
        tasks = [scrape(uid, f'https://myanimelist.net/anime/{uid}', session) for uid in uids]
        results = await asyncio.gather(*tasks)
        return results

# Run the asynchronous scraping
results = asyncio.run(main(uids))

# Convert the results to a DataFrame and save to a CSV file
scraped_data = pd.DataFrame(results)
scraped_data.to_csv('./data/scraped_anime_data_manual.csv', index=False)

print("Scraping completed and data saved to scraped_anime_data_manual.csv")
