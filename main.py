from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import pandas as pd

import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

async def fetch_bluesky(query: str, limit: int = 10):
    url = f'https://bsky.app/search?q={query}'
    posts = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print(f"Fetching content: {url}")
        
        try:
            await page.goto(url, timeout=60000)
            print(f"Loaded content: {url}")

            
            await page.wait_for_selector('div.css-146c3p1[data-testid="postText"]', timeout=60000)
            content_html = await page.content()
            soup = BeautifulSoup(content_html, 'html.parser')
            
            tweets = soup.find_all('div', class_='css-146c3p1', attrs={'data-testid': 'postText'}, limit=limit)
            if tweets:
                for tweet in tweets:
                    content = tweet.text.strip()
                    posts.append({'Content': content})
                    
        except Exception as e:
            print(e)
        finally:
            await page.close()

    df = pd.DataFrame(posts)
    return df

async def main():
    query = input("Enter a BlueSky search query: ")
    limit = 25
    df = await fetch_bluesky(query, limit)  

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    def calculate_hf_scores(text):
        encoded_text = tokenizer(text, return_tensors='pt')
        output = model(**encoded_text)

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        result_dict = {
            'neg': scores[0],
            'neu': scores[1],
            'pos': scores[2]
        }
        return result_dict

    result = {}

    for i, row in df.iterrows():
        try:
            text = row['Content']
            hf_result = calculate_hf_scores(text)

            result[i] = hf_result
        except RuntimeError:
            continue

    results_df = pd.DataFrame(result).T
    results_df = results_df.merge(df, left_index=True, right_index=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(results_df['pos'], results_df['neg'])
    plt.xlabel('Positive Sentiment')
    plt.ylabel('Negative Sentiment')
    plt.title(f'BlueSky Sentiment Data on {query}, using HF transformers')
    plt.show()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
    