import requests
import os
import time
from collections import OrderedDict
import random
from lxml.html import fromstring
from itertools import cycle
import re


def save_html_file(content, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)


def get_proxies():
    url = 'https://free-proxy-list.net/'
    response = requests.get(url)
    proxies = re.findall(r'[0-9]+(?:\.[0-9]+){3}:[0-9]+', response.text)
    return list(set(proxies))


def scrape_canlii(start_year: int, end_year: int, output_dir: str, base_url: str, initial_index: int = 1):
    assert start_year <= end_year, "The start year must be less than or equal to the end year."

    # Create a directory to store the downloaded HTML files
    os.makedirs(output_dir, exist_ok=True)

    proxies = get_proxies()
    proxy_pool = cycle(proxies)

    headers_list = [
        # Firefox 77 Mac
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        },
        # Firefox 77 Windows
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.google.com/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        },
        # Chrome 83 Mac
        {
            "Connection": "keep-alive",
            "DNT": "1",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Dest": "document",
            "Referer": "https://www.google.com/",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8"
        },
        # Chrome 83 Windows 
        {
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-User": "?1",
            "Sec-Fetch-Dest": "document",
            "Referer": "https://www.google.com/",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9"
        }
    ]
    # Create ordered dict from Headers above
    ordered_headers_list = []
    for headers in headers_list:
        h = OrderedDict()
        for header,value in headers.items():
            h[header]=value
        ordered_headers_list.append(h)

    # Iterate over the years
    for year in range(start_year, end_year + 1):
        if year == start_year:
            i = initial_index
        else:
            i = 1

        case_fr_exists = True
        while i < 500:
            if case_fr_exists:
                lang = 'fr'
            else:
                lang = 'en'
            # Construct the URL for the current year and id
            url = base_url.format(lang=lang, year=year, id=i)

            if os.path.exists(os.path.join(output_dir, f"{url.split('/')[-1]}")):
                i += 1
                continue

            # Fetch the HTML content of the website
            headers = random.choice(headers_list)
            proxy = next(proxy_pool)
            try:
                response = requests.get(url, headers=headers, proxies={"http": proxy, "https": proxy}, timeout=10)
            except:
                print(f"Connnection error, removing {proxy} from pool")
                proxies.remove(proxy)
                proxy_pool = cycle(proxies)
                if len(proxies) == 0:
                    print("No more proxies, exiting")
                    print(year, i)
                    return
                continue

            # If the request was successful (status code 200), parse the HTML content
            if response.status_code == 200:
                html_content = response.text

                # Save the fetched HTML content as a file
                filename = os.path.join(output_dir, f"{url.split('/')[-1]}")
                save_html_file(html_content, filename)
                print(f"Saved {url} as {filename}")
                i += 1
                case_fr_exists = True
            else:
                if response.status_code in [404, 403]:
                    if case_fr_exists:
                        case_fr_exists = False
                    else:
                        case_fr_exists = True
                        i += 1
                print(f"Failed to fetch the HTML content of {url}. Status code: {response.status_code}")

                if response.status_code == 429:
                    print(f"Connnection error, removing {proxy} from pool")
                    proxies.remove(proxy)
                    proxy_pool = cycle(proxies)
                    if len(proxies) == 0:
                        print("No more proxies, exiting")
                        print(year, i)
                        return
                    continue
            
            time.sleep(5)


if __name__ == '__main__':
    start_year = 2011
    end_year = 2022
    output_dir = '/home/hadrien/Documents/canlii/'
    base_url = 'https://www.canlii.org/{lang}/qc/qccm/doc/{year}/{year}qccm{id}/{year}qccm{id}.html'

    scrape_canlii(start_year, end_year, output_dir, base_url, 90)