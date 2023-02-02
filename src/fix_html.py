from bs4 import BeautifulSoup as bs

def fix_html(html):
    soup = bs(html, 'html.parser')
    for tag in soup.find_all():
        tag['tid'] = tag['href'].replace('http://', 'https://')
    return soup.prettify()

if __name__ == '__main__':
    for int in range(0,7):
        with open(f'../data/jobs/10/processed_data/100000{int}.html', 'r') as f:
            html = f.read()