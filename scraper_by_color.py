from icrawler.builtin import GoogleImageCrawler
import os

colors = {
    "red": "site:pinterest.com red gemstones",
    "green": "site:pinterest.com green gemstones",
    "blue": "site:pinterest.com blue gemstones",
    "yellow": "site:pinterest.com yellow gemstones",
    "purple": "site:pinterest.com purple gemstones",
    "black": "site:pinterest.com black gemstones"
}

dataset_dir = "gemstone_colors_dataset"
os.makedirs(dataset_dir, exist_ok=True)

for color, query in colors.items():
    save_path = os.path.join(dataset_dir, color)
    os.makedirs(save_path, exist_ok=True)

    crawler = GoogleImageCrawler(storage={"root_dir": save_path})
    crawler.crawl(keyword=query, max_num=150)
