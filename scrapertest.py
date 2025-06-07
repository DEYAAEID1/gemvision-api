from icrawler.builtin import GoogleImageCrawler

crawler = GoogleImageCrawler(storage={'root_dir': 'dataset/ruby'})
crawler.crawl(keyword='natural ruby gemstone site:pinterest.com', max_num=50)
