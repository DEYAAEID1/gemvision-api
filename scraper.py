from icrawler.builtin import GoogleImageCrawler

crawler = GoogleImageCrawler(storage={'root_dir': 'dataset/ruby'})
crawler.crawl(keyword='site:pinterest.com natural ruby gemstone', max_num=50)
