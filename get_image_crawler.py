from icrawler.builtin import GoogleImageCrawler


google_crawler = GoogleImageCrawler(storage={'root_dir': 'other_test_image'})
google_crawler.crawl(keyword='face', max_num=30)
