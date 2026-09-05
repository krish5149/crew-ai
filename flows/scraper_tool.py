from crewai_tools import ScrapeWebsiteTool

scraper = ScrapeWebsiteTool(
    website_url="https://example.com/job/python-developer"
)

content = scraper.run()
print(content)