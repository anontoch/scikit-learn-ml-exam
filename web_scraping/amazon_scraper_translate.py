from playwright.sync_api import sync_playwright
from deep_translator import GoogleTranslator

def scrape_product(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # IMPORTANT: False to debug
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        )
        page = context.new_page()

        page.goto(url, timeout=60000)

        # Wait for page to load properly
        page.wait_for_timeout(5000)

        # Try getting title (fallback methods)
        try:
            title = page.title()
        except:
            title = "Unknown"

        # Get images (better approach)
        imgs = page.locator("img").all()
        img_urls = []

        for img in imgs:
            src = img.get_attribute("src")
            if src and "jpg" in src:
                img_urls.append(src)

        img_urls = img_urls[:5]

        # Translate
        translated_title = GoogleTranslator(source='auto', target='en').translate(title)

        browser.close()

        return {
            "title_cn": title,
            "title_en": translated_title,
            "images": img_urls
        }


data = scrape_product("https://www.amazon.co.jp/Prevention-Compatible-Anti-Theft-Motorcycle-Countermeasure/dp/B00BLWYXO0/ref=lp_26269916051_1_2?pf_rd_p=04a09edd-da51-4757-a034-f654f3da9d84&pf_rd_r=09695W4XV5R7AWFRGQYC&sbo=RZvfv%2F%2FHxDF%2BO5021pAnSA%3D%3D&th=1")
print(data)