from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import urllib.request
from selenium.webdriver import ActionChains
from selenium.webdriver.support.relative_locator import locate_with
import time

def _in_viewport(driver, element):
    script = (
        "for(var e=arguments[0],f=e.offsetTop,t=e.offsetLeft,o=e.offsetWidth,n=e.offsetHeight;\n"
        "e.offsetParent;)f+=(e=e.offsetParent).offsetTop,t+=e.offsetLeft;\n"
        "return f<window.pageYOffset+window.innerHeight&&t<window.pageXOffset+window.innerWidth&&f+n>\n"
        "window.pageYOffset&&t+o>window.pageXOffset"
    )
    return driver.execute_script(script, element)

# service = Service(executable_path='/home/yw583/make-it-move/chromedriver.exe')
service = Service(executable_path="/Users/yolandaw/drivers/chromedriver")
driver = webdriver.Chrome(service=service)
driver.get('https://www.reddit.com/r/gifMemes/')
print(driver.title)

gif = driver.find_element(By.TAG_NAME, value="video")
# get one element, scroll down, get another element under the previous one
for i in range(100):
    # gif = driver.find_element(By.TAG_NAME, value="video")
    start = time.time()
    print(start)
    video = gif.find_element(By.TAG_NAME, value="source")
    video_url = video.get_property('src')
    urllib.request.urlretrieve(url=video_url, filename=f'data/meme{i}.mp4')
    end = time.time()
    print(f"got meme{i}.mp4, time: {end - start}")
    ActionChains(driver).scroll_to_element(gif).perform()
    assert _in_viewport(driver, gif)
    try:
        gif = driver.find_element(locate_with(By.TAG_NAME,  "video").below(gif))
    except:
        driver.implicitly_wait(15)
        gif = driver.find_element(locate_with(By.TAG_NAME,  "video").below(gif))
driver.quit()
