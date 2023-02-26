import selenium
from selenium import webdriver
from selenium import DesiredCapabilities
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

service = Service(executable_path='/home/yw583/make-it-move/chromedriver.exe')
driver = webdriver.Chrome(service=service)
driver.get('https://www.reddit.com/r/gifMemes/')
print(driver.title)

gifs = driver.find_elements(By.TAG_NAME, value="video")
for gif in gifs:
    print(driver.find_element(By.TAG_NAME, value="source").get_attribute("src"))
driver.implicitly_wait(5)
driver.quit()
