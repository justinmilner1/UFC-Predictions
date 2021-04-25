from src.createdata.preprocess import Preprocessor
from src.createdata.scrape_fight_data import FightDataScraper
from src.createdata.scrape_fighter_details import FighterDetailsScraper

print("Creating fight data \n")
fight_data_scraper = FightDataScraper()
fight_data_scraper.create_fight_data_csv()  # Scrapes raw ufc fight data from website

print("Creating fighter data \n")
fighter_details_scraper = FighterDetailsScraper()
fighter_details_scraper.create_fighter_data_csv()  # Scrapes raw ufc fighter data from website

input("If this is for inferencing, manually add the upcoming fights to total_fight_data now. Then press enter.")

print("Starting Preprocessing \n")
preprocessor = Preprocessor()
preprocessor.process_raw_data()  # Preprocesses the raw data and saves the csv files in data folder
