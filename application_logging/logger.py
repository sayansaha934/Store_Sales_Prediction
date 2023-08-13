import pymongo
from datetime import datetime

class App_Logger:
    def __init__(self):
        pass

        # self.client = pymongo.MongoClient(
        #     "mongodb+srv://Sales:Sales@store-sales.gmwm3.mongodb.net/?retryWrites=true&w=majority")




    def log(self, db, collection, tag, log_message):
        pass
        # database = self.client[db]
        # collection = database[collection]
        # record={
        #     "timestamp" : str(datetime.now()),
        #     "tag" : tag,
        #     "log_message" : log_message
        # }

        # collection.insert_one(record)
