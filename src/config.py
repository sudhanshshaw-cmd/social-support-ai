from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017")
db = client["socialsupport"]
print(db.canonical_profiles.find_one({"applicant_id": "APP100000"}, {"_id":0}))