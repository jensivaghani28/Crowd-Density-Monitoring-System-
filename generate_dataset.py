import pandas as pd
import random
from datetime import datetime, timedelta

rows = []

locations = ["Gate A", "Gate B", "Gate C", "Parking", "Stage Area"]
weather = ["Sunny", "Cloudy", "Rainy"]
events = ["Normal", "Festival", "Concert", "Emergency"]

start_time = datetime(2026, 1, 1, 8, 0)

for i in range(1500):  # 1500 rows (GOOD for project)
    timestamp = start_time + timedelta(minutes=30*i)
    location = random.choice(locations)
    
    people_count = random.randint(50, 1000)
    area_size = random.randint(40, 100)
    density = round(people_count / area_size, 2)

    row = [
        timestamp,
        location,
        people_count,
        area_size,
        density,
        random.choice(weather),
        random.choice(events)
    ]

    rows.append(row)

df = pd.DataFrame(rows, columns=[
    "timestamp", "location", "people_count",
    "area_size", "density", "weather", "event_type"
])

df.to_csv("crowd_data.csv", index=False)

print("Dataset created successfully!")