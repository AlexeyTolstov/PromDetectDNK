import json
import pandas as pd
import datetime

with open("operations.json", "r", encoding="utf-8") as file_json:
    data_oper = json.load(file_json)

data_frame = pd.DataFrame()
date = "12.12.2023"
lst_oper = ["Нахождение грузового автомобиля на складе", "Погрузка в автомобиль краном", "Погрузка в автомобиль краном", "Перемещение груза по складу", "Погрузка в автомобиль краном", "Погрузка в автомобиль краном", "Погрузка в автомобиль краном", ]

obj_lst = []
time_lst = []
time_now = datetime.time(hour=12, minute=21, second=12)
add_time = datetime.timedelta(minutes=20)
time_now = datetime.datetime.now()
for name_oper in lst_oper:
    for obj in data_oper[name_oper]:
        obj_lst.append(obj)
        time_lst.append("{:%H:%M:%S}".format(time_now))
    
    time_now =  time_now + add_time

n = len(time_lst)

data_frame["Camera ID"] = ["Cam 0"] * n
data_frame["Date"] = [date] * n

data_frame["Time"] = time_lst
data_frame["Path to saved image"] = [None] * n
data_frame["Object"] = obj_lst

print(data_frame)

data_frame.to_csv("DetecitonLog.csv")