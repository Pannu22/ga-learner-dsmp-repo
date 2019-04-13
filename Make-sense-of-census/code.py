# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'
data = np.genfromtxt(path,delimiter=',',skip_header=1)
#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]
census = np.concatenate((data,new_record))
print(census)
#Code starts here



# --------------
#Code starts here
age = census[:,0]
max_age = np.max(age)
print('Max age :',max_age)
min_age = np.min(age)
print('Min age :',min_age)
age_mean = np.mean(age)
print('Mean of age :',age_mean)
age_std = np.std(age)
print('Standard deaviation of age',age_std)


# --------------
#Code starts here
race_0 = census[census[:,2] == 0]
race_1 = census[census[:,2] == 1]
race_2 = census[census[:,2] == 2]
race_3 = census[census[:,2] == 3]
race_4 = census[census[:,2] == 4]
len_0 = len(race_0)
len_1 = len(race_1)
len_2 = len(race_2)
len_3 = len(race_3)
len_4 = len(race_4)
len_race = np.array([len_0,len_1,len_2,len_3,len_4])
min_value = np.min(len_race)
if len_0 == min_value :
    minority_race = 0
elif len_1 == min_value :
    minority_race = 1
elif len_2 == min_value :
    minority_race = 2
elif len_3 == min_value :
    minority_race = 3
elif len_4 == min_value :
    minority_race = 4

print(minority_race)



# --------------
#Code starts here
senior_citizens = census[census[:,0] > 60].astype(np.int16)
working_hours_sum = np.sum(senior_citizens[:,6])
senior_citizens_len = len(senior_citizens)
avg_working_hours = working_hours_sum/senior_citizens_len
print('Average working hours :',avg_working_hours)


# --------------
#Code starts here
high = census[census[:,1] > 10].astype(np.int16)
low = census[census[:,1] <= 10].astype(np.int16)
avg_pay_high = np.mean(high[:,7])
avg_pay_low = np.mean(low[:,7])
if avg_pay_high > avg_pay_low :
    print('It is true that better education leads to better pay')
else :
    print('It is not true that better education leads to better pay')


