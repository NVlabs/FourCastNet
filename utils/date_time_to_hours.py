#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The code was authored by the following people:
#
#Jaideep Pathak - NVIDIA Corporation
#Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
#Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
#Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory 
#Ashesh Chattopadhyay - Rice University 
#Morteza Mardani - NVIDIA Corporation 
#Thorsten Kurth - NVIDIA Corporation 
#David Hall - NVIDIA Corporation 
#Zongyi Li - California Institute of Technology, NVIDIA Corporation 
#Kamyar Azizzadenesheli - Purdue University 
#Pedram Hassanzadeh - Rice University 
#Karthik Kashinath - NVIDIA Corporation 
#Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import numpy as np
from datetime import datetime


#day_of_year = datetime.now().timetuple().tm_yday  # returns 1 for January 1st
#time_tuple = datetime.now().timetuple()
date_strings = ["2016-01-01 00:00:00", "2016-09-13 00:00:00", "2016-09-17 00:00:00", "2016-09-21 00:00:00", "2016-09-25 00:00:00", "2016-09-29 00:00:00", "2016-10-03 00:00:00", "2016-10-07 00:00:00"]

ics = []

for date_ in date_strings:
    date_obj = datetime.strptime(date_, '%Y-%m-%d %H:%M:%S') #datetime.fromisoformat(date_) 
    print(date_obj.timetuple())
    day_of_year = date_obj.timetuple().tm_yday - 1
    hour_of_day = date_obj.timetuple().tm_hour
    hours_since_jan_01_epoch = 24*day_of_year + hour_of_day
    ics.append(int(hours_since_jan_01_epoch/6))
    print(day_of_year, hour_of_day)
    print("hours = ", hours_since_jan_01_epoch )
    print("steps = ", hours_since_jan_01_epoch/6) 


print(ics)

ics = []
for date_ in date_strings:
    date_obj = datetime.fromisoformat(date_) #datetime.strptime(date_, '%Y-%m-%d %H:%M:%S') #datetime.fromisoformat(date_) 
    print(date_obj.timetuple())
    day_of_year = date_obj.timetuple().tm_yday - 1
    hour_of_day = date_obj.timetuple().tm_hour
    hours_since_jan_01_epoch = 24*day_of_year + hour_of_day
    ics.append(int(hours_since_jan_01_epoch/6))
    print(day_of_year, hour_of_day)
    print("hours = ", hours_since_jan_01_epoch )
    print("steps = ", hours_since_jan_01_epoch/6) 


print(ics)

