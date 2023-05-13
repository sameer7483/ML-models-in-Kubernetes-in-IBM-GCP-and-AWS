# import requests

# for i in range(1, 100):
#     myobj = {"userinput": "science"}
#     r = requests.post('http://34.170.211.169:5000/predict', files = { 'userinput': (None, myobj['userinput']),})
#     print(r)

import requests
import concurrent.futures
import time


# define the number of parallel requests to send
num_requests = 1000000

# max workers = The maximum number of threads that can be used to
#                 execute the given calls.

# define a function to send a single request
def send_request(i):
    myobj = {"userinput": "The 2023 NFL Draft is over and all 259 selections have been made. All that's left to do is brace for the knee-jerk reactions from media folk, ourselves included. We liked what most of the teams did over the weekend because, in reality, this is nothing more than an exercise in checking boxes"}
    r = requests.post('http://34.170.211.169:5000/predict', files = { 'userinput': (None, myobj['userinput'])})
    print('finish processing ', i)
    return r.status_code
futures = []
# use a ThreadPoolExecutor to send multiple requests in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=3600) as executor:
    # create a list of futures for each request
    for i in range(num_requests):
        futures.append(executor.submit(send_request, i))
        print('Submitted request ', i)
        if (i % 1200 == 0):
            time.sleep(5) # to slow down, to not overwhelm memory buffers
    
    # wait for all futures to complete and print the results
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        print(result)