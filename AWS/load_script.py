import requests
import concurrent.futures
import time

# define the number of parallel requests to send
num_requests = 1000000

# max workers = The maximum number of threads that can be used to
#                 execute the given calls.

# define a function to send a single request
def send_request():
    headers = {'Content-Type': 'application/json'}
    myobj = {"userinput": "science"}
    r = requests.post('http://1dadaf53-us-south.lb.appdomain.cloud:5000/predict', files = { 'userinput': (None, myobj['userinput']),})
    return r.status_code
futures = []
# use a ThreadPoolExecutor to send multiple requests in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=8000) as executor:
    # create a list of futures for each request
    for i in range(num_requests):
        futures.append(executor.submit(send_request))
        print('Submitted request ', i)
        # if (i % 1200 == 0):
        #     time.sleep(1) # to slow down, to not overwhelm memory buffers
    
    # wait for all futures to complete and print the results
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        print(result)