# Python program for implementation of Bubble Sort 

def bubbleSort(arr): 
    n = len(arr) 

    # Traverse through all array elements 
    for i in range(n): 

        # Last i elements are already in place 
        for j in range(0, n-i-1): 

            # traverse the array from 0 to n-i-1 
            # Swap if the element found is greater 
            # than the next element 
            if arr[j] > arr[j+1] : 
                arr[j], arr[j+1] = arr[j+1], arr[j] 

# Driver code to test above 
arr = [64, 34, 25, 12, 22, 11, 90] 

bubbleSort(arr) 

print(arr)

def selection_sort(A):
    # Python program for implementation of Selection 
# Sort 

    n = len(A)
    # Traverse through all array elements 
    for i in range(n): 
        
        # Find the minimum element in remaining 
        # unsorted array 
        min_idx = i 
        for j in range(i+1, n): 
            if A[min_idx] > A[j]: 
                min_idx = j 
                
        # Swap the found minimum element with 
        # the first element      
        A[i], A[min_idx] = A[min_idx], A[i] 

arr = [64, 34, 25, 12, 22, 11, 90] 
selection_sort(arr)
print(arr)

x = [[1,2,3],[4,5,6],[7,8,9]]
ans = sum(map(max,x))
print(ans)

def insertion_sort(arr):
        
    for i in range(len(arr)):
        cursor = arr[i]
        pos = i
        while pos > 0 and arr[pos - 1] > cursor:
            
            # Swap the number down the list
            arr[pos] = arr[pos - 1]
            pos = pos - 1
        # Break and do the final swap
        arr[pos] = cursor

    return arr

arr = [64, 34, 25, 12, 22, 11, 90] 
ans = insertion_sort(arr)
print(ans)

# two sum **************************************************
def two_sum(nums,target):
    n = len(nums)
    for i in range(n):
        result = 0
        for j in range(i+1,n):
            result = nums[i] + nums[j]
            if result == target:
                return (i,j)
    return (-1,-1)

# use dictionaries instead ***********************************
def two_sum2(nums,target):
    my_hash = {}
    for i,j in enumerate(nums):
        result = target - j
        if result not in my_hash:
            my_hash[j] = i
        else:
            return my_hash[result],i
    return (-1,-1)



# maximum subarray ****************************************************
nums = [-2,1,-3,4,-1,2,1,-5,4]

def maximum_subarray(nums):
    n = len(nums)
    if n == 0: return 0
    best_seen = nums[0] # assume first value is first seen
    runningTotal = 0
    for i in range(n):
        # set running total to current number
        runningTotal = nums[i]
        if runningTotal > best_seen: best_seen = runningTotal
        for j in range(i+1,n):
            runningTotal += nums[j]
            # compare running total to best seen value
            if runningTotal > best_seen: best_seen = runningTotal
    return best_seen

def maximum_subarray2(nums):
    n = len(nums)
    if n == 0: return 0
    prev_val = nums[0]
    best_seen = nums[0]
    for i in range(n):
        prev_val = prev_val + nums[i] if prev_val > 0 else nums[i]
        best_seen = max(prev_val,best_seen)
    return best_seen
## *********************************************************************