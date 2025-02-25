"""
有序数据更易于高效地查找、分析和处理.
排序规则可以是数字大小，字符ASCII码顺序或自定义规则。
排序算法评价维度：
1、运行效率：时间复杂度低，总体操作数量较少。
2、就地性：节省内存空间。
3、稳定性：相等元素的相对位置不发生改变。
"""

"""--------------------选择排序--------------------"""
print("--------------------选择排序--------------------")

# 1、将数组分为已排序区间和未排序区间(初始状态没有已排序区间，只有未排序区间)
# 2、从未排序区间中选取最小值，放到已排序区间的末尾(即未排序区间的开头)
def selction_sort(nums:list[int]) -> list[int]:
    if len(nums) <= 1:
        return nums #长度为1的数组或空的数组不需要排序
    n = len(nums)
    for i in range(n): #未排序区间索引为[i,n-1]
        for j in range(i+1, n):
            if nums[i] > nums[j]:
                # 交换两元素位置
                nums[i] = nums[i] + nums[j]
                nums[j] = nums[i] - nums[j]
                nums[i] = nums[i] - nums[j]

nums1 = [4,1,3,1,5,2]
selction_sort(nums1)
print(nums1)
# 时间复杂度O(n^2)，空间复杂度O(1)，原地排序，非稳定排序。
print()

"""--------------------冒泡排序--------------------"""
print("--------------------冒泡排序--------------------")

# 连续的比较与交换相邻元素实现排序
def bubble_sort(nums:list[int]) -> list[int]:
    if len(nums) <= 1:
        return nums #长度为1的数组或空的数组不需要排序
    n = len(nums)
    for i in range(n):
        for j in range(0,n-i-1):
            if nums[j] > nums[j+1]: # 左端元素比右端元素大，则交换位置
                nums[j] = nums[j+1] + nums[j]
                nums[j+1] = nums[j] - nums[j+1]
                nums[j] = nums[j] - nums[j+1]
nums2 = [4,1,3,1,5,2]
bubble_sort(nums2)
print(nums2)

# 效率优化
# 如果某轮循环内没有发生元素交换，则说明已经排序完成
def bubble_sort_with_flag(nums:list[int]) -> list[int]:
    if len(nums) <= 1:
        return nums #长度为1的数组或空的数组不需要排序
    n = len(nums)
    for i in range(n):
        flag = False # 记录是否有交换，初始化为没有交换
        for j in range(0, n - i - 1):
            if nums[j] > nums[j + 1]:  # 左端元素比右端元素大，则交换位置
                nums[j] = nums[j + 1] + nums[j]
                nums[j + 1] = nums[j] - nums[j + 1]
                nums[j] = nums[j] - nums[j + 1]
                flag = True # 发生交换，flag为真
        if not flag:
            break #没有发生交换，退出循环

nums3 = [4,1,3,1,5,2]
bubble_sort_with_flag(nums3)
print(nums3)
# 时间复杂度O(n^2)，自适应排序。引入flag优化后，最佳时间复杂度为O(n)
# 空间复杂度O(1)，原地排序，稳定排序。
print()

"""--------------------插入排序--------------------"""
print("--------------------插入排序--------------------")

# 1、将数组分为已排序区间和未排序区间(初始状态将第一个元素视为已排序区间，往后的元素视为未排序区间)。
# 2、从未排序区间中选择一个元素，将其插入到已排序区间中的正确位置。
def insertion_sort(nums:list[int]) -> list[int]:
    if len(nums) <= 1:
        return nums #长度为1的数组或空的数组不需要排序
    for i in range(1, len(nums)): #第一个(下标0)是已排序区间，所以从1开始
        base = nums[i] # 取出未排序区间的第一个数
        j = i - 1
        while j>=0 and nums[j] > base:
            nums[j+1] = nums[j]
            j -= 1
        nums[j+1] = base

nums4 = [4,1,3,1,5,2]
insertion_sort(nums4)
print(nums4)
# 时间复杂度O(n^2)，自适应排序。若输入数据完全有序，则时间复杂度维O(n)
# 空间复杂度O(1)，原地排序，稳定排序。
print()

"""--------------------快速排序--------------------"""
print("--------------------快速排序--------------------")

# 1、选取数组中的某个元素为基数
# 2、所有大于基数的元素移到基数右边，所有小于基数的元素移到基数左边
# 3、对基数两边的左子数组和右子数组分别进行上述步骤(1，2步骤称为哨兵划分)

def partition(nums:list[int], left, right) -> list[int]: #哨兵划分
    if left >= right:
        return  # 长度为1的数组或空的数组不需要划分

    base = nums[(left + right) // 2]  # 取处于中间的元素为基准
    i, j = left, right
    while i < j:
        while i < j and nums[i] < base:  # 从左往右找到第一个大于基数的元素
            i += 1
        while i < j and nums[j] > base:  # 从右往左找到第一个小于基数的元素
            j -= 1
        # 交换两元素
        if i < j:
            nums[i], nums[j] = nums[j], nums[i]
            # 交换完成后，直接移动指针
            i += 1
            j -= 1

    # 循环结束时，i 对应的位置为基数
    return i #返回基数的索引

def quick_sort(nums:list[int], left, right) -> list[int]:
    if left >= right:
        return  #长度为1的数组或空的数组不需要排序

    base_index = partition(nums, left, right)

    quick_sort(nums, left, base_index - 1) #对左子数组快速排序
    quick_sort(nums, base_index + 1, right) #对右子数组快速排序

nums5 = [4,1,3,1,5,2]
quick_sort(nums5, 0, len(nums5)-1)
print(nums5)

# 尾递归优化，优先对长度较短的子数组进行快速排序
def quick_sort_tail_recursion_improved(nums:list[int], left, right) -> list[int]:
    if left >= right:
        return #长度为1的数组或空的数组不需要排序

    while left < right:
        base_index = partition(nums, left, right)
        if base_index - left < right - base_index:
            quick_sort_tail_recursion_improved(nums, left, base_index - 1)
            left = base_index + 1 #剩余未排序区间为右子数组
        else:
            quick_sort_tail_recursion_improved(nums, base_index + 1, right)
            right = base_index - 1 #剩余未排序区间为左子数组

nums6 = [4,1,3,1,5,2]
quick_sort_tail_recursion_improved(nums6, 0, len(nums6)-1)
print(nums6)