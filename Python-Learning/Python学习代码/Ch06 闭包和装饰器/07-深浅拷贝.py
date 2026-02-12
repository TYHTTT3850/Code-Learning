"""
可变类型和不可变类型：
内存中的值可以被改变的类型叫可变类型，内存中的值不能被改变的类型叫不可变类型。
不可变类型：数值类型(int、float、bool)、字符串、元组
可变类型：列表、字典、集合

1、深浅拷贝分别指的是copy模块的copy()和deepcopy()函数，前者是浅拷贝，后者是深拷贝。
2、浅拷贝：创建一个新的对象，但对于对象中的元素，仍然是原来对象中元素的引用。(拷贝的少)
    深拷贝：创建一个新的对象，并且递归地复制对象中的元素，最终得到一个完全独立的对象。(拷贝的多)
3、深浅拷贝主要针对可变类型的元素，如果是不可变类型，则用法和普通赋值一样，没有区别。
"""

import copy

# 1. 普通赋值
# 结论：赋值只是对象的引用，内存地址完全相同。
print("--- 1. 普通赋值 (赋值引用) ---")
origin = [1, 2, [3, 4]]
assigned = origin
origin[0] = 99
print(f"原对象: {origin}, id: {id(origin)}")
print(f"赋值对象: {assigned}, id: {id(assigned)}") #一起跟着改变
print(f"结论: 指向同一内存地址? {origin is assigned}\n")

# 2. 浅拷贝 + 可变对象(如 list)
# 结论：外层容器复制了(新地址），但内部元素仍然指向原有对象的引用。
print("--- 2. 浅拷贝 + 可变对象 ---")
origin_mutable = [1, [2, 3]]
shallow_mutable = copy.copy(origin_mutable)

# 修改外层元素(互不影响,因为这样只是让origin_mutable[0]指向一个新地址，shallow_mutable[0]仍然指向原来的地址)
origin_mutable[0] = 999
# 修改内层嵌套元素(互相影响，因为shallow_mutable[1]和origin_mutable[1]指向同一个list对象)
origin_mutable[1][0] = 888

print(f"原对象: {origin_mutable}, 外层id: {id(origin_mutable)}, 内层list id: {id(origin_mutable[1])}")
print(f"浅拷贝: {shallow_mutable}, 外层id: {id(shallow_mutable)}, 内层list id: {id(shallow_mutable[1])}")
print(f"结论: 外层地址不同? {origin_mutable is not shallow_mutable}, 内层嵌套地址相同? {origin_mutable[1] is shallow_mutable[1]}\n")

# 3. 浅拷贝 + 不可变对象(如 tuple)
# 结论：如果不可变对象内部只包含不可变元素，浅拷贝不会创建新对象，而是直接返回原引用
print("--- 3. 浅拷贝 + 不可变对象 ---")
origin_immutable = (1, 2, 3)
shallow_immutable = copy.copy(origin_immutable)

print(f"原对象: {origin_immutable}, id: {id(origin_immutable)}")
print(f"浅拷贝: {shallow_immutable}, id: {id(shallow_immutable)}")
print(f"结论: 对于纯不可变对象，浅拷贝是否直接返回原引用? {origin_immutable is shallow_immutable}\n")

# 4. 深拷贝 + 可变对象
# 结论：完全复制，递归拷贝所有层级，修改原对象完全不影响新对象。
print("--- 4. 深拷贝 + 可变对象 ---")
origin_deep = [1, [2, 3]]
deep_mutable = copy.deepcopy(origin_deep)

origin_deep[1][0] = 777

print(f"原对象: {origin_deep}, 内层list id: {id(origin_deep[1])}")
print(f"深拷贝: {deep_mutable}, 内层list id: {id(deep_mutable[1])}")
print(f"结论: 内层嵌套地址也不相同? {origin_deep[1] is not deep_mutable[1]} (完全独立)\n")

# 5. 深拷贝 + 不可变对象
# 结论：如果不可变对象内部不包含可变元素，深拷贝通常也返回原引用；
#      如果不可变对象包含可变元素（如 tuple 套 list），则会创建新对象。
print("--- 5. 深拷贝 + 不可变对象 ---")
# 情况 A: 纯不可变
pure_immutable = (1, 2)
deep_pure = copy.deepcopy(pure_immutable)
print(f"纯不可变元组 id对比: 原{id(pure_immutable)} vs 深拷贝{id(deep_pure)} (相同? {pure_immutable is deep_pure})")

# 情况 B: 不可变中包含可变 (tuple holds list)
# 此时必须创建新元组，因为内部的列表必须被深拷贝
complex_immutable = (1, [2, 3])
deep_complex = copy.deepcopy(complex_immutable)
print(f"含可变元素的元组 id对比: 原{id(complex_immutable)} vs 深拷贝{id(deep_complex)} (相同? {complex_immutable is deep_complex})")
print(f"结论: 含可变元素的不可变对象，深拷贝会创建新对象。\n")
