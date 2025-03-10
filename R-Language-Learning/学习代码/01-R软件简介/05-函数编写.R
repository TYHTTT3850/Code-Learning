# 无参函数
greet <- function() {
  print("Hello, world!")
}

# 有参函数
add_numbers <- function(a, b) {
  return(a + b)
}

# 带有名参数和默认参数的函数
power_of <- function(base = 2, exponent = 1) {
  return(base^exponent)
}

# 调用无参函数
greet()

# 调用有参函数
result_add <- add_numbers(5, 3)
print(result_add)  # 输出: 8

# 使用有名参数调用带默认值的函数
result_power1 <- power_of(exponent = 3, base = 4)  # 指定所有参数
print(result_power1)  # 输出: 64

result_power2 <- power_of(exponent = 3)  # 使用默认base值
print(result_power2)  # 输出: 8 (2^3)

result_power3 <- power_of()  # 使用全部默认值
print(result_power3)  # 输出: 2 (2^1)
