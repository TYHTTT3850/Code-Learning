staff = {"王力鸿":{"部门":"科技部","工资":3000,"级别":1},
         "周杰伦":{"部门":"市场部","工资":5000,"级别":2},
         "林俊节":{"部门":"市场部","工资":7000,"级别":3},
         "张学油":{"部门":"科技部","工资":4000,"级别":1},
         "刘德滑":{"部门":"市场部","工资":6000,"级别":2}
         }
print(f"更新前员工信息：")
for key in staff:
    print(f"{key}:{staff[key]}")
print()
# 所有1级员工上升1级，工资加1000
for key in staff:
    if staff[key]["级别"] == 1:
        staff[key]["级别"] = 2
        staff[key]["工资"] += 1000
print(f"更新后员工信息：")
for key in staff:
    print(f"{key}:{staff[key]}")