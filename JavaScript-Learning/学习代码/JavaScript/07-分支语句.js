
// 1、分支语句
/*
if语句：
if(条件表达式1) {
满足条件要执行的语句
}
else if(条件表达式2){
满足条件要执行的语句
}
...
else if(条件表达式N){
满足条件要执行的语句
}
else{
以上条件都不满足时要执行的语句
}
 
三元表达式：简单的分支判断
条件 ? 表达式1 : 表达式2 
如果条件为真，则执行表达式1，如果条件为假，则执行表达式2

switch语句：一般用于等值判断
switch (表达式) {
case 值1:
代码1
break

case 值2:
代码2
break
...
default:
代码n

// }
*/

let num = prompt("输入一个大于100的数字")
if (num > 100) {
    console.log("输入正确")
}
else {
    console.log("输入错误")
}

console.log(5 < 3 ? '真的' : '假的')

let num1 = 2
switch (num1) {
    case 1:
        console.log("值为1")
        break
    case 2:
        console.log("值为2")
        break
    default:
        console.log("既不是1也不是2")
}
