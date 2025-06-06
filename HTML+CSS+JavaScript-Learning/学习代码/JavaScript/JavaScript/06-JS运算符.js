
// 1、算术运算符
console.log(1 + 2 * 3 / 2) //  4 
let num = 10
console.log(num + 10)  // 20
console.log(num + num)  // 20

// 取模(取余数)使用场景：用来判断某个数是否能够被整除
console.log(4 % 2) //  0  
console.log(6 % 3) //  0
console.log(5 % 3) //  2
console.log(3 % 5) //  3

// 注意事项 : 如果我们计算失败，则返回的结果是 NaN (not a number)
console.log('hello' - 2)
console.log('hello' * 2)
console.log('hello' + 2)   // 字符串拼接(隐式转换)

// 2、赋值运算符
let num1 = 1
// num = num + 1
// 采取赋值运算符
// num += 1 等价于 num = num + 1
num1 += 3
console.log(num1)
// 同理还有 -= *= /= %=

// 3、自增运算符
// 前置自增
let i = 1
let i1 = ++i
console.log(i1) // 先加再赋值

// 后置自增
let i2 = 1
let i3 = i2++
console.log(i3) // 先赋值再加

// 3、比较运算符，返回 true 或 false
console.log(3 > 5)
console.log(3 >= 3)
console.log(2 == 2)

// 比较运算符有隐式转换 把'2' 转换为 2  双等号 只判断值
console.log(2 == '2')  // true

// === 全等 判断 值 和 数据类型都一样才行

// 以后判断是否相等 请用 ===  
console.log(2 === '2')
console.log(NaN === NaN) // NaN 不等于任何人，包括他自己
console.log(2 !== '2')  // true，类型不一样  
console.log(2 != '2') // false，值比较，有隐式转换 
console.log('a' < 'b') // true
console.log('aa' < 'ab') // true
console.log('aa' < 'aac') // true

// 4、逻辑运算符

// 逻辑与(合取)
console.log(true && true)
console.log(false && true)
console.log(3 < 5 && 3 > 2)
console.log(3 < 5 && 3 < 2)

// 逻辑或(析取)
console.log(true || true)
console.log(false || true)
console.log(false || false)

// 逻辑非
console.log(!true)
console.log(!false)

let num2 = 6
console.log(num2 > 5 && num2 < 10)

//逻辑运算符优先级： ！> && > || ，搞不清楚就加()
