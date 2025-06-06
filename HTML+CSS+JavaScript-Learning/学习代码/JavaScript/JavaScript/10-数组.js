// 1. 语法，使用 [] 来定义一个空数组
// 定义一个空数组，然后赋值给变量 classes
// let classes = [];

// 2. 定义非空数组
let classes = ['小明', '小刚', '小红', '小丽', '小米']

// 3. 访问数组，语法格式为：变量名[索引值]
document.writeln(classes[0]) // 结果为：小明
document.writeln(classes[1]) // 结果为：小刚
document.writeln(classes[4]) // 结果为：小米

// 4. 通过索引值还可以为数组单重新赋值
document.writeln(classes[3]) // 结果为：小丽
// 重新为索引值为 3 的单元赋值
classes[3] = '小小丽'
document.writeln(classes[3]); // 结果为： 小小丽

// 5. 数组单值类型可以是任意数据类型
// a) 数组单元值的类型为字符类型
let list = ['HTML', 'CSS', 'JavaScript']
// b) 数组单元值的类型为数值类型
let scores = [78, 84, 70, 62, 75]
// c) 混合多种类型
let mixin = [true, 1, false, 'hello']

let arr = ['html', 'css', 'javascript']
// 数组对应着一个 length 属性，它的含义是获取数组的长度
console.log(arr.length) // 3

// 1. push 动态向数组的尾部添加一个单元
arr.push('Nodejs')
console.log(arr)
arr.push('Vue')

// 2. unshit 动态向数组头部添加一个单元
arr.unshift('VS Code')
console.log(arr)

// 3. splice 添加或删除数组中的元素。删除：splice(起始位置， 删除的个数)。添加：splice(起始位置，删除个数，添加数组元素)
arr.splice(2, 1) // 从索引值为2的位置开始删除1个单元
console.log(arr)
arr.splice(1, 0, 'pink', 'hotpink') // 在索引号是1的位置添加 pink  hotpink，删除0个就相当于没删除

// 4. pop 删除最后一个单元
arr.pop()
console.log(arr)

// 5. shift 删除第一个单元
arr.shift()
console.log(arr)
