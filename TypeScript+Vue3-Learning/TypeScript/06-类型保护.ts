type Value = string | number

let value:Value = '123'

// value.toFixed() // 报错。因为 value 可能是 string 类型，也可能是 number 类型

// 使用类型保护
function isString(parameters:Value): parameters is string {
    return typeof value === 'string'
} // parameters is string 是 TypeScript 中的一个类型谓词，它的作用是告诉编译器，在函数返回 true 时，parameters 类型就是 string。这样，TypeScript 在函数体内会知道 parameters 的类型被缩小为 string，并且可以安全地访问字符串特有的属性和方法。

const flag = isString(value)
console.log(flag)

function printLength(value: Value) {
    if (isString(value)) {
        console.log(value.length);  // 这里 TypeScript 知道 value 是 string 类型
    } else {
        console.log("Not a string");
    }
}

printLength("Hello, world!");  // 输出: 13
printLength(42);               // 输出: Not a string
