//普通函数(具名函数)声明基本格式： function 函数名(参数列表){函数体}
//匿名函数基本格式：function(参数列表){函数体}

//具名函数示例
function greet(name) {
    return `Hello, ${name}!`;
}

console.log(greet("Mike"));  // 输出：Hello, Alice!

//匿名函数示例
//1、赋值给变量
let greet2 = function (name) {
    return `Hello, ${name}!`;
};

console.log(greet2("Bob"));  // 输出：Hello, Bob!。

//2、作为参数传入，直接调用：1、(function(){ xxx  })() 或者 2、(function(){xxxx}());
console.log((function (name) { return `Hello,${name}` })("Alice")) 
