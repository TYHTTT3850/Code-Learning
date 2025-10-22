// 泛型基础
// 定义时不确定是什么类型，只有在用的时候才确定是什么类型

// 最基本的用法
type Person<T> = {
    name:string,
    age:number,
    info:T
}

const person:Person<{hobby:string[]}> = {
    name:"Tom",
    age:12,
    info:{
        hobby:["eat","sleep"]
    }
}

// 为泛型增加约束(extends关键字)
type Pig<T extends {hobby:string[]}> = {//必须有{hobby:string[]}类型，有没有其他类型不管
    name:string,
    age:number,
    info:T
}

const pig:Pig<{hobby:string[],b:string}> = {
    name:"111",
    age:11,
    info:{
        hobby:["eat","sleep"],
        b:"abcd"
    }
}

// 泛型的默认值
type Dog<T = {hobby:string[]}> = {//默认 T 是 {hobby:string[]} 类型，但也可以设定成其他类型
    name:string,
    age:number,
    info:T
}

const dog:Dog<string> = {
    name:"222",
    age:22,
    info:"None"
}
