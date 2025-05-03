// 通过 type 关键字自定义新的类型(类型别名)，例如：
type OrderID = number //定义订单ID类型，现在是number，需要更改时，只需改成其他类型即可，如string、bigint

// 联合类型
type AccountID = number|string|bigint

// 对象
type Gender = "man"|"woman"
type Person = {
    name:string,
    age:number,
    info:{
        gender:Gender
    }
}
type Pig = {
    name:string,
    age:number,
    hobby:{
        eat:boolean
    }
}

// 交叉类型
type Animal = Person & Pig //两个类型的所有属性都要有
const animal:Animal = {
    name:"Tom",
    age:12,
    info:{
        gender:"man"
    },
    hobby:{
        eat:true
    }
}
