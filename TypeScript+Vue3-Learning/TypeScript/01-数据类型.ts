// Typescript是一个静态数据类型的编程语言
// 1、基础数据类型
// number、string、boolean、null、undefined、symbol、bigint
let a:number = 1
let b:string = "hello"
let c:boolean = true
let d:null = null
let e:undefined = undefined
let f:symbol = Symbol("symbol")
let g:bigint = BigInt(100)

// 2、引用数据类型
let h:Object = {}//对象
let i:Array<number> = [1,2,3]//数组
let j:Date = new Date()//日期
let k:Function = function(){}//函数
let l:RegExp = /hello///正则表达式