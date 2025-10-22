// extends 关键字有三大用法：1、对象继承。2、泛型约束。3、条件类型

// 1、对象继承
class Animal {
    move() {
      console.log("Moving");
    }
  }
  
class Dog extends Animal {
bark() {
    console.log("Woof!");
}
}

// 2、泛型约束
function printLength<T extends { length: number }>(arg: T) {
    console.log(arg.length);
}

// 3、条件类型
type IsString<T> = T extends string ? "Yes" : "No"; // 主要和三元运算符一起使用
// 如果 T 是 string，就返回类型 "Yes"，否则返回类型 "No"、
// "Yes" 和 "No" 都是字面量类型。
// 当某个变量的类型为字面量类型时，这个变量只能赋值为对应的字面量

const is_string:IsString<string> = "Yes"
type A = IsString<number> //类型 A 的 别名为 "No"
const Type:A = "No" // 若赋值为"Yes"或其他任何"No"以外的字符串，都会报错 
