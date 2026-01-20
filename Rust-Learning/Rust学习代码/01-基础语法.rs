fn main() {
    const NUM:i32 = 10;//常量必须指明类型
    println!("{NUM}");
    let mut x = 5;//加上mut关键字就是可变变量，变量有自动类型推导
    println!("The value of x is: {x}");
    x = 6;//若没有mut关键字，则此行报错
    println!("The value of x is: {x}");

    //内外作用域
    {
        let y = 5;
        println!("inner scope: {y}");
    }
    let y = 10;
    println!("outer scope: {y}");

}
