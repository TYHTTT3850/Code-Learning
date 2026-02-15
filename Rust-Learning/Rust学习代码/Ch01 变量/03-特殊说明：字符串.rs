fn main() {
    //rust中，字符串字面量和字符串类型被视为不同的实体
    let string_literal = "hello world!"; //字符串字面量
    let string_type:String = "hello world".to_string(); //字符串类型
    println!("{}\n{}", string_literal,string_type);
}
