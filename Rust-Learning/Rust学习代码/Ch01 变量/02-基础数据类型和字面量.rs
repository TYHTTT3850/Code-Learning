fn main() {
    //整型（Integers）

    // 有符号整型
    let i8_val: i8 = -8;
    let i16_val: i16 = -16;
    let i32_val: i32 = -32;
    let i64_val: i64 = -64;
    let i128_val: i128 = -128;
    let isize_val: isize = -1; // 平台相关

    // 无符号整型
    let u8_val: u8 = 8;
    let u16_val: u16 = 16;
    let u32_val: u32 = 32;
    let u64_val: u64 = 64;
    let u128_val: u128 = 128;
    let usize_val: usize = 1; // 平台相关

    // 整型字面量
    let decimal = 42;          // 默认 i32
    let hex = 0xff;            // 十六进制
    let octal = 0o77;           // 八进制
    let binary = 0b1010;        // 二进制
    let separated = 1_000_000;  // 数字分隔符


    //浮点型(Floating-Point)
    let f32_val: f32 = 3.14;
    let f64_val: f64 = 2.718281828;

    // 默认推断为 f64
    let float_infer = 0.1 + 0.2;

    //布尔型(Boolean)
    let bool_true: bool = true;
    let bool_false = false; // 类型推断

    //字符型(char)
    let char_a: char = 'A';
    let char_cn: char = '中';
    let char_emoji: char = '😀';

    println!("整型：");
    println!(
        "i8={}, i16={}, i32={}, i64={}, i128={}, isize={}",
        i8_val, i16_val, i32_val, i64_val, i128_val, isize_val
    );

    println!("\n无符号整型：");
    println!(
        "u8={}, u16={}, u32={}, u64={}, u128={}, usize={}",
        u8_val, u16_val, u32_val, u64_val, u128_val, usize_val
    );

    println!("\n整型字面量：");
    println!(
        "decimal={}, hex={}, octal={}, binary={}, separated={}",
        decimal, hex, octal, binary, separated
    );

    println!("\n=浮点型：");
    println!("f32={}, f64={}, inferred={}", f32_val, f64_val, float_infer);

    println!("\n布尔型：");
    println!("bool_true={}, bool_false={}", bool_true, bool_false);

    println!("\n字符型：");
    println!(
        "char_a='{}', char_cn='{}', char_emoji='{}'",
        char_a, char_cn, char_emoji
    );
}
