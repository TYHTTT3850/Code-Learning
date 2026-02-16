fn main() {
    // ===== 1. 整数 → 浮点 =====
    let a: i32 = 10;
    let b = a as f64; // 10.0

    // ===== 2. 浮点 → 整数（向零截断）=====
    let c: f64 = 3.9;
    let d = c as i32; // 3

    // ===== 3. 有符号 → 无符号（按位解释）=====
    let e: i8 = -1;
    let f = e as u8; // 255

    // ===== 4. 大位宽 → 小位宽（截断低位保留）=====
    let g: u16 = 300;
    let h = g as u8; // 44

    // ===== 5. bool → 数值 =====
    let t = true as u8;  // 1
    let f2 = false as u8; // 0

    // ===== 6. char → 数值（Unicode 标量值）=====
    let ch = 'A';
    let code = ch as u32; // 65

    println!(
        "{} {} {} {} {} {} {}",
        b, d, f, h, t, f2, code
    );
}