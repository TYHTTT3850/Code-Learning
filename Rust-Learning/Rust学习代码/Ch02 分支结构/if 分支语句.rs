fn main() {
    let number = 7;

    // 基本 if
    if number < 5 {
        println!("number 小于 5");
    }

    // if 作为表达式（返回值）
    let result =
        if number % 2 == 0 {
        "偶数"
        }
        else {
        "奇数"
        };
    println!("number 是 {}", result);

    // 多条件判断
    let score = 85;
    if score >= 90 {
        println!("A");
    }
    else if score >= 80 {
        println!("B");
    }
    else if score >= 60 {
        println!("C");
    }
    else {
        println!("不及格");
    }

    // 注意：{}不能省略，即使 if 中只有单行语句
    //if number % 2 == 0 println!("偶数") // 报错
}