package CodeHere;

public class AllOperators {
    public static void main(String[] args) {
        //算术运算符
        int a = 10, b = 3;
        System.out.println("=== 算术运算符 ===");
        System.out.println("a + b = " + (a + b));//此时+是连接符，字符串之间的+也是连接符(能算则算，不能算即为连接)
        System.out.println("a - b = " + (a - b));
        System.out.println("a * b = " + (a * b));
        System.out.println("a / b = " + (a / b));  // 整数除法，浮点类型即为正常的除法
        System.out.println("a % b = " + (a % b));  // 取余

        //赋值运算符
        int c = 5;
        System.out.println("\n=== 赋值运算符 ===");
        c += 2; // c = c + 2
        System.out.println("c += 2 -> " + c);
        c *= 3;
        System.out.println("c *= 3 -> " + c);
        c %= 4;
        System.out.println("c %= 4 -> " + c);

        //自增自减运算符
        int x = 5;
        System.out.println("\n=== 自增自减 ===");
        System.out.println("x++ -> " + (x++)); // 先使用再+1
        System.out.println("++x -> " + (++x)); // 先+1再使用
        System.out.println("x-- -> " + (x--));
        System.out.println("--x -> " + (--x));

        //关系（比较）运算符
        System.out.println("\n=== 关系运算符 ===");
        System.out.println("a == b -> " + (a == b));
        System.out.println("a != b -> " + (a != b));
        System.out.println("a > b  -> " + (a > b));
        System.out.println("a < b  -> " + (a < b));
        System.out.println("a >= b -> " + (a >= b));
        System.out.println("a <= b -> " + (a <= b));

        //逻辑运算符
        boolean p = true, q = false;
        System.out.println("\n=== 逻辑运算符 ===");
        System.out.println("p && q -> " + (p && q)); // 与
        System.out.println("p || q -> " + (p || q)); // 或
        System.out.println("!p -> " + (!p));         // 非

        //位运算符
        int m = 5;  // 二进制 0101
        int n = 3;  // 二进制 0011
        System.out.println("\n=== 位运算符 ===");
        System.out.println("m & n -> " + (m & n)); // 与 (0001) -> 1
        System.out.println("m | n -> " + (m | n)); // 或 (0111) -> 7
        System.out.println("m ^ n -> " + (m ^ n)); // 异或 (0110) -> 6
        System.out.println("~m -> " + (~m));       // 取反 (按位取反)
        System.out.println("m << 1 -> " + (m << 1)); // 左移 10
        System.out.println("m >> 1 -> " + (m >> 1)); // 右移 2
        System.out.println("m >>> 1 -> " + (m >>> 1)); // 无符号右移 2

        //条件（三元）运算符
        System.out.println("\n=== 三元运算符 ===");
        int max = (a > b) ? a : b;
        System.out.println("max = (a > b) ? a : b -> " + max);

        //类型运算符 instanceof
        System.out.println("\n=== instanceof 运算符 ===");
        String s = "Hello";
        System.out.println("s instanceof String -> " + (s instanceof String));
        Object obj = s;
        System.out.println("obj instanceof Object -> " + (obj instanceof Object));
        System.out.println("obj instanceof String -> " + (obj instanceof String));
    }
}
