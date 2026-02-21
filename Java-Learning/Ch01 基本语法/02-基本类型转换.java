package CodeHere;

public class BasicTypeCasting {
    public static void main(String[] args) {
        // 自动类型转换（小 → 大）
        // 小和大指的存储时占用的字节大小
        int i = 100;
        double d = i; // int → double
        System.out.println("自动转换 int → double: " + d);

        // 强制类型转换（大 → 小）
        double d2 = 9.78;
        int i2 = (int) d2; // double → int，若不强制转换而是直接赋值会报错！！！
        System.out.println("强制转换 double → int: " + i2);

        // char 与 int 之间
        char c = 'A';
        int ascii = c; // char → int
        System.out.println("字符 'A' 的 ASCII 值: " + ascii);

        int num = 66;
        char letter = (char) num; // int → char
        System.out.println("ASCII 66 对应字符: " + letter);

        // 混合类型运算：int 和 double
        int l = 5;
        double dnum = 2.5;
        double result = l * dnum; // int 自动提升为 double
        System.out.println("int * double → double: " + result);

        // char 与 int 运算
        // 实际上 byte，char，short 运算时会直接提升为 int
        char ch = 'A'; // 65
        int sum = ch + 5; // char → int，再计算
        System.out.println("char + int → int: " + sum);
        System.out.println("对应字符: " + (char) sum);
    }
}

