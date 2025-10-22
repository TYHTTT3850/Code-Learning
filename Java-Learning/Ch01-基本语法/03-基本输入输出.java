import java.util.Scanner;

public class BasicIO {
    public static void main(String[] args) {
        // 创建 Scanner 对象，从标准输入（键盘）读取
        Scanner sc = new Scanner(System.in);

        // 输入字符串
        System.out.print("请输入你的名字：");
        String name = sc.nextLine();

        // 输入整数
        System.out.print("请输入你的年龄：");
        int age = sc.nextInt();

        // 输入浮点数
        System.out.print("请输入你的身高（m）：");
        double height = sc.nextDouble();

        // 输出结果
        System.out.println("\n=== 输出结果 ===");
        System.out.println("姓名：" + name);
        System.out.println("年龄：" + age);
        System.out.println("身高：" + height + "m");

        sc.close(); // 关闭输入流
    }
}
