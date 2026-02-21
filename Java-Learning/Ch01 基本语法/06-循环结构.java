package CodeHere;

public class LoopExample {
    public static void main(String[] args) {

        // ===== 1. for 循环示例 =====
        System.out.println("for 循环示例：");
        for (int i = 1; i <= 5; i++) {
            System.out.println("第 " + i + " 次循环");
        }

        System.out.println("----------------------------");

        // ===== 2. while 循环示例 =====
        System.out.println("while 循环示例：");
        int count = 1;
        while (count <= 5) {
            System.out.println("循环计数：" + count);
            count++;
        }

        System.out.println("----------------------------");

        // ===== 3. do-while 循环示例 =====
        System.out.println("do-while 循环示例：");
        int num = 1;
        do {
            System.out.println("当前数字：" + num);
            num++;
        } while (num <= 5);

        System.out.println("----------------------------");

        // ===== 4. 嵌套循环示例（九九乘法表） =====
        System.out.println("九九乘法表：");
        for (int i = 1; i <= 9; i++) {
            for (int j = 1; j <= i; j++) {
                System.out.print(i + "×" + j + "=" + (i * j) + "\t");
            }
            System.out.println(); // 换行
        }

        System.out.println("----------------------------");

        // ===== 5. break 关键字示例 =====
        System.out.println("break 示例（提前结束循环）：");
        for (int i = 1; i <= 10; i++) {
            if (i == 5) {
                System.out.println("遇到 5，循环提前结束！");
                break; // 终止整个循环
            }
            System.out.println("i = " + i);
        }

        System.out.println("----------------------------");

        // ===== 6. continue 关键字示例 =====
        System.out.println("continue 示例（跳过本次循环）：");
        for (int i = 1; i <= 10; i++) {
            if (i % 2 == 0) {
                continue; // 跳过偶数
            }
            System.out.println("奇数：" + i);
        }

        System.out.println("----------------------------");

        System.out.println("程序执行完毕！");
    }
}