package CodeHere;

import java.util.Scanner;

public class BranchExample {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);

        // ===== if 分支结构示例 =====
        System.out.print("请输入你的成绩（0-100）：");
        int score = input.nextInt();

        if (score >= 90) {
            System.out.println("成绩等级：优秀");
        } else if (score >= 75) {
            System.out.println("成绩等级：良好");
        } else if (score >= 60) {
            System.out.println("成绩等级：及格");
        } else {
            System.out.println("成绩等级：不及格");
        }

        System.out.println("----------------------------");

        // ===== switch 分支结构示例 =====
        System.out.print("请输入今天是星期几（1-7）：");
        int day = input.nextInt();

        switch (day) {//switch只支持整数判断
            case 1:
                System.out.println("今天是星期一");
                break;
            case 2:
                System.out.println("今天是星期二");
                break;
            case 3:
                System.out.println("今天是星期三");
                break;
            case 4:
                System.out.println("今天是星期四");
                break;
            case 5:
                System.out.println("今天是星期五");
                break;
            case 6:
            case 7:
                System.out.println("今天是周末！");
                break;
            default:
                System.out.println("输入的星期数不合法！");
        }

        input.close();
    }
}

