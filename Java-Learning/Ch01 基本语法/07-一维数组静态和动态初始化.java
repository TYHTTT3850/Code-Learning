package CodeHere;

public class ArrayExample {
    public static void main(String[] args) {

        // ===== 1. 静态初始化 =====
        // 在定义数组的同时，直接给出所有元素的值
        int[] nums1 = {10, 20, 30, 40, 50};

        System.out.println("静态初始化数组内容：");
        for (int i = 0; i < nums1.length; i++) {
            System.out.println("nums1[" + i + "] = " + nums1[i]);
        }

        System.out.println("----------------------------");

        // ===== 2. 动态初始化 =====
        // 只定义长度，先不赋值，之后再逐个元素赋值
        int[] nums2 = new int[5]; // 默认初始值全为0

        for (int i = 0; i < nums2.length; i++) {
            nums2[i] = (i + 1) * 10; // 动态赋值
        }

        System.out.println("动态初始化数组内容：");
        for (int i = 0; i < nums2.length; i++) {
            System.out.println("nums2[" + i + "] = " + nums2[i]);
        }

        System.out.println("----------------------------");

        // ===== 3. 验证默认值特性 =====
        double[] arrD = new double[3];
        boolean[] arrB = new boolean[3];
        String[] arrS = new String[3];

        System.out.println("不同类型数组的默认值：");
        System.out.println("double 数组默认值: " + arrD[0]);
        System.out.println("boolean 数组默认值: " + arrB[0]);
        System.out.println("String 数组默认值: " + arrS[0]);
    }
}
