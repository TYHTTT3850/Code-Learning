public class Array2DInitExample {
    public static void main(String[] args) {
        //静态初始化：声明即赋值
        int[][] staticArr = {
                {1, 2, 3},
                {4, 5, 6}
        };

        //动态初始化（规则矩阵）：先分配大小，再赋值
        int[][] dynamicRect = new int[2][3]; // 默认值为 0
        for (int i = 0; i < dynamicRect.length; i++) {
            for (int j = 0; j < dynamicRect[i].length; j++) {
                dynamicRect[i][j] = (i + 1) * 10 + j; // 示例值
            }
        }

        // 直接嵌套循环打印
        System.out.println("staticArr:");
        for (int i = 0; i < staticArr.length; i++) {
            for (int j = 0; j < staticArr[i].length; j++) {
                System.out.print(staticArr[i][j]);
                if (j < staticArr[i].length - 1) System.out.print(" ");
            }
            System.out.println();
        }

        System.out.println("dynamicRect:");
        for (int i = 0; i < dynamicRect.length; i++) {
            for (int j = 0; j < dynamicRect[i].length; j++) {
                System.out.print(dynamicRect[i][j]);
                if (j < dynamicRect[i].length - 1) System.out.print(" ");
            }
            System.out.println();
        }

    }
}
