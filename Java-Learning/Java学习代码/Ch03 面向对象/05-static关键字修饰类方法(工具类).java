/*
实际开发中，经常会遇到一些工具方法，每次都重写一遍，比较麻烦，我们可以把这些工具方法封装到一个类中，以后直接调用就行了。
以数组为例
*/
public class ArrayTools{
    //私有化构造方法，禁止外部创建对象，因为我们只调用方法，不想要对象
    private ArrayTools(){}

    //定义方法，静态的，直接调用，不需要创建对象
    public static void printArray(int[] arr){
        System.out.print("[");
        for (int j : arr) {
            System.out.print(j + ",");
        }
        System.out.print("]");
        System.out.println();
    }

    public static double average(int[] arr){
        int sum = 0;
        for (int j : arr) {
            sum += j;
        }
        return (double)sum / arr.length;
    }
}
