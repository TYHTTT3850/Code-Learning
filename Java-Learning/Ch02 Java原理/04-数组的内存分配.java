import java.util.Arrays;

public class test {
    public static void main(String[] args) {
        int[] arr = {1, 2, 3, 4, 5};//实际上完整的是int[] arr = new int[]{1, 2, 3, 4, 5}，但是编译器会自动推断出类型，所以可以省略掉new int[]部分
        System.out.println(arr[0]);//输出1
        System.out.println(arr);//输出地址
        //实际上，数组是引用数据类型，存储的是地址，而不是具体的值，真正的值存储在堆内存中。
    }
}