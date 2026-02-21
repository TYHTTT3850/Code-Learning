import java.util.Arrays;

public class test {
    public static void main(String[] args) {
        int[] arr = {1, 2, 3, 4, 5};
        swap(arr);//由于数组是引用类型，所以传递的参数是地址，方法内对数组元素的修改会影响到原数组
        System.out.println(arr[0] + " " + arr[1]);
    }

    public static void swap(int[] array){
        int temp = array[0];
        array[0] = array[1];
        array[1] = temp;
    }
}

