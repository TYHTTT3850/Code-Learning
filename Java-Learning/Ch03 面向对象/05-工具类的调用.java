public class test{
    public static void main(String[] args){
        int[] arr = {1,2,3,4,5};
        ArrayTools.printArray(arr);//直接用类名调用方法
        double ave = ArrayTools.average(arr);
        System.out.println("平均值是：" + ave);

    }
}

