public class test {// test的字节码文件进入方法区
    public static void swap(int a, int b){
        int temp = a;
        a = b;
        b = temp;
    }
    public static void main(String[] args) {//main进栈
        int a = 10;
        int b = 20;
        swap(a, b);//swap进栈，执行完出栈，而里面的a和b是swap方法的局部变量，和main方法的a和b没有关系
        System.out.println("a: " + a + ", b: " + b); // a: 10, b: 20
    }
}
