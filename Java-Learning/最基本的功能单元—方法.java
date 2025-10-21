public class Function {
	   //方法是java最基本的功能单元
    public static int sum(int a,int b){
		      return a+b;
	   }
    //方法可以重载
    //重载是指同名函数具有不同的参数列表，注意：返回值类型不同不是重载，而是错误
    public int add(int a, int b) {
        return a + b;
    }

    public int add(int a, int b, int c) {
        return a + b + c;
    }

    public static void main(String []args) {
        int s = sum(3,5);
		      System.out.println(s);
    }
}