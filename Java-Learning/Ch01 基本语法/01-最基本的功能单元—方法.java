public class Method {
    //方法是java最基本的功能单元
    /*
    基本的方法的定义格式：
    public static 返回值类型 方法名称(参数列表){
        方法体
    }
    */
    public static int sum(int a,int b){
		      return a+b;
	   }
    //方法可以重载
    //重载是指同名函数具有不同的参数列表，注意：返回值类型不同不是重载，而是错误
    public static int add(int a, int b) {
        return a + b;
    }

    public static int add(int a, int b, int c) {
        return a + b + c;
    }

    //通过return提前结束方法的执行
    public static int division(int a,int b){
        if(b==0){
            System.out.println("除数不能为0");
            return -1;
        }
        return a/b;
    }

    public static void main(String []args) {
        int s = sum(3,5);
        int d1 = add(1,2);// 重载
        int d2 = add(1,2,3);
        System.out.println(s);
        System.out.println(d1);
        System.out.println(d2);
    }
}