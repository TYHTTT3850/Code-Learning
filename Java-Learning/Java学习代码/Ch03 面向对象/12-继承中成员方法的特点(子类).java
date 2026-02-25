public class ThirdGenerationPhone extends SecondGenerationPhone {
    //注解：给JVM看的。注释：给程序员看的
    @Override//这个注解意思是：这个方法是重写父类的方法，如果没有重写成功，编译器会报错
    public void call() {//方法声明保持一致，方法体可以改变
        System.out.println("利用手机视频通话");
        super.call();//直接调用父类方法，如果是 this 关键字，会先访问本类，在访问父类
    }

    public void playGame() {
        System.out.println("利用手机玩游戏");
    }
}
