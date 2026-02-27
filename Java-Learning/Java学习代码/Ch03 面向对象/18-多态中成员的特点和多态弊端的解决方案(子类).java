public class Child extends Parent {
    String name = "Child";
    int age = 20;
    //重写父类方法
    @Override
    public void show1() {
        System.out.println("Child show1");
    }

    //子类特有方法
    public void childShow() {
        System.out.println("Child show");
    }
}
