public class Child extends Parent {

    int x = 20; // 隐藏父类同名成员变量

    void printValues() {
        System.out.println(this.x);        // 20 → 访问子类自己的成员变量
        System.out.println(super.x);  // 10 → 通过 super 访问父类成员变量
        // super 只能在子类内部使用，用于显式指定父类成员
    }
}