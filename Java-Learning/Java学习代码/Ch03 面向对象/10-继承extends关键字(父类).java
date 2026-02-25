/*
当类与类之间存在相同的内容，并且子类是父类的一种时。可以考虑继承
继承可以提高代码复用
Java只支持单继承，一个子类只能继承一个父类，不能继承多个父类。
Java支持多层继承，一个子类可以继承一个父类，父类又可以继承另一个父类，以此类推。直接继承的父类称为直接父类，父类的父类叫间接父类。
Java中有顶级父类Object，所有类都直接或间接继承自Object类。
*/
public class Person {
    String name;
    int age;

    public void eat(){
        System.out.println("吃饭");
    }
}
