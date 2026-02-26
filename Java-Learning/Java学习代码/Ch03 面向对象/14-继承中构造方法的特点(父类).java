public class Person {
    String name;
    int age;

    //构造方法
    public Person(){
        System.out.println("父类的空参构造执行了");
    }

    public Person(String name, int age){
        this.name = name;
        this.age = age;
        System.out.println("父类的有参构造执行了");
    }
}
