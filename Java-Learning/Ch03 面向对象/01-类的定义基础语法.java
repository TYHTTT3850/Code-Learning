public class Person{//创建一个类
    String name;
    int age;

    public Person(String name, int age){//类的构造方法
        this.name = name;//this代表当前对象
        this.age = age;//成员变量
    }

    public String getName(){//成员方法
        return name;
    }

    public int getAge(){
        return age;
    }
}
