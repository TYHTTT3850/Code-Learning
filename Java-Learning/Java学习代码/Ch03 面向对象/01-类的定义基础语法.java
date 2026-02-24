public class Person{//创建一个类，单独定义在Person.java文件中，不和 main 函数写一起
    String name;
    int age;
    /*
    构造方法注意事项：
    1.如果没有定义构造方法，系统给出默认的无参构造方法
    2.如果自己写了构造方法，系统不提供默认的构造方法
    3.养成习惯，都手动写无参构造方法和全参数构造方法
    */
    public Person(){}
    
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
