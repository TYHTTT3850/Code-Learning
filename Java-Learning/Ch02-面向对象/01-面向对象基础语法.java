public static class Person{//创建一个类
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
public static void main(String[] args){
    Person p1 = new Person("Alice", 30);//初始化一个对象
    Person p2 = new Person("Bob", 25);
    System.out.println(p1.getName() + " " + p1.getAge());//调用对象的成员方法
    System.out.println(p2.getName() + " " + p2.getAge());
}
