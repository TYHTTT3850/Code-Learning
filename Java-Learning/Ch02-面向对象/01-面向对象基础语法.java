public static class Person{
    String name;
    int age;

    public Person(String name, int age){//类的构造方法
        this.name = name;//this代表当前对象
        this.age = age;
    }

    public String getName(){
        return name;
    }

    public int getAge(){
        return age;
    }
}
public static void main(String[] args){
    Person p1 = new Person("Alice", 30);
    Person p2 = new Person("Bob", 25);
    System.out.println(p1.getName() + " " + p1.getAge());
}